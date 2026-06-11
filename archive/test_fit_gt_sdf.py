"""Tests for fit_gt_sdf.py — expressivity diagnostic for the 1-Lip network."""
import numpy as np
import pytest
import torch
import trimesh

from fit_gt_sdf import (
    load_mesh,
    sample_dataset,
    save_loss_plot,
    save_mc_mesh,
    signed_distance_repo_convention,
)
from lip_tracer.config import ModelConfig
from lip_tracer.model import FTheta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sphere_mesh(radius: float = 0.5, subdivisions: int = 3) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def tiny_ftheta(**kwargs) -> FTheta:
    defaults = dict(hidden=32, depth=2, group_size=2, activation="groupsort",
                    input_encoding="identity", multires=4)
    defaults.update(kwargs)
    return FTheta(**defaults)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_signed_distance_sign_convention():
    """Outside points should have positive SDF, inside points negative."""
    mesh = sphere_mesh(radius=0.5)
    outside = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    inside  = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    sdf_out = signed_distance_repo_convention(mesh, outside, chunk=1)
    sdf_in  = signed_distance_repo_convention(mesh, inside,  chunk=1)
    assert sdf_out[0] > 0, f"expected positive outside, got {sdf_out[0]}"
    assert sdf_in[0]  < 0, f"expected negative inside,  got {sdf_in[0]}"


def test_signed_distance_magnitude():
    """A point 0.2 units from the surface of a radius-0.5 sphere should give |SDF| ≈ 0.2."""
    mesh = sphere_mesh(radius=0.5)
    pt = np.array([[0.7, 0.0, 0.0]], dtype=np.float32)  # 0.2 outside
    sdf = signed_distance_repo_convention(mesh, pt, chunk=1)
    assert abs(sdf[0] - 0.2) < 0.02, f"expected ~0.2, got {sdf[0]:.4f}"


def test_sample_dataset_shapes():
    mesh = sphere_mesh()
    data = sample_dataset(mesh, bound=1.0, n_near=50, n_vol=50,
                          near_std=0.02, sdf_chunk=100, seed=0)
    assert data["near"].shape == (50, 3)
    assert data["near_sdf"].shape == (50,)
    assert data["vol"].shape  == (50, 3)
    assert data["vol_sdf"].shape  == (50,)


def test_sample_dataset_vol_in_bounds():
    mesh = sphere_mesh()
    bound = 0.8
    data = sample_dataset(mesh, bound=bound, n_near=100, n_vol=200,
                          near_std=0.01, sdf_chunk=100, seed=1)
    assert (np.abs(data["vol"]) <= bound + 1e-5).all()


def test_ftheta_output_shape():
    f = tiny_ftheta()
    x = torch.randn(16, 3)
    out = f(x)
    assert out.shape == (16,), f"expected (16,), got {out.shape}"


def test_ftheta_nact_activation():
    f = tiny_ftheta(activation="nact")
    x = torch.randn(8, 3)
    out = f(x)
    assert out.shape == (8,)


def test_ftheta_groupsort_n():
    f = tiny_ftheta(hidden=32, group_size=4)
    x = torch.randn(8, 3)
    assert f(x).shape == (8,)


@pytest.mark.parametrize("multires", [0, 2, 4])
def test_ftheta_neus_encoding(multires):
    f = tiny_ftheta(hidden=64, input_encoding="neus", multires=multires)
    x = torch.randn(8, 3)
    assert f(x).shape == (8,)


# ---------------------------------------------------------------------------
# Integration test: can the network actually fit a sphere SDF?
# ---------------------------------------------------------------------------

def test_fit_sphere_sdf_loss_decreases():
    """FTheta loss should drop significantly after a short fit on a sphere SDF."""
    torch.manual_seed(0)
    np.random.seed(0)

    mesh = sphere_mesh(radius=0.5, subdivisions=3)
    data = sample_dataset(mesh, bound=1.2, n_near=2000, n_vol=2000,
                          near_std=0.02, sdf_chunk=500, seed=0)

    device = "cpu"
    near     = torch.from_numpy(data["near"]).to(device)
    near_sdf = torch.from_numpy(data["near_sdf"]).to(device)
    vol      = torch.from_numpy(data["vol"]).to(device)
    vol_sdf  = torch.from_numpy(data["vol_sdf"]).to(device)

    f   = tiny_ftheta(hidden=64, depth=4).to(device)
    opt = torch.optim.Adam(f.parameters(), lr=1e-3)

    def eval_loss():
        with torch.no_grad():
            x = torch.cat([near, vol])
            y = torch.cat([near_sdf, vol_sdf])
            return torch.nn.functional.l1_loss(f(x), y).item()

    loss_before = eval_loss()

    for _ in range(300):
        ni = torch.randint(0, len(near), (256,))
        vi = torch.randint(0, len(vol),  (256,))
        x  = torch.cat([near[ni], vol[vi]])
        y  = torch.cat([near_sdf[ni], vol_sdf[vi]])
        loss = torch.nn.functional.l1_loss(f(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
        opt.step()

    loss_after = eval_loss()
    assert loss_after < loss_before * 0.5, (
        f"loss did not decrease enough: {loss_before:.4f} → {loss_after:.4f}"
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def test_save_loss_plot(tmp_path):
    history = [(0, 0.5, 0.4, 0.6), (100, 0.3, 0.25, 0.35), (200, 0.1, 0.08, 0.12)]
    out = tmp_path / "loss.png"
    save_loss_plot(history, out)
    assert out.exists() and out.stat().st_size > 0


def test_save_mc_mesh(tmp_path):
    f = tiny_ftheta(hidden=32, depth=2)
    out = tmp_path / "mesh.ply"
    save_mc_mesh(f, out, bound=1.0, res=16, device="cpu")
    # mesh may or may not have a zero crossing — just check no crash
    # and if written it's a valid file
    if out.exists():
        assert out.stat().st_size > 0


def test_load_mesh(tmp_path):
    mesh = sphere_mesh()
    path = tmp_path / "sphere.ply"
    mesh.export(str(path))
    loaded = load_mesh(path)
    assert isinstance(loaded, trimesh.Trimesh)
    assert len(loaded.vertices) > 0
    assert len(loaded.faces) > 0

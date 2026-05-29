"""Minimal always-on profiling for ICLR-quality compute/memory breakdown.

Drops into the training loop with three call sites:
    1. dump_static_accounting(...) once at startup → params.csv, flops_analytical.csv
    2. with prof.timed("name"): ...                → per-phase ms + peak MB
    3. prof.end_step(step)                         → flushes phases.csv every N steps

CUDA-event timing + reset_peak_memory_stats add ~tens of µs per phase,
so this is cheap enough to leave on for the full 300k-step run.
"""
from __future__ import annotations

import csv
import pickle
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn


# ─── static accounting (run-once at startup) ────────────────────────────────

def count_params(module: nn.Module) -> dict[str, int]:
    """Params grouped by submodule type (PE, hidden blocks aggregated, head).

    Walks one level deep; if a child is itself a Sequential/ModuleList, aggregates
    its children by type (e.g. all ConvexPotentialLayer → one row).
    """
    groups: dict[str, int] = {}
    for name, child in module.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n == 0:
            groups[name] = 0   # PE has no params but worth reporting
            continue
        sub = list(child.named_children())
        if isinstance(child, (nn.Sequential, nn.ModuleList)) and len(sub) > 0:
            by_type: dict[str, list[int]] = {}
            for _, sc in sub:
                k = type(sc).__name__
                p = sum(pp.numel() for pp in sc.parameters())
                by_type.setdefault(k, []).append(p)
            for k, lst in by_type.items():
                groups[f"{name}.{k}×{len(lst)}"] = sum(lst)
        else:
            groups[name] = n
    groups["__total__"] = sum(p.numel() for p in module.parameters())
    return groups


def analytical_flops(model_cfg, train_cfg, trace_cfg) -> dict[str, float]:
    """Per-step FLOPs derived from config (hardware-independent).

    Uses 2·d_in·d_out per Linear (mul + add). Avg trace iters is a placeholder
    written here from the measured histogram in StepProfiler.dump_summary().
    """
    pe_extra = 6 * model_cfg.multires if model_cfg.input_encoding == "pe" else 0
    d_in = 3 + pe_extra
    H, D = model_cfg.hidden, model_cfg.depth
    layers = max(D, 2)
    fwd = 2 * d_in * H + 2 * (layers - 2) * H * H + 2 * H * 1
    bwd = 2 * fwd

    B = train_cfg.batch
    idr_n = train_cfg.idr_n_samples if train_cfg.w_idr_mask > 0 else 0
    n_eik = train_cfg.n_eik_vol if train_cfg.w_eikonal > 0 else 0
    n_msdf = train_cfg.n_mvs_sdf if train_cfg.w_mvs_sdf > 0 else 0

    return {
        "mlp_fwd_FLOPs_per_ray":            fwd,
        "mlp_bwd_FLOPs_per_ray":            bwd,
        "trace_primary_FLOPs_per_step":     B * trace_cfg.iters * fwd,        # worst-case
        "idr_refine_FLOPs_per_step":        B * idr_n * fwd,
        "eikonal_vol_FLOPs_per_step":       n_eik * 4.0 * fwd,                # ~4× fwd for grad
        "mvs_sdf_FLOPs_per_step":           n_msdf * fwd,
        "rays_per_step":                    B,
        "model_depth":                      D,
        "model_hidden":                     H,
        "input_dim_after_PE":               d_in,
    }


def dump_static_accounting(out_dir: Path, model: nn.Module,
                           model_cfg, train_cfg, trace_cfg) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    params = count_params(model)
    with (out_dir / "params.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["group", "params"])
        for k, v in params.items():
            w.writerow([k, v])
    flops = analytical_flops(model_cfg, train_cfg, trace_cfg)
    with (out_dir / "flops_analytical.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["metric", "value"])
        for k, v in flops.items():
            w.writerow([k, f"{v:.6g}"])


# ─── per-step profiler (always-on) ──────────────────────────────────────────

class StepProfiler:
    """Times each phase with CUDA events; tracks peak-MB per phase.

    Use as:
        prof = StepProfiler(run_dir / "profile")
        for step in ...:
            with prof.timed("trace"):  ...
            with prof.timed("ncc"):    ...
            prof.end_step(step)
    """

    def __init__(self, out_dir: Path, flush_every: int = 1000, rays_per_step: int = 0):
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.rays_per_step = rays_per_step
        self.cuda = torch.cuda.is_available()
        self.csv_path = self.out_dir / "phases.csv"
        self._wrote_header = self.csv_path.exists()
        self._time_ms: dict[str, list[float]] = defaultdict(list)
        self._peak_mb: dict[str, list[float]] = defaultdict(list)
        self._step_wall: list[float] = []
        self._wall_t0 = None
        self._iters_hist: list[int] = []     # per-step avg trace iters

    def step_begin(self):
        self._wall_t0 = time.time()

    def step_end(self, step: int):
        if self._wall_t0 is not None:
            self._step_wall.append(time.time() - self._wall_t0)
        if self.flush_every and (step + 1) % self.flush_every == 0:
            self._flush(step + 1)

    def record_trace_iters(self, mean_iters: float):
        self._iters_hist.append(float(mean_iters))

    def timed(self, name: str) -> "_PhaseCtx":
        return _PhaseCtx(self, name)

    def _add(self, name: str, t_ms: float, peak_mb: float):
        self._time_ms[name].append(t_ms)
        self._peak_mb[name].append(peak_mb)

    def _flush(self, step: int):
        names = sorted(self._time_ms.keys())
        row: dict = {"step": step}
        for n in names:
            ts = self._time_ms[n]
            row[f"{n}_ms"] = sum(ts) / len(ts) if ts else 0.0
        for n in names:
            ms = self._peak_mb[n]
            row[f"{n}_peakMB"] = max(ms) if ms else 0.0
        if self._step_wall:
            mean_step = sum(self._step_wall) / len(self._step_wall)
            row["step_wall_ms"] = mean_step * 1000.0
            if self.rays_per_step > 0 and mean_step > 0:
                row["rays_per_sec"] = self.rays_per_step / mean_step
        if self._iters_hist:
            row["avg_trace_iters"] = sum(self._iters_hist) / len(self._iters_hist)
        existing_cols: list[str] = []
        if self._wrote_header:
            with self.csv_path.open("r") as fh:
                existing_cols = next(csv.reader(fh), [])
        cols = existing_cols if existing_cols else list(row.keys())
        # extend cols if new keys appeared
        for k in row.keys():
            if k not in cols:
                cols.append(k)
        # rewrite header if columns expanded
        if cols != existing_cols:
            rows: list[list[str]] = []
            if self.csv_path.exists() and existing_cols:
                with self.csv_path.open("r") as fh:
                    r = csv.reader(fh); next(r, None)
                    rows = list(r)
            with self.csv_path.open("w", newline="") as fh:
                w = csv.writer(fh); w.writerow(cols)
                for old in rows:
                    old += [""] * (len(cols) - len(old))
                    w.writerow(old)
            self._wrote_header = True
        with self.csv_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writerow({k: row.get(k, "") for k in cols})
        self._time_ms.clear(); self._peak_mb.clear()
        self._step_wall.clear(); self._iters_hist.clear()


class _PhaseCtx:
    __slots__ = ("p", "name", "_s", "_e", "_t0")
    def __init__(self, p: StepProfiler, name: str):
        self.p, self.name = p, name
    def __enter__(self):
        if self.p.cuda:
            torch.cuda.reset_peak_memory_stats()
            self._s = torch.cuda.Event(enable_timing=True)
            self._e = torch.cuda.Event(enable_timing=True)
            self._s.record()
        else:
            self._t0 = time.time()
        return self
    def __exit__(self, *_):
        if self.p.cuda:
            self._e.record(); self._e.synchronize()
            t_ms = self._s.elapsed_time(self._e)
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            t_ms = (time.time() - self._t0) * 1000.0
            peak_mb = 0.0
        self.p._add(self.name, t_ms, peak_mb)


# ─── one-shot CUDA memory snapshot (peak-step pickle) ───────────────────────

class MemorySnapshot:
    """Record allocator history around one step, dump pickle for memory_viz."""

    def __init__(self, out_dir: Path, at_step: int = 200):
        self.out_dir = Path(out_dir)
        self.at_step = at_step
        self._recording = False
        self._done = False

    def tick(self, step: int):
        if self._done or not torch.cuda.is_available():
            return
        if step == self.at_step and not self._recording:
            try:
                torch.cuda.memory._record_memory_history(max_entries=100_000)
                self._recording = True
            except Exception:
                self._done = True
        elif self._recording and step == self.at_step + 1:
            try:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                snap = torch.cuda.memory._snapshot()
                with (self.out_dir / "mem.pickle").open("wb") as fh:
                    pickle.dump(snap, fh)
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
            finally:
                self._done = True

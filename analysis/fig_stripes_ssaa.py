"""ICLR figure: surface striations are baked-in SDF texture, not sampling aliasing.

Evidence from the --alias-diag SSAA sweep of run_20260602_220736_scan118_occ_4954645
(view 50): box-filtered 64-spp rendering removes the sub-pixel chart speckle
(wavelength < 6 px) but leaves the stripe band (10-50 px) untouched, so the
striations are geometry of f_theta and must be fixed at training time.

Output: figures/fig_scan118_stripes_ssaa.png (+ .pdf)
"""

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RUN = "outputs/run_20260602_220736_scan118_occ_4954645/st_buffers_view50"
CROP = (220, 560, 476, 816)          # 256x256 on-surface patch (cherub base/hands)
PX_MM = 0.3798                       # median adjacent-pixel hit distance, DTU mm
SCALE = 313.87958                    # scan118 scale_mat -> normalized units
LAM_PE_PX = (2 * np.pi / 32) * SCALE / PX_MM   # top PE band sin(2^5 x): ~162 px
LAM_NCC_PX = 4.0                     # NCC patch footprint: 5x5 samples over +-2 px

SS_LEVELS = [1, 2, 4, 8]


def load_patch(ss):
    img = Image.open(f"{RUN}/view50_ss{ss}_normals.png").convert("RGB")
    a = np.asarray(img, np.float32) / 255.0
    x0, y0, x1, y1 = CROP
    return a[y0:y1, x0:x1]


def shade(rgb):
    """Grayscale relief from an RGB-encoded normal map (white bg preserved).

    Light = patch mean normal tilted 45 deg, so gentle ripples modulate the
    diffuse term strongly regardless of the buffer's normal convention.
    """
    n = rgb * 2.0 - 1.0
    bg = (rgb > 0.98).all(-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    nm = n[~bg].mean(0)
    nm /= np.linalg.norm(nm)
    t = np.cross(nm, [0.0, 0.0, 1.0])
    t /= np.linalg.norm(t)
    l = nm * np.cos(np.deg2rad(45)) + t * np.sin(np.deg2rad(45))
    s = np.clip(n @ l, 0, 1)
    s = 0.10 + 0.90 * s
    s[bg] = 1.0
    return s


def radial_spectrum(gray):
    p = gray - gray.mean()
    w = np.hanning(p.shape[0])
    F = np.abs(np.fft.fftshift(np.fft.fft2(p * w[:, None] * w[None, :])))
    h = p.shape[0]
    y, x = np.mgrid[-h // 2 : h // 2, -h // 2 : h // 2]
    r = np.hypot(x, y).astype(int)
    rad = np.bincount(r.ravel(), F.ravel()) / np.bincount(r.ravel())
    return rad / h**2  # amplitude per pixel


def band_rms(gray, lo, hi):
    p = gray - gray.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(p)))
    h = p.shape[0]
    y, x = np.mgrid[-h // 2 : h // 2, -h // 2 : h // 2]
    r = np.hypot(x, y)
    m = (r >= lo) & (r < hi)
    return np.sqrt((F[m] ** 2).sum()) / h**2


def main():
    patches = {ss: load_patch(ss) for ss in SS_LEVELS}
    grays = {ss: p.mean(-1) for ss, p in patches.items()}
    h = grays[1].shape[0]

    # band boundaries in radial-frequency bins (wavelength = h / r)
    stripe = (h / 50, h / 10)   # wavelength 10-50 px
    speckle = (h / 6, h / 2)    # wavelength 2-6 px
    rms = {
        name: [band_rms(grays[ss], lo, hi) for ss in SS_LEVELS]
        for name, (lo, hi) in [("stripe", stripe), ("speckle", speckle)]
    }
    spec1, spec8 = radial_spectrum(grays[1]), radial_spectrum(grays[8])

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
    })

    fig = plt.figure(figsize=(5.5, 3.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.55,
                          wspace=0.32, left=0.09, right=0.985, top=0.94,
                          bottom=0.11)

    # --- row 1: image evidence -------------------------------------------
    titles = ["(a) 1 spp", "(b) 64 spp ($8{\\times}8$ SSAA)",
              "(c) difference $|$(a)$-$(b)$|\\times 8$"]
    sh1, sh8 = shade(patches[1]), shade(patches[8])
    diff = np.abs(grays[1] - grays[8])
    imgs = [sh1, sh8, 1.0 - np.clip(diff * 8, 0, 1)]
    for i, (im, t) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(t, pad=3)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
    # scale bar: 50 px in mm
    import matplotlib.patheffects as pe
    ax = fig.axes[0]
    bar = 50
    ax.plot([12, 12 + bar], [h - 14, h - 14], "w-", lw=2,
            path_effects=[pe.Stroke(linewidth=3, foreground="k"),
                          pe.Normal()])
    ax.text(12 + bar / 2, h - 22, f"{bar * PX_MM:.0f} mm", ha="center",
            va="bottom", fontsize=6, color="w",
            path_effects=[pe.Stroke(linewidth=1.5, foreground="k"),
                          pe.Normal()])

    # --- (d) band amplitude vs spp ---------------------------------------
    axd = fig.add_subplot(gs[1, 0])
    spp = [s * s for s in SS_LEVELS]
    axd.plot(spp, rms["stripe"], "o-", c="#c1272d", lw=1.2, ms=3.5)
    axd.plot(spp, rms["speckle"], "s-", c="#0072bd", lw=1.2, ms=3.5)
    axd.set_xscale("log")
    axd.set_xticks(spp, [str(s) for s in spp])
    axd.set_xlabel("samples / pixel", labelpad=1.5)
    axd.set_ylabel("band rms", labelpad=1.5)
    axd.set_ylim(0, 0.055)
    axd.set_title("(d) supersampling response", pad=3)
    axd.text(8, 0.044, "stripes ($\\lambda$ 10–50 px)", ha="center",
             fontsize=6.5, color="#c1272d")
    axd.text(8, 0.0125, "speckle ($\\lambda<6$ px)", ha="center",
             fontsize=6.5, color="#0072bd")
    axd.annotate("$-1\\%$", (spp[-1], rms["stripe"][-1]), xytext=(-2, -9),
                 textcoords="offset points", ha="right", fontsize=6.5,
                 color="#c1272d")
    axd.annotate("$-26\\%$", (spp[-1], rms["speckle"][-1]), xytext=(-2, 5),
                 textcoords="offset points", ha="right", fontsize=6.5,
                 color="#0072bd")

    # --- (e) radial spectrum ---------------------------------------------
    axe = fig.add_subplot(gs[1, 1:])
    rbins = np.arange(len(spec1))
    lam = h / np.maximum(rbins, 1e-9)
    sel = (rbins >= 2) & (lam >= 2)
    axe.loglog(lam[sel], spec1[sel], c="#c1272d", lw=1.0, label="1 spp")
    axe.loglog(lam[sel], spec8[sel], c="#0072bd", lw=1.0, ls="--",
               label="64 spp")
    axe.axvspan(10, 50, color="#c1272d", alpha=0.10, lw=0)
    axe.axvspan(2, 6, color="#0072bd", alpha=0.10, lw=0)
    axe.set_xlim(230, 2)
    axe.set_ylim(2.5e-5, 5e-3)
    axe.text(0.50, 0.83, "stripes: SSAA-invariant\n(baked into $f_\\theta$)",
             fontsize=6.5, ha="center", color="#7a1418",
             transform=axe.transAxes)
    axe.text(0.88, 0.96, "aliasing:\nremoved\nby SSAA", fontsize=6.5,
             ha="center", va="top", color="#074b7a", transform=axe.transAxes)
    axe.axvline(LAM_PE_PX, color="0.4", lw=0.8, ls=":")
    axe.text(LAM_PE_PX * 0.94, 3.2e-5, "top PE band", rotation=90,
             fontsize=6, va="bottom", ha="right", color="0.35")
    axe.axvline(LAM_NCC_PX, color="0.4", lw=0.8, ls=":")
    axe.text(LAM_NCC_PX * 1.12, 3.2e-5, "NCC patch", rotation=90, fontsize=6,
             va="bottom", ha="right", color="0.35")
    axe.set_xlabel("wavelength (px)", labelpad=1.5)
    axe.set_ylabel("amplitude", labelpad=1.5)
    axe.set_title("(e) radial spectrum of the normal map", pad=22)
    secax = axe.secondary_xaxis(
        "top", functions=(lambda l: l * PX_MM, lambda m: m / PX_MM))
    secax.set_xlabel("wavelength (mm)", labelpad=2, fontsize=6.5)
    secax.tick_params(labelsize=6)
    axe.legend(frameon=False, loc="lower left", handlelength=1.6,
               borderaxespad=0.2)

    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig_scan118_stripes_ssaa.{ext}", dpi=600)
    print("wrote figures/fig_scan118_stripes_ssaa.png")
    for name in rms:
        print(name, np.round(rms[name], 4))


if __name__ == "__main__":
    main()

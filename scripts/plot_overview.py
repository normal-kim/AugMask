"""README overview figure: what augmentation and loss masking do (Beijing, MCAR, 50% missing).

    python scripts/plot_overview.py    # writes assets/overview_{light,dark}.png
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from augmask import LGBMSAugmenter, MeanModeFill  # noqa: E402
from augmask.data import build_scenario, split_indices  # noqa: E402
from augmask.data.datasets import load_dataset  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", line="#d8d7d2",
                  observed="#2a78d6", filled="#eb6834", off="#e6e5e0"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", line="#3a3a38",
                 observed="#3987e5", filled="#d95926", off="#2e2e2c"),
}
X, Y = "TEMP", "DEWP"


def panels(n_show=2500):
    """(values, observed) of the two columns: complete data, mean-filled and LGBM-S-filled training rows."""
    df, schema = load_dataset("beijing")
    split = split_indices(len(df), seed=0)
    aug = build_scenario(df, schema, split, "mcar", 0.5, seed=0, augmenter=LGBMSAugmenter())
    mean = build_scenario(df, schema, split, "mcar", 0.5, seed=0, augmenter=MeanModeFill())
    cols = [schema.num_block.index(c) for c in (X, Y)]
    observed = aug.encoded["train"].M_num[:, cols]
    to_data = lambda scn: scn.num_tf.inverse_transform(scn.filled["train"][0])[:, cols]
    complete = df.iloc[split["train"]][[X, Y]].to_numpy()
    rows = np.random.default_rng(0).choice(len(complete), n_show, replace=False)
    return rows, {
        "Complete data": (complete, np.ones_like(observed)),
        "Incomplete data (mean fill)": (to_data(mean), observed),
        "Incomplete data (augmented fill)": (to_data(aug), observed),
    }


def grid(ax, x0, mask, color, t, title, note, crossed=False):
    """A small table of cells: observed cells in blue, the others in ``color``."""
    n_rows, n_cols = mask.shape
    for i in range(n_rows):
        for j in range(n_cols):
            face = t["observed"] if mask[i, j] else color
            ax.add_patch(Rectangle((x0 + j, n_rows - 1 - i), 0.9, 0.9, facecolor=face, edgecolor="none"))
            if crossed and not mask[i, j]:
                ax.text(x0 + j + 0.45, n_rows - 1 - i + 0.45, "×", ha="center", va="center", color=t["muted"], fontsize=11)
    ax.text(x0 + n_cols / 2 - 0.05, n_rows + 0.5, title, ha="center", va="bottom", color=t["text"], fontsize=11, weight="bold")
    ax.text(x0 + n_cols / 2 - 0.05, -0.45, note, ha="center", va="top", color=t["muted"], fontsize=10, linespacing=1.4)


def draw(rows, data, t, path):
    plt.rcParams.update({"font.size": 10, "text.color": t["text"], "axes.labelcolor": t["muted"],
                         "xtick.color": t["muted"], "ytick.color": t["muted"]})
    W, H, side, gap, top, below = 10.0, 6.0, 2.0, 0.55, 1.05, 2.2  # inches; the panel row spans the schematic
    left = (W - 3 * side - 2 * gap) / 2
    fig = plt.figure(figsize=(W, H), facecolor=t["surface"])
    fig.suptitle("Why constant fills fail, and what AugMask does instead", fontsize=14, weight="bold",
                 color=t["text"], y=1 - 0.2 / H)
    lim, yticks = {X: (-20, 42), Y: (-30, 30)}, [-30, -15, 0, 15, 30]
    for k, (title, (values, obs)) in enumerate(data.items()):
        ax = fig.add_axes([(left + k * (side + gap)) / W, (H - top - side) / H, side / W, side / H],
                          facecolor=t["surface"])
        v, o = values[rows], obs[rows].all(1)
        ax.scatter(*v[o].T, s=4, color=t["observed"], alpha=0.45, linewidths=0, label="both values observed (complete)")
        if (~o).any():
            ax.scatter(*v[~o].T, s=4, color=t["filled"], alpha=0.6, linewidths=0, label="at least one value filled (incomplete)")
        ax.set(xlim=lim[X], ylim=lim[Y], yticks=yticks, xlabel="Temperature (°C)",
               ylabel="Dew point (°C)" if k == 0 else None)
        ax.set_title(title, color=t["text"], fontsize=10.5)
        for spine in ax.spines.values():
            spine.set_color(t["line"])
        ax.tick_params(length=0, labelsize=9)
        if k == 1:
            legend = fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1 - 0.5 / H),
                                ncol=2, frameon=False, markerscale=3, labelcolor=t["text"], fontsize=9.5)
            for handle in legend.legend_handles:
                handle.set_alpha(1)

    ax = fig.add_axes([0, 0.03 / H, 1, below / H], facecolor=t["surface"])
    ax.set_axis_off()
    mask = np.array([[1, 0, 1, 1, 0, 1], [0, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0], [1, 0, 1, 1, 1, 1]], bool)
    grid(ax, 0, mask, t["filled"], t, "Inputs: missing entries augmented",
         "sampled by LGBM-S given the\nobserved values and the mask")
    grid(ax, 14, mask, t["off"], t, "Loss: observed entries only",
         "sampled values are context,\nnever targets", crossed=True)
    ax.add_patch(FancyArrowPatch((6.6, 2), (13.4, 2), arrowstyle="-|>", mutation_scale=18, color=t["muted"], lw=1.5))
    ax.text(10.0, 2.35, "denoiser", ha="center", va="bottom", color=t["muted"], fontsize=10)
    ax.set(xlim=(-4.0, 24.0), ylim=(-2.6, 5.4), aspect="equal")
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def main():
    rows, data = panels()
    out = Path("assets")
    out.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        draw(rows, data, theme, out / f"overview_{name}.png")


if __name__ == "__main__":
    main()

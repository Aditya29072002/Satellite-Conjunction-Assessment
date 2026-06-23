import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── Unified style ─────────────────────────────────────────────────────────────
STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}
plt.rcParams.update(STYLE)

PALETTE = {
    "naive": "#d62728",
    "grid":  "#2ca02c",
    "kdtree":"#1f77b4",
    "accent":"#ff7f0e",
    "ideal": "#888888",
}

# ── Real measured data (from your runs) ───────────────────────────────────────
CORES      = [1, 2, 4, 8, 16]
# Phase 8 real-workload strong scaling:
TIME_REAL  = [1.0821, 0.5605, 0.2940, 0.2476, 0.2045]
SPEEDUP    = [1.00, 1.93, 3.68, 4.37, 5.29]
EFFIC      = [100.0, 96.5, 92.0, 54.6, 33.1]

def fig_scaling_polished():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Parallel Strong Scaling on Real Time-Window Screening Workload",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(CORES, SPEEDUP, "o-", color=PALETTE["accent"], lw=2.2,
            ms=8, label="Measured")
    ax.plot(CORES, CORES, "--", color=PALETTE["ideal"], lw=1.5, label="Ideal linear")
    ax.fill_between(CORES, SPEEDUP, CORES, color=PALETTE["accent"], alpha=0.08)
    ax.set_xlabel("Number of CPU Cores")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Cores")
    ax.set_xticks(CORES)
    ax.legend(frameon=True)

    ax = axes[1]
    bars = ax.bar([str(c) for c in CORES], EFFIC, color=PALETTE["kdtree"],
                  edgecolor="white", width=0.65)
    ax.axhline(100, color=PALETTE["naive"], ls="--", lw=1.5, label="Ideal 100%")
    for bar, v in zip(bars, EFFIC):
        ax.text(bar.get_x()+bar.get_width()/2, v+2, f"{v:.0f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Number of CPU Cores")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Efficiency vs Cores")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig("outputs/polished_scaling.png")
    plt.close()
    print("Saved: outputs/polished_scaling.png")

def fig_complexity_polished():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Algorithmic Complexity: Naive vs Spatial Partitioning",
                 fontsize=13, fontweight="bold")
    Ns = np.array([100, 500, 1000, 2000, 5000, 10000, 15830, 25000])
    naive = Ns*(Ns-1)//2
    smart = Ns*130

    ax = axes[0]
    ax.plot(Ns, naive/1e6, "o-", color=PALETTE["naive"], lw=2.2, ms=6,
            label="Naive  O(N$^2$)")
    ax.plot(Ns, smart/1e6, "s-", color=PALETTE["grid"], lw=2.2, ms=6,
            label="Spatial  O(N·k)")
    ax.axvline(15830, color="#555", ls=":", lw=1.5)
    ax.text(15830, ax.get_ylim()[1]*0.5, " full\n catalog", fontsize=8, color="#555")
    ax.set_xlabel("Number of Objects (N)")
    ax.set_ylabel("Comparisons (millions)")
    ax.set_title("Linear Scale")
    ax.legend(frameon=True)

    ax = axes[1]
    ax.semilogy(Ns, naive, "o-", color=PALETTE["naive"], lw=2.2, ms=6,
                label="Naive  O(N$^2$)")
    ax.semilogy(Ns, smart, "s-", color=PALETTE["grid"], lw=2.2, ms=6,
                label="Spatial  O(N·k)")
    ax.axvline(15830, color="#555", ls=":", lw=1.5)
    ax.set_xlabel("Number of Objects (N)")
    ax.set_ylabel("Comparisons (log scale)")
    ax.set_title("Logarithmic Scale")
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig("outputs/polished_complexity.png")
    plt.close()
    print("Saved: outputs/polished_complexity.png")

def fig_keyresults():
    """One combined 'money figure' summarizing the whole project."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Key Results Summary", fontsize=14, fontweight="bold")

    # Panel 1: operation reduction at full catalog
    ax = axes[0]
    labels = ["Naive\nO(N²)", "Spatial\nO(N·k)"]
    vals = [125_286_535, 15830*130]
    bars = ax.bar(labels, vals, color=[PALETTE["naive"], PALETTE["grid"]],
                  edgecolor="white", width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Pairwise comparisons")
    ax.set_title("Full Catalog (N=15,830)\n61× Fewer Comparisons")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v*1.4, f"{v:,}",
                ha="center", fontsize=8, fontweight="bold")

    # Panel 2: speedup
    ax = axes[1]
    ax.plot(CORES, SPEEDUP, "o-", color=PALETTE["accent"], lw=2.2, ms=8,
            label="Measured")
    ax.plot(CORES, CORES, "--", color=PALETTE["ideal"], label="Ideal")
    ax.set_xlabel("CPU Cores"); ax.set_ylabel("Speedup")
    ax.set_title("Parallel Speedup\n(peak 5.3× at 16 cores)")
    ax.set_xticks(CORES); ax.legend(frameon=True)

    # Panel 3: correctness
    ax = axes[2]
    ax.axis("off")
    txt = (
        "VALIDATION\n"
        "─────────────────────────\n\n"
        "Full-catalog agreement:\n"
        "  KD-Tree: 211 pairs\n"
        "  Grid:    211 pairs\n"
        "  Disagreement: 0\n\n"
        "Parallel correctness:\n"
        "  Detections identical\n"
        "  across 1–16 cores\n"
        "  (6,167 every run)\n\n"
        "24h time-window screen:\n"
        "  15 genuine conjunctions\n"
        "  3 critical (<10 km)\n"
        "  closest: 3.59 km"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
            fontfamily="monospace", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#cccccc"))
    ax.set_title("Correctness & Detections")

    plt.tight_layout()
    plt.savefig("outputs/polished_keyresults.png")
    plt.close()
    print("Saved: outputs/polished_keyresults.png")

if __name__ == "__main__":
    print("Generating polished figures...\n")
    fig_scaling_polished()
    fig_complexity_polished()
    fig_keyresults()
    print("\nAll polished figures saved to outputs/")
    
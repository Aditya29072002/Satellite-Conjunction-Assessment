import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import KDTree
import json
from multiprocessing import cpu_count

POS_PATH = "data/positions.npy"
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

def load_data():
    positions = np.load(POS_PATH)
    with open("data/satellites.json") as f:
        sats = json.load(f)
    return positions, sats

# ── Figure 1: Algorithm comparison diagram ───────────────────────────────────
def fig_algorithm_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 1 - Conjunction Assessment Algorithms", fontsize=13)
    panels = [
        ("Naive O(N^2) - All Pairs", "#ffcccc",
         "Every satellite checks every\nother satellite.\n\nAt N=25,000:\n312 million pairs / timestep\n\nComplexity: O(N^2)"),
        ("Smart Grid - Spatial Hashing", "#ccffcc",
         "Satellites grouped by grid cell.\nOnly nearby cells checked.\n\nAt N=25,000:\n~650,000 comparisons\n\nComplexity: O(N.k)"),
    ]
    for ax, (title, color, desc) in zip(axes, panels):
        ax.set_facecolor(color)
        ax.text(0.5, 0.72, title, ha="center", va="center",
                fontsize=13, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.34, desc, ha="center", va="center",
                fontsize=11, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("outputs/fig1_algorithm_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Fig 1 saved")

# ── Figure 2: Complexity projection ───────────────────────────────────────────
def fig_complexity_projection():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 2 - Algorithmic Complexity Comparison", fontsize=13)
    Ns = np.array([100, 500, 1000, 5000, 10000, 25000])
    naive = Ns * (Ns - 1) // 2
    smart = Ns * 130
    axes[0].plot(Ns, naive / 1e6, "o-", color="red", label="Naive O(N^2)", lw=2)
    axes[0].plot(Ns, smart / 1e6, "s-", color="green", label="Grid O(N.k)", lw=2)
    axes[0].axvline(25000, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title("Comparisons Required (Linear Scale)")
    axes[0].set_xlabel("Number of Objects (N)")
    axes[0].set_ylabel("Millions of Comparisons")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].semilogy(Ns, naive, "o-", color="red", label="Naive O(N^2)", lw=2)
    axes[1].semilogy(Ns, smart, "s-", color="green", label="Grid O(N.k)", lw=2)
    axes[1].set_title("Comparisons Required (Log Scale)")
    axes[1].set_xlabel("Number of Objects (N)")
    axes[1].set_ylabel("Comparisons (log scale)")
    axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("outputs/fig2_complexity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Fig 2 saved")

# ── Figure 3: Conjunction risk heatmap ────────────────────────────────────────
def fig_conjunction_heatmap(positions):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 3 - Spatial Distribution of Conjunction Risk", fontsize=13)
    sub = positions[:1500]
    tree = KDTree(sub)
    risk = np.array([len(tree.query_ball_point(p, r=500)) - 1 for p in sub])
    sc1 = axes[0].scatter(sub[:, 0], sub[:, 1], c=risk, cmap="hot_r", s=5, alpha=0.7)
    axes[0].set_title("Conjunction Risk - Equatorial Projection")
    axes[0].set_xlabel("X (km)"); axes[0].set_ylabel("Y (km)")
    plt.colorbar(sc1, ax=axes[0], label="Neighbors within 500 km")
    sc2 = axes[1].scatter(sub[:, 0], sub[:, 2], c=risk, cmap="hot_r", s=5, alpha=0.7)
    axes[1].set_title("Conjunction Risk - Polar Projection")
    axes[1].set_xlabel("X (km)"); axes[1].set_ylabel("Z (km)")
    plt.colorbar(sc2, ax=axes[1], label="Neighbors within 500 km")
    plt.tight_layout()
    plt.savefig("outputs/fig3_conjunction_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Fig 3 saved")

# ── Figure 4: Detected close approaches ───────────────────────────────────────
def fig_close_approaches(positions):
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Figure 4 - Detected Close Approaches", fontsize=13)
    sub = positions[:1000]
    tree = KDTree(sub)
    threshold = 100.0
    pairs = list(tree.query_pairs(threshold))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=3, c="steelblue", alpha=0.4)
    flagged = set()
    for i, j in pairs[:80]:
        ax1.plot([sub[i,0], sub[j,0]], [sub[i,1], sub[j,1]], [sub[i,2], sub[j,2]],
                 color="red", linewidth=1.2, alpha=0.7)
        flagged.update([i, j])
    if flagged:
        fi = list(flagged)
        ax1.scatter(sub[fi,0], sub[fi,1], sub[fi,2], s=20, c="red")
    ax1.set_title(f"3D Close Approaches\n({len(pairs)} pairs < {int(threshold)} km)")

    ax2 = fig.add_subplot(122)
    dists = [np.linalg.norm(sub[i] - sub[j]) for i, j in pairs]
    if dists:
        ax2.hist(dists, bins=30, color="salmon", edgecolor="white")
        ax2.axvline(10, color="red", linestyle="--", label="10 km (critical)")
        ax2.axvline(25, color="orange", linestyle="--", label="25 km (warning)")
        ax2.set_title("Distribution of Close-Approach Distances")
        ax2.set_xlabel("Distance (km)"); ax2.set_ylabel("Count")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No close pairs found\nin this subset",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Distribution of Close-Approach Distances")
    plt.tight_layout()
    plt.savefig("outputs/fig4_close_approaches.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fig 4 saved ({len(pairs)} pairs detected)")

# ── Figure 5: Summary dashboard ───────────────────────────────────────────────
def fig_summary_dashboard(positions):
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("HPC Conjunction Assessment - Summary Dashboard",
                 fontsize=15, color="white", fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    sub = positions[:1000]
    tree = KDTree(sub)
    risk = np.array([len(tree.query_ball_point(p, 300)) - 1 for p in sub])
    alts = np.linalg.norm(sub, axis=1) - 6371

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.set_facecolor("#0d1117")
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:15j]
    R = 6371
    ax1.plot_surface(R*np.cos(u)*np.sin(v), R*np.sin(u)*np.sin(v), R*np.cos(v),
                     color="#1e90ff", alpha=0.2)
    ax1.scatter(sub[:,0], sub[:,1], sub[:,2], c=risk, cmap="YlOrRd", s=3, alpha=0.8)
    ax1.set_title("Orbital Risk Map", color="white")
    ax1.tick_params(colors="white", labelsize=6)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")
    valid = alts[(alts > 100) & (alts < 40000)]
    ax2.hist(valid, bins=40, color="#58a6ff", edgecolor="#0d1117")
    ax2.set_title("Altitude Distribution", color="white")
    ax2.set_xlabel("Altitude (km)", color="white"); ax2.set_ylabel("Count", color="white")
    ax2.tick_params(colors="white")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#161b22")
    ax3.hist(risk, bins=30, color="#f78166", edgecolor="#0d1117")
    ax3.set_title("Conjunction Risk Score", color="white")
    ax3.set_xlabel("Neighbors within 300 km", color="white")
    ax3.tick_params(colors="white")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#161b22")
    Ns = [500, 1000, 2000, 5000, 10000]
    naive_ops = [n*(n-1)//2 for n in Ns]
    smart_ops = [n*130 for n in Ns]
    x = np.arange(len(Ns))
    ax4.bar(x-0.2, naive_ops, 0.4, label="Naive", color="#f78166", edgecolor="#0d1117")
    ax4.bar(x+0.2, smart_ops, 0.4, label="Grid", color="#3fb950", edgecolor="#0d1117")
    ax4.set_xticks(x); ax4.set_xticklabels(Ns, color="white", fontsize=8)
    ax4.set_title("Operations: Naive vs Grid", color="white")
    ax4.set_ylabel("Comparisons", color="white"); ax4.set_yscale("log")
    ax4.legend(facecolor="#161b22", labelcolor="white")
    ax4.tick_params(colors="white")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#161b22")
    cores = list(range(1, cpu_count()+1))
    p = 0.90
    actual = [1/((1-p) + p/c) for c in cores]
    ax5.plot(cores, cores, "--", color="gray", label="Ideal")
    ax5.plot(cores, actual, "o-", color="#d2a8ff", label="Amdahl (p=0.9)", lw=2)
    ax5.set_title("Parallel Speedup Model", color="white")
    ax5.set_xlabel("CPU Cores", color="white"); ax5.set_ylabel("Speedup", color="white")
    ax5.legend(facecolor="#161b22", labelcolor="white")
    ax5.tick_params(colors="white")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#161b22")
    ax6.set_xticks([]); ax6.set_yticks([])
    high_risk = int((risk > risk.mean() + risk.std()).sum())
    stats = (
        f"DATASET SUMMARY\n"
        f"{'-'*28}\n"
        f"Objects analyzed:   {len(sub):,}\n"
        f"Avg neighbors:      {risk.mean():.1f}\n"
        f"High-risk objects:  {high_risk}\n"
        f"Altitude range:     {valid.min():.0f}-{valid.max():.0f} km\n\n"
        f"PERFORMANCE GAIN\n"
        f"{'-'*28}\n"
        f"Naive (N=1000):     {1000*999//2:,} ops\n"
        f"Grid  (N=1000):     {1000*130:,} ops\n"
        f"Reduction:          {(1000*999//2)/(1000*130):.0f}x fewer ops\n\n"
        f"Cores available:    {cpu_count()}"
    )
    ax6.text(0.05, 0.95, stats, transform=ax6.transAxes, fontsize=9.5,
             color="white", va="top", fontfamily="monospace",
             bbox=dict(facecolor="#21262d", edgecolor="#30363d", pad=8))
    ax6.set_title("Key Metrics", color="white")

    plt.savefig("outputs/fig5_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("Fig 5 saved")

if __name__ == "__main__":
    positions, sats = load_data()
    print(f"Generating paper figures for {len(positions)} objects...\n")
    fig_algorithm_diagram()
    fig_complexity_projection()
    fig_conjunction_heatmap(positions)
    fig_close_approaches(positions)
    fig_summary_dashboard(positions)
    print("\nAll figures saved to outputs/")
    import os
    for f in sorted(os.listdir("outputs")):
        print(f"  outputs/{f}")

        
import json
import numpy as np
import time
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timezone
from scipy.spatial import KDTree

LOAD_PATH = "data/satellites.json"
THRESHOLD = 25.0

def propagate_all(sats):
    start = datetime.now(timezone.utc)
    jd, fr = jday(start.year, start.month, start.day,
                  start.hour, start.minute, start.second)
    pts, names = [], []
    for s in sats:
        try:
            rec = Satrec.twoline2rv(s["line1"], s["line2"])
            e, r, v = rec.sgp4(jd, fr)
            if e == 0 and not any(np.isnan(r)):
                pts.append(r); names.append(s["name"])
        except Exception:
            pass
    return np.array(pts), names

def grid_screen(positions, threshold):
    cell_size = threshold * 3
    grid = {}
    for idx, pos in enumerate(positions):
        cell = tuple((pos / cell_size).astype(int))
        grid.setdefault(cell, []).append(idx)
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)]
    pairs = []
    for cell_key, members in grid.items():
        neighbors = []
        for off in offsets:
            nb = (cell_key[0]+off[0], cell_key[1]+off[1], cell_key[2]+off[2])
            if nb in grid:
                neighbors.extend(grid[nb])
        for a in members:
            for b in neighbors:
                if a < b and np.linalg.norm(positions[a]-positions[b]) < threshold:
                    pairs.append((a, b))
    return set(pairs)

if __name__ == "__main__":
    with open(LOAD_PATH) as f:
        sats = json.load(f)

    print(f"Propagating full catalog ({len(sats)} TLEs)...")
    t0 = time.time()
    positions, names = propagate_all(sats)
    print(f"  Propagated {len(positions)} objects in {time.time()-t0:.2f}s\n")

    N = len(positions)
    naive_pairs = N*(N-1)//2
    print(f"Catalog size N = {N:,}")
    print(f"Naive O(N^2) would need {naive_pairs:,} pairs - infeasible\n")

    # KD-Tree on full catalog
    t0 = time.time()
    tree = KDTree(positions)
    kd_pairs = tree.query_pairs(THRESHOLD)
    kd_time = time.time() - t0
    print(f"KD-Tree:  {len(kd_pairs):,} close pairs in {kd_time:.3f}s")

    # Grid on full catalog
    t0 = time.time()
    grid_pairs = grid_screen(positions, THRESHOLD)
    grid_time = time.time() - t0
    print(f"Grid:     {len(grid_pairs):,} close pairs in {grid_time:.3f}s")

    # Agreement check
    kd_set = set(tuple(sorted(p)) for p in kd_pairs)
    gd_set = set(tuple(sorted(p)) for p in grid_pairs)
    agree = len(kd_set & gd_set)
    print(f"\nAgreement: {agree:,} pairs found by BOTH methods")
    print(f"  KD-only: {len(kd_set - gd_set)}  Grid-only: {len(gd_set - kd_set)}")

    # Speedup estimate vs naive (extrapolated)
    print(f"\nAt N={N:,}, spatial methods examine ~{N*130:,} candidate pairs")
    print(f"vs {naive_pairs:,} for naive - a {naive_pairs/(N*130):.0f}x reduction")

    # Visualization
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"Phase 9 - Full Catalog Screening ({N:,} objects)", fontsize=13)

    ax1 = fig.add_subplot(131)
    ax1.scatter(positions[:,0], positions[:,1], s=0.5, alpha=0.3, color="navy")
    ax1.set_title("Full Catalog - Equatorial Projection")
    ax1.set_xlabel("X (km)"); ax1.set_ylabel("Y (km)")
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(132)
    methods = ["Naive\n(infeasible)", "KD-Tree", "Grid"]
    ops = [naive_pairs, N*130, N*130]
    colors = ["red", "blue", "green"]
    bars = ax2.bar(methods, ops, color=colors, edgecolor="white")
    ax2.set_yscale("log")
    ax2.set_title("Comparisons Required (log scale)")
    ax2.set_ylabel("Pairs examined")
    for bar, v in zip(bars, ops):
        ax2.text(bar.get_x()+bar.get_width()/2, v*1.3, f"{v:,}",
                 ha="center", fontsize=8)

    ax3 = fig.add_subplot(133)
    dists = [np.linalg.norm(positions[a]-positions[b]) for a, b in list(kd_pairs)[:2000]]
    if dists:
        ax3.hist(dists, bins=40, color="salmon", edgecolor="white")
        ax3.axvline(10, color="red", linestyle="--", label="10 km")
        ax3.set_title(f"Close-Approach Distances\n({len(kd_pairs):,} total pairs)")
        ax3.set_xlabel("Distance (km)"); ax3.set_ylabel("Count")
        ax3.legend()

    plt.tight_layout()
    plt.savefig("outputs/phase9_fullcatalog.png", dpi=150)
    plt.show()
    print("\nSaved: outputs/phase9_fullcatalog.png")

    
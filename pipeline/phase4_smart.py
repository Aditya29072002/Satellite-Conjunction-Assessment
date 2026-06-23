import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
from scipy.spatial import KDTree

POS_PATH  = "data/positions.npy"
THRESHOLD = 10.0
N_EPOCHS  = 120          # number of time snapshots = the parallel workload
CORE_LIST = [1, 2, 4, 8, 16]   # core counts to test

_POSITIONS = None        # global, set once per worker

def _init_worker(positions):
    global _POSITIONS
    _POSITIONS = positions

def grid_conjunction_count(positions, threshold=THRESHOLD):
    """One full grid-based conjunction screen. Returns number of close pairs."""
    cell_size = threshold * 3
    grid = {}
    for idx, pos in enumerate(positions):
        cell = tuple((pos / cell_size).astype(int))
        grid.setdefault(cell, []).append(idx)
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)]
    count = 0
    for cell_key, members in grid.items():
        neighbors = []
        for off in offsets:
            nb = (cell_key[0]+off[0], cell_key[1]+off[1], cell_key[2]+off[2])
            if nb in grid:
                neighbors.extend(grid[nb])
        for a in members:
            for b in neighbors:
                if a < b and np.linalg.norm(positions[a] - positions[b]) < threshold:
                    count += 1
    return count

def epoch_task(args):
    """Rotate the catalog by a small angle (simulating orbital motion) and screen it."""
    _, angle = args
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotated = _POSITIONS @ R.T
    return grid_conjunction_count(rotated)

def run_parallel(positions, n_workers, n_epochs=N_EPOCHS):
    angles = np.linspace(0.0, 0.1, n_epochs)
    tasks = list(enumerate(angles))
    t0 = time.time()
    if n_workers == 1:
        _init_worker(positions)
        results = [epoch_task(t) for t in tasks]
    else:
        with Pool(n_workers, initializer=_init_worker, initargs=(positions,)) as pool:
            results = pool.map(epoch_task, tasks,
                               chunksize=max(1, n_epochs // (n_workers * 2)))
    return sum(results), time.time() - t0

# ── method comparison (single-snapshot) ───────────────────────────────────────
def naive_vectorized(positions, threshold=THRESHOLD):
    t0 = time.time()
    diff = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    mask = (dists < threshold) & (np.triu(np.ones_like(dists), k=1) > 0)
    return int(mask.sum()), time.time() - t0

def kdtree_check(positions, threshold=THRESHOLD):
    t0 = time.time()
    tree = KDTree(positions)
    pairs = tree.query_pairs(threshold)
    return len(pairs), time.time() - t0

def benchmark_methods(positions):
    sizes = [100, 250, 500, 750, 1000, 1500, 2000]
    res = {"naive": [], "kdtree": [], "grid": []}
    print(f"{'N':>6} | {'Naive':>9} | {'KDTree':>9} | {'Grid':>9}")
    print("-" * 45)
    for n in sizes:
        sub = positions[:n]
        tn = naive_vectorized(sub)[1] if n <= 1000 else None
        tk = kdtree_check(sub)[1]
        t0 = time.time(); grid_conjunction_count(sub); tg = time.time() - t0
        res["naive"].append(tn); res["kdtree"].append(tk); res["grid"].append(tg)
        ns = f"{tn:.4f}s" if tn else "   --   "
        print(f"{n:>6} | {ns:>9} | {tk:.4f}s | {tg:.4f}s")
    return sizes, res

def benchmark_scaling(positions):
    print(f"\nStrong scaling ({N_EPOCHS} epochs, N={len(positions)}):")
    print(f"{'Cores':>6} | {'Time':>9} | {'Speedup':>8} | {'Efficiency':>10}")
    print("-" * 42)
    times, speedups = [], []
    for c in CORE_LIST:
        _, t = run_parallel(positions, c)
        times.append(t)
        s = times[0] / t
        speedups.append(s)
        print(f"{c:>6} | {t:.4f}s | {s:>6.2f}x | {s/c*100:>8.1f}%")
    return CORE_LIST, times, speedups

def plot_all(sizes, res, cores, times, speedups, positions):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase 4 - Smart Grid + Parallel Conjunction Assessment",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    valid = [(s, t) for s, t in zip(sizes, res["naive"]) if t]
    if valid:
        ax.plot(*zip(*valid), "o-", color="red", label="Naive O(N^2)")
    ax.plot(sizes, res["kdtree"], "s-", color="blue", label="KD-Tree")
    ax.plot(sizes, res["grid"], "^-", color="green", label="Grid")
    ax.set_title("Execution Time: All Methods")
    ax.set_xlabel("Satellites (N)"); ax.set_ylabel("Time (s)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cores, speedups, "o-", color="darkorange", linewidth=2, label="Actual")
    ax.plot(cores, cores, "--", color="gray", label="Ideal linear")
    ax.set_title("Strong Scaling Speedup")
    ax.set_xlabel("CPU Cores"); ax.set_ylabel("Speedup")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    effs = [s / c * 100 for s, c in zip(speedups, cores)]
    ax.bar([str(c) for c in cores], effs, color="teal", edgecolor="white")
    ax.axhline(100, color="red", linestyle="--", label="Ideal 100%")
    ax.set_title("Parallel Efficiency (%)")
    ax.set_xlabel("Cores"); ax.set_ylabel("Efficiency (%)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    n_demo = 2000
    naive_ops = n_demo * (n_demo - 1) // 2
    grid_ops = n_demo * 26
    ax.bar(["Naive O(N^2)", "Grid Method"], [naive_ops, grid_ops],
           color=["red", "green"], edgecolor="white")
    ax.set_title(f"Operations at N={n_demo}")
    ax.set_ylabel("Approx. Comparisons"); ax.set_yscale("log")
    for i, v in enumerate([naive_ops, grid_ops]):
        ax.text(i, v * 1.3, f"{v:,}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(2, 3, 5, projection="3d")
    sub = positions[:300]
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=2, c="orange", alpha=0.6)
    ax.set_title("3D Grid Partitioning (sample)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    ax = axes[1, 2]
    ax.plot(cores, times, "o-", color="crimson", linewidth=2)
    ax.set_title("Execution Time vs Cores")
    ax.set_xlabel("Cores"); ax.set_ylabel("Time (s)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/phase4_smart_parallel.png", dpi=150)
    plt.show()
    print("Saved: outputs/phase4_smart_parallel.png")

if __name__ == "__main__":
    positions = np.load(POS_PATH)
    print(f"Loaded {len(positions)} positions | CPUs available: {cpu_count()}\n")
    sizes, res = benchmark_methods(positions)
    cores, times, speedups = benchmark_scaling(positions)
    plot_all(sizes, res, cores, times, speedups, positions)

    
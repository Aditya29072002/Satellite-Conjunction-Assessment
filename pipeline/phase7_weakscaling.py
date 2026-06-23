import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

POS_PATH        = "data/positions.npy"
THRESHOLD       = 10.0
EPOCHS_PER_CORE = 30
CORE_LIST       = [1, 2, 4, 8, 16]
N_RUNS          = 3

_POSITIONS = None

def _init_worker(positions):
    global _POSITIONS
    _POSITIONS = positions

def grid_conjunction_count(positions, threshold=THRESHOLD):
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
    _, angle = args
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotated = _POSITIONS @ R.T
    return grid_conjunction_count(rotated)

def run_weak(positions, n_workers, epochs_per_core):
    total_epochs = n_workers * epochs_per_core   # work grows with cores
    angles = np.linspace(0.0, 0.1, total_epochs)
    tasks = list(enumerate(angles))
    t0 = time.time()
    if n_workers == 1:
        _init_worker(positions)
        [epoch_task(t) for t in tasks]
    else:
        with Pool(n_workers, initializer=_init_worker, initargs=(positions,)) as pool:
            pool.map(epoch_task, tasks, chunksize=epochs_per_core)
    return time.time() - t0

if __name__ == "__main__":
    positions = np.load(POS_PATH)
    print(f"Loaded {len(positions)} positions | CPUs: {cpu_count()}")
    print(f"Weak scaling: {EPOCHS_PER_CORE} epochs/core, avg of {N_RUNS} runs\n")

    avg_times = {}
    for c in CORE_LIST:
        runs = [run_weak(positions, c, EPOCHS_PER_CORE) for _ in range(N_RUNS)]
        avg_times[c] = np.mean(runs)
        print(f"  {c:2d} cores | {c*EPOCHS_PER_CORE:3d} epochs total | "
              f"avg {avg_times[c]:.4f}s")

    base = avg_times[CORE_LIST[0]]
    print("\n" + "="*54)
    print("WEAK SCALING TABLE (paste into paper)")
    print("="*54)
    print(f"{'Cores':>6} | {'Epochs':>7} | {'Time (s)':>9} | {'Weak Eff (%)':>12}")
    print("-"*54)
    weak_eff = []
    for c in CORE_LIST:
        eff = base / avg_times[c] * 100   # ideal = 100% (flat time)
        weak_eff.append(eff)
        print(f"{c:>6} | {c*EPOCHS_PER_CORE:>7} | {avg_times[c]:>9.4f} | {eff:>11.1f}%")
    print("="*54)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Phase 7 - Weak Scaling (constant work per core)", fontsize=13)

    axes[0].plot(CORE_LIST, [avg_times[c] for c in CORE_LIST],
                 "o-", color="crimson", linewidth=2, label="Actual")
    axes[0].axhline(base, color="gray", linestyle="--", label="Ideal (flat)")
    axes[0].set_title("Execution Time vs Cores")
    axes[0].set_xlabel("CPU Cores (work scales with cores)")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_ylim(0, max(avg_times.values())*1.3)
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar([str(c) for c in CORE_LIST], weak_eff,
                color="teal", edgecolor="white")
    axes[1].axhline(100, color="red", linestyle="--", label="Ideal 100%")
    axes[1].set_title("Weak Scaling Efficiency")
    axes[1].set_xlabel("CPU Cores"); axes[1].set_ylabel("Efficiency (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/phase7_weakscaling.png", dpi=150)
    plt.show()
    print("Saved: outputs/phase7_weakscaling.png")

    
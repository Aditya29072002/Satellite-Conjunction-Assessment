import numpy as np
import time
from multiprocessing import Pool, cpu_count

POS_PATH  = "data/positions.npy"
THRESHOLD = 10.0
N_EPOCHS  = 120
CORE_LIST = [1, 2, 4, 8, 16]
N_RUNS    = 3

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

def run_parallel(positions, n_workers, n_epochs=N_EPOCHS):
    angles = np.linspace(0.0, 0.1, n_epochs)
    tasks = list(enumerate(angles))
    t0 = time.time()
    if n_workers == 1:
        _init_worker(positions)
        [epoch_task(t) for t in tasks]
    else:
        with Pool(n_workers, initializer=_init_worker, initargs=(positions,)) as pool:
            pool.map(epoch_task, tasks, chunksize=max(1, n_epochs // (n_workers * 2)))
    return time.time() - t0

if __name__ == "__main__":
    positions = np.load(POS_PATH)
    print(f"Loaded {len(positions)} positions | CPUs: {cpu_count()}")
    print(f"Averaging over {N_RUNS} runs, {N_EPOCHS} epochs each\n")

    # Collect averaged times per core count
    avg_times = {}
    for c in CORE_LIST:
        runs = []
        for r in range(N_RUNS):
            t = run_parallel(positions, c)
            runs.append(t)
        avg_times[c] = np.mean(runs)
        print(f"  {c:2d} cores: runs = {[f'{x:.3f}' for x in runs]}  ->  avg {avg_times[c]:.4f}s")

    base = avg_times[CORE_LIST[0]]
    print("\n" + "="*52)
    print("FINAL RESULTS TABLE (paste into paper)")
    print("="*52)
    print(f"{'Cores':>6} | {'Time (s)':>9} | {'Speedup':>8} | {'Efficiency':>11}")
    print("-"*52)
    for c in CORE_LIST:
        s = base / avg_times[c]
        eff = s / c * 100
        print(f"{c:>6} | {avg_times[c]:>9.4f} | {s:>7.2f}x | {eff:>9.1f}%")
    print("="*52)
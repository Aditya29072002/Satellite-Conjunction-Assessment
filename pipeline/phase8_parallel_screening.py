import json
import numpy as np
import time
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count

LOAD_PATH    = "data/satellites.json"
SUBSET       = 1500
WINDOW_HOURS = 24
STEP_MINUTES = 5            # finer steps => more timesteps => heavier real workload
THRESHOLD_KM = 25.0
CORE_LIST    = [1, 2, 4, 8, 16]
N_RUNS       = 3

_RECS  = None
_START = None

def _init_worker(recs, start):
    global _RECS, _START
    _RECS  = recs
    _START = start

def screen_one_timestep(tmin):
    """Propagate all objects to time tmin and screen for close pairs. Real work."""
    t = _START + timedelta(minutes=float(tmin))
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
    pts, idxs = [], []
    for ni, rec in enumerate(_RECS):
        e, r, v = rec.sgp4(jd, fr)
        if e == 0 and not any(np.isnan(r)):
            pts.append(r); idxs.append(ni)
    if len(pts) < 2:
        return []
    pts = np.array(pts)
    tree = KDTree(pts)
    out = []
    for a, b in tree.query_pairs(THRESHOLD_KM):
        d = np.linalg.norm(pts[a] - pts[b])
        out.append((idxs[a], idxs[b], d, tmin))
    return out

def build_recs(sats, subset):
    recs, names = [], []
    for s in sats[:subset]:
        try:
            recs.append(Satrec.twoline2rv(s["line1"], s["line2"]))
            names.append(s["name"])
        except Exception:
            pass
    return recs, names

def run_screening(recs, start, timesteps, n_workers):
    t0 = time.time()
    if n_workers == 1:
        _init_worker(recs, start)
        results = [screen_one_timestep(t) for t in timesteps]
    else:
        with Pool(n_workers, initializer=_init_worker,
                  initargs=(recs, start)) as pool:
            results = pool.map(screen_one_timestep, timesteps,
                               chunksize=max(1, len(timesteps)//(n_workers*2)))
    elapsed = time.time() - t0
    flat = [p for chunk in results for p in chunk]
    return flat, elapsed

if __name__ == "__main__":
    with open(LOAD_PATH) as f:
        sats = json.load(f)
    recs, names = build_recs(sats, SUBSET)
    start = datetime.now(timezone.utc)
    n_steps = int(WINDOW_HOURS * 60 / STEP_MINUTES) + 1
    timesteps = (np.arange(n_steps) * STEP_MINUTES).tolist()

    print(f"Real time-window screening")
    print(f"Objects: {len(recs)} | Window: {WINDOW_HOURS}h | "
          f"Steps: {n_steps} ({STEP_MINUTES} min) | CPUs: {cpu_count()}")
    print(f"Total SGP4 evaluations: {len(recs)*n_steps:,}")
    print(f"Averaging over {N_RUNS} runs\n")

    avg_times = {}
    n_pairs_seen = 0
    for c in CORE_LIST:
        runs = []
        for _ in range(N_RUNS):
            flat, t = run_screening(recs, start, timesteps, c)
            runs.append(t)
            n_pairs_seen = len(flat)
        avg_times[c] = np.mean(runs)
        print(f"  {c:2d} cores | avg {avg_times[c]:.4f}s | "
              f"detections this run: {n_pairs_seen}")

    base = avg_times[CORE_LIST[0]]
    print("\n" + "="*54)
    print("REAL-WORKLOAD STRONG SCALING (paste into paper)")
    print("="*54)
    print(f"{'Cores':>6} | {'Time (s)':>9} | {'Speedup':>8} | {'Efficiency':>11}")
    print("-"*54)
    speedups = []
    for c in CORE_LIST:
        s = base / avg_times[c]
        speedups.append(s)
        print(f"{c:>6} | {avg_times[c]:>9.4f} | {s:>7.2f}x | {s/c*100:>9.1f}%")
    print("="*54)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Phase 8 - Parallel Real Time-Window Screening "
                 f"({len(recs)} objects, {WINDOW_HOURS}h)", fontsize=13)
    axes[0].plot(CORE_LIST, speedups, "o-", color="darkorange",
                 linewidth=2, label="Actual")
    axes[0].plot(CORE_LIST, CORE_LIST, "--", color="gray", label="Ideal linear")
    axes[0].set_title("Strong Scaling Speedup (real workload)")
    axes[0].set_xlabel("CPU Cores"); axes[0].set_ylabel("Speedup")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    effs = [s/c*100 for s, c in zip(speedups, CORE_LIST)]
    axes[1].bar([str(c) for c in CORE_LIST], effs, color="teal", edgecolor="white")
    axes[1].axhline(100, color="red", linestyle="--", label="Ideal 100%")
    axes[1].set_title("Parallel Efficiency (real workload)")
    axes[1].set_xlabel("CPU Cores"); axes[1].set_ylabel("Efficiency (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/phase8_parallel_screening.png", dpi=150)
    plt.show()
    print("Saved: outputs/phase8_parallel_screening.png")

    
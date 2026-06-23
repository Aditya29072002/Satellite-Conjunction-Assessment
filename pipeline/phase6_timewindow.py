import json
import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta
from scipy.spatial import KDTree
import time

LOAD_PATH    = "data/satellites.json"
SUBSET       = 800          # objects to screen (kept modest: this is N x timesteps work)
WINDOW_HOURS = 24
STEP_MINUTES = 10           # time resolution
THRESHOLD_KM = 25.0         # screening distance for a "close approach"

def propagate_window(sats, subset, window_hours, step_minutes):
    """Propagate each object across the time window. Returns (T, N, 3) array."""
    sats = sats[:subset]
    start = datetime.now(timezone.utc)
    n_steps = int(window_hours * 60 / step_minutes) + 1
    times_min = np.arange(n_steps) * step_minutes

    # Build Satrec objects once
    recs, valid_names = [], []
    for s in sats:
        try:
            recs.append(Satrec.twoline2rv(s["line1"], s["line2"]))
            valid_names.append(s["name"])
        except Exception:
            pass

    N = len(recs)
    print(f"Propagating {N} objects across {window_hours}h "
          f"in {n_steps} steps ({step_minutes} min each)...")

    all_pos = np.full((n_steps, N, 3), np.nan)
    t0 = time.time()
    for ti, tmin in enumerate(times_min):
        t = start + timedelta(minutes=float(tmin))
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        for ni, rec in enumerate(recs):
            e, r, v = rec.sgp4(jd, fr)
            if e == 0 and not any(np.isnan(r)):
                all_pos[ti, ni] = r
    print(f"  Propagation done in {time.time()-t0:.2f}s")
    return all_pos, times_min, valid_names

def screen_window(all_pos, times_min, threshold):
    """At each timestep, find close pairs. Track closest approach per pair."""
    n_steps, N, _ = all_pos.shape
    closest = {}   # (i,j) -> (min_dist, time_min)
    t0 = time.time()
    for ti in range(n_steps):
        pts = all_pos[ti]
        good = ~np.isnan(pts[:, 0])
        idx_map = np.where(good)[0]
        pts_good = pts[good]
        if len(pts_good) < 2:
            continue
        tree = KDTree(pts_good)
        for a, b in tree.query_pairs(threshold):
            i, j = idx_map[a], idx_map[b]
            d = np.linalg.norm(pts_good[a] - pts_good[b])
            key = (i, j)
            if key not in closest or d < closest[key][0]:
                closest[key] = (d, times_min[ti])
    print(f"  Screening done in {time.time()-t0:.2f}s")
    return closest

def plot_results(closest, times_min, threshold):
    if not closest:
        print("No close approaches found - try a larger threshold or subset.")
        return
    dists  = np.array([v[0] for v in closest.values()])
    tcas   = np.array([v[1] for v in closest.values()]) / 60.0  # hours

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"Phase 6 - Time-Window Conjunction Screening "
                 f"({WINDOW_HOURS}h, threshold {int(threshold)} km)", fontsize=13)

    axes[0].hist(dists, bins=30, color="salmon", edgecolor="white")
    axes[0].axvline(10, color="red", linestyle="--", label="10 km critical")
    axes[0].set_title("Minimum Approach Distance per Pair")
    axes[0].set_xlabel("Closest distance (km)")
    axes[0].set_ylabel("Number of pairs")
    axes[0].legend(fontsize=8)

    axes[1].hist(tcas, bins=24, color="steelblue", edgecolor="white")
    axes[1].set_title("Time of Closest Approach")
    axes[1].set_xlabel("Time into window (hours)")
    axes[1].set_ylabel("Number of events")

    axes[2].scatter(tcas, dists, alpha=0.5, s=15, color="purple")
    axes[2].axhline(10, color="red", linestyle="--", label="10 km critical")
    axes[2].set_title("Distance vs Time of Closest Approach")
    axes[2].set_xlabel("Time into window (hours)")
    axes[2].set_ylabel("Closest distance (km)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/phase6_timewindow.png", dpi=150)
    plt.show()
    print("Saved: outputs/phase6_timewindow.png")

if __name__ == "__main__":
    with open(LOAD_PATH) as f:
        sats = json.load(f)

    all_pos, times_min, names = propagate_window(
        sats, SUBSET, WINDOW_HOURS, STEP_MINUTES)
    closest = screen_window(all_pos, times_min, THRESHOLD_KM)

    print(f"\nFound {len(closest)} unique close-approach pairs "
          f"over {WINDOW_HOURS}h.")
    if closest:
        crit = sum(1 for d, _ in closest.values() if d < 10)
        mind = min(d for d, _ in closest.values())
        print(f"  Critical (<10 km): {crit}")
        print(f"  Closest approach overall: {mind:.2f} km")
        # Show top 5 closest
        top = sorted(closest.items(), key=lambda kv: kv[1][0])[:5]
        print("\n  Top 5 closest approaches:")
        print(f"  {'Obj A':>22} | {'Obj B':>22} | {'Dist':>7} | {'T(h)':>5}")
        for (i, j), (d, t) in top:
            na = names[i][:22] if i < len(names) else str(i)
            nb = names[j][:22] if j < len(names) else str(j)
            print(f"  {na:>22} | {nb:>22} | {d:6.2f}km | {t/60:4.1f}")

    plot_results(closest, times_min, THRESHOLD_KM)



    
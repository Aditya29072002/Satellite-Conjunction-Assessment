import json
import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta

LOAD_PATH    = "data/satellites.json"
SUBSET       = 1500
WINDOW_HOURS = 24
STEP_MINUTES = 2          # finer steps for smooth curves

PAIRS_TO_PLOT = [
    ("O3B FM4", "O3B FM10"),
    ("O3B PFM", "O3B FM19"),
    ("ANIK F3", "ECHOSTAR 15"),
    ("GEO-KOMPSAT-2A", "GEO-KOMPSAT-2B"),
]

def build_states(sats, subset):
    recs, names = [], []
    for s in sats[:subset]:
        try:
            recs.append(Satrec.twoline2rv(s["line1"], s["line2"]))
            names.append(s["name"])
        except Exception:
            pass
    return recs, names

def find_index(names, query):
    for i, nm in enumerate(names):
        if query.upper() in nm.upper():
            return i
    return None

def propagate_pair(rec_a, rec_b, start, n_steps, step_min):
    times, dists = [], []
    for ti in range(n_steps):
        t = start + timedelta(minutes=ti*step_min)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        ea, ra, va = rec_a.sgp4(jd, fr)
        eb, rb, vb = rec_b.sgp4(jd, fr)
        if ea == 0 and eb == 0 and not any(np.isnan(ra)) and not any(np.isnan(rb)):
            d = np.linalg.norm(np.array(ra) - np.array(rb))
            times.append(ti*step_min/60.0)
            dists.append(d)
    return np.array(times), np.array(dists)

if __name__ == "__main__":
    with open(LOAD_PATH) as f:
        sats = json.load(f)
    recs, names = build_states(sats, SUBSET)
    start = datetime.now(timezone.utc)
    n_steps = int(WINDOW_HOURS*60/STEP_MINUTES)+1

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Phase 15 - Conjunction Approach Curves (24h window)",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for k, (na, nb) in enumerate(PAIRS_TO_PLOT):
        ia, ib = find_index(names, na), find_index(names, nb)
        ax = axes[k]
        if ia is None or ib is None:
            ax.text(0.5, 0.5, f"Pair not found:\n{na} / {nb}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{na} vs {nb}", fontsize=10)
            continue
        times, dists = propagate_pair(recs[ia], recs[ib], start, n_steps, STEP_MINUTES)
        if len(dists) == 0:
            continue
        kmin = int(np.argmin(dists))
        tmin, dmin = times[kmin], dists[kmin]

        ax.plot(times, dists, color="steelblue", linewidth=1.8)
        ax.scatter([tmin], [dmin], color="red", s=60, zorder=5,
                   label=f"Closest: {dmin:.2f} km at {tmin:.1f}h")
        ax.axhline(dmin, color="red", linestyle=":", alpha=0.4)
        ax.set_title(f"{names[ia]}  vs  {names[ib]}", fontsize=10)
        ax.set_xlabel("Time into window (hours)")
        ax.set_ylabel("Separation distance (km)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        print(f"{names[ia][:20]:>20} vs {names[ib][:20]:<20} "
              f"closest {dmin:6.2f} km at {tmin:5.1f}h")

    plt.tight_layout()
    plt.savefig("outputs/phase15_approach_curves.png", dpi=150)
    plt.show()
    print("\nSaved: outputs/phase15_approach_curves.png")

    
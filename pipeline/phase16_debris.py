import json
import numpy as np
import requests
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta
from scipy.spatial import KDTree

ACTIVE_PATH = "data/satellites.json"
THRESHOLD   = 25.0
WINDOW_HOURS = 12      # shorter window since debris set is large
STEP_MINUTES = 10

DEBRIS_SOURCES = {
    "Fengyun-1C": "https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle",
    "Cosmos-1408": "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle",
    "Iridium-33": "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle",
}

def fetch_debris(name, url):
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or "1 " not in r.text:
            return []
        lines = [l.strip() for l in r.text.strip().splitlines() if l.strip()]
        objs = []
        for i in range(0, len(lines)-2, 3):
            if lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
                objs.append({"name": f"{name}-DEB {lines[i]}",
                             "line1": lines[i+1], "line2": lines[i+2]})
        return objs
    except Exception as e:
        print(f"  Failed {name}: {e}")
        return []

def build_recs(objs):
    recs, names = [], []
    for s in objs:
        try:
            recs.append(Satrec.twoline2rv(s["line1"], s["line2"]))
            names.append(s["name"])
        except Exception:
            pass
    return recs, names

def propagate_at(recs, t):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
    pos = np.full((len(recs), 3), np.nan)
    for i, rec in enumerate(recs):
        e, r, v = rec.sgp4(jd, fr)
        if e == 0 and not any(np.isnan(r)):
            pos[i] = r
    return pos

if __name__ == "__main__":
    print("Fetching real debris catalogs from CelesTrak...")
    all_debris = []
    counts = {}
    for name, url in DEBRIS_SOURCES.items():
        d = fetch_debris(name, url)
        counts[name] = len(d)
        all_debris.extend(d)
        print(f"  {name}: {len(d)} fragments")

    # Load active satellites
    with open(ACTIVE_PATH) as f:
        active = json.load(f)
    active = active[:3000]   # subset of active sats
    for s in active:
        s["name"] = "ACTIVE: " + s["name"]

    combined = active + all_debris
    recs, names = build_recs(combined)
    is_debris = np.array(["DEB" in nm for nm in names])
    print(f"\nCombined catalog: {len(recs)} objects "
          f"({is_debris.sum()} debris, {(~is_debris).sum()} active)")

    if is_debris.sum() == 0:
        print("No debris fetched (CelesTrak groups may have changed). "
              "Showing active-only distribution.")

    # Screen combined catalog over short window, count active-vs-debris conjunctions
    start = datetime.now(timezone.utc)
    n_steps = int(WINDOW_HOURS*60/STEP_MINUTES)+1
    active_debris_pairs = 0
    debris_debris_pairs = 0
    closest_threats = []

    for ti in range(n_steps):
        t = start + timedelta(minutes=ti*STEP_MINUTES)
        pos = propagate_at(recs, t)
        good = ~np.isnan(pos[:,0])
        idx_map = np.where(good)[0]
        pg = pos[good]
        if len(pg) < 2:
            continue
        tree = KDTree(pg)
        for a, b in tree.query_pairs(THRESHOLD):
            i, j = idx_map[a], idx_map[b]
            di, dj = is_debris[i], is_debris[j]
            d = np.linalg.norm(pg[a]-pg[b])
            if di and dj:
                debris_debris_pairs += 1
            elif di != dj:   # one active, one debris
                active_debris_pairs += 1
                active_name = names[j] if di else names[i]
                debris_name = names[i] if di else names[j]
                closest_threats.append((active_name, debris_name, d, ti*STEP_MINUTES/60.0))

    print(f"\nConjunctions over {WINDOW_HOURS}h (threshold {THRESHOLD} km):")
    print(f"  Active <-> Debris: {active_debris_pairs}")
    print(f"  Debris <-> Debris: {debris_debris_pairs}")

    # dedupe active-debris threats, keep closest per pair
    seen = {}
    for an, dn, d, t in closest_threats:
        key = (an, dn)
        if key not in seen or d < seen[key][0]:
            seen[key] = (d, t)
    threats = sorted([(an, dn, d, t) for (an, dn), (d, t) in seen.items()],
                     key=lambda x: x[2])
    if threats:
        print(f"\nTop active-satellite vs debris close approaches:")
        print(f"  {'Active Sat':>28} | {'Debris':>22} | {'Dist':>7}")
        for an, dn, d, t in threats[:8]:
            print(f"  {an[:28]:>28} | {dn[:22]:>22} | {d:6.2f}km")

    # ── Visualization ─────────────────────────────────────────────────────────
    # ── Save debris snapshot to JSON for the dashboard ────────────────────────
    pos = propagate_at(recs, start)
    good = ~np.isnan(pos[:,0])
    snapshot = {
        "active": pos[good & ~is_debris].tolist(),
        "debris": pos[good & is_debris].tolist(),
        "counts": counts,
        "active_debris_pairs": int(active_debris_pairs),
        "debris_debris_pairs": int(debris_debris_pairs),
        "threats": [{"active": an, "debris": dn, "dist": round(d, 2)}
                    for an, dn, d, t in threats[:20]],
    }
    with open("data/debris_results.json", "w") as f:
        json.dump(snapshot, f)
    print("Saved: data/debris_results.json")

    fig = plt.figure(figsize=(15, 5.5))
    fig.suptitle("Phase 16 - Active Satellites vs Real Debris Clouds", fontsize=13)

    ax1 = fig.add_subplot(131, projection="3d")
    act = good & ~is_debris
    deb = good & is_debris
    ax1.scatter(pos[act,0], pos[act,1], pos[act,2], s=2, c="blue",
                alpha=0.4, label="Active")
    ax1.scatter(pos[deb,0], pos[deb,1], pos[deb,2], s=2, c="red",
                alpha=0.4, label="Debris")
    ax1.set_title("3D: Active (blue) vs Debris (red)")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(132)
    ax2.scatter(pos[act,0], pos[act,1], s=2, c="blue", alpha=0.4, label="Active")
    ax2.scatter(pos[deb,0], pos[deb,1], s=2, c="red", alpha=0.4, label="Debris")
    circle = plt.Circle((0,0), 6371, color="green", fill=False, linewidth=1.5)
    ax2.add_patch(circle); ax2.set_aspect("equal")
    ax2.set_title("Equatorial Projection")
    ax2.set_xlabel("X (km)"); ax2.set_ylabel("Y (km)")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(133)
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    ax3.bar(labels, vals, color=["orange","crimson","purple"][:len(labels)],
            edgecolor="white")
    ax3.set_title("Debris Fragments by Source Event")
    ax3.set_ylabel("Tracked fragments")
    for k, v in enumerate(vals):
        ax3.text(k, v+max(vals)*0.02 if vals else 0, str(v),
                 ha="center", fontsize=9)
    plt.setp(ax3.get_xticklabels(), rotation=15, fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/phase16_debris.png", dpi=150)
    plt.show()
    print("\nSaved: outputs/phase16_debris.png")


    
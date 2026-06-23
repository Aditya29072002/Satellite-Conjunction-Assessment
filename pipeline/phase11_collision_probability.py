import json
import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta
from scipy.spatial import KDTree
from scipy.special import erf

LOAD_PATH    = "data/satellites.json"
SUBSET       = 1500
WINDOW_HOURS = 24
STEP_MINUTES = 5
THRESHOLD_KM = 25.0      # screening distance
EPS_KM       = 0.5       # below this = self-conjunction (filtered)
HARD_BODY_R  = 0.020     # combined object radius in km (20 m, typical assumption)
PC_ACTION    = 1e-4      # maneuver threshold

# Representative 1-sigma position uncertainty (km) by altitude regime.
# These are MODELED values (TLEs carry no covariance) - stated in paper.
def sigma_for_altitude(alt_km):
    if alt_km < 2000:        # LEO - frequently tracked, smaller uncertainty
        return np.array([0.5, 1.5, 0.5])   # radial, along-track, cross-track
    elif alt_km < 35000:     # MEO
        return np.array([1.0, 3.0, 1.0])
    else:                    # GEO - less frequent tracking
        return np.array([1.5, 5.0, 1.5])

def build_states(sats, subset):
    """Return Satrec objects, names, and altitude for a subset."""
    recs, names = [], []
    for s in sats[:subset]:
        try:
            recs.append(Satrec.twoline2rv(s["line1"], s["line2"]))
            names.append(s["name"])
        except Exception:
            pass
    return recs, names

def propagate_at(recs, t):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
    pos = np.full((len(recs), 3), np.nan)
    vel = np.full((len(recs), 3), np.nan)
    for i, rec in enumerate(recs):
        e, r, v = rec.sgp4(jd, fr)
        if e == 0 and not any(np.isnan(r)):
            pos[i] = r
            vel[i] = v
    return pos, vel

def collision_probability_2d(r_rel, v_rel, sigma1, sigma2, hbr):
    """
    Foster 2D Pc method.
    r_rel: relative position vector (km) at closest approach
    v_rel: relative velocity vector (km/s)
    sigma1, sigma2: 1-sigma uncertainty vectors (km) of each object
    hbr: combined hard-body radius (km)
    Returns probability of collision.
    """
    miss = np.linalg.norm(r_rel)
    vmag = np.linalg.norm(v_rel)
    if vmag < 1e-9:
        return 0.0

    # Combined covariance (assume independent, diagonal in RTN ~ approximated in ECI)
    # Total variance per axis = sum of both objects' variances
    sig_comb = np.sqrt(sigma1**2 + sigma2**2)

    # Build the conjunction plane: perpendicular to relative velocity
    v_hat = v_rel / vmag
    # Two orthonormal vectors spanning the plane perpendicular to v_hat
    arb = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arb, v_hat)) > 0.9:
        arb = np.array([0.0, 1.0, 0.0])
    u1 = arb - np.dot(arb, v_hat) * v_hat
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v_hat, u1)

    # Project combined sigma onto plane axes (variance projection)
    var_u1 = (sig_comb[0]*u1[0])**2 + (sig_comb[1]*u1[1])**2 + (sig_comb[2]*u1[2])**2
    var_u2 = (sig_comb[0]*u2[0])**2 + (sig_comb[1]*u2[1])**2 + (sig_comb[2]*u2[2])**2
    s1 = np.sqrt(max(var_u1, 1e-12))
    s2 = np.sqrt(max(var_u2, 1e-12))

    # Project miss vector onto plane
    x = np.dot(r_rel, u1)
    y = np.dot(r_rel, u2)

    # 2D Gaussian integrated over circle of radius hbr.
    # Closed-form approximation via grid integration over the hard-body disk.
    n = 40
    xs = np.linspace(-hbr, hbr, n)
    ys = np.linspace(-hbr, hbr, n)
    dx = xs[1] - xs[0]
    pc = 0.0
    for xi in xs:
        for yi in ys:
            if xi*xi + yi*yi <= hbr*hbr:
                px = (x + xi)
                py = (y + yi)
                val = (1.0/(2*np.pi*s1*s2)) * np.exp(
                    -0.5*((px/s1)**2 + (py/s2)**2))
                pc += val * dx * dx
    return min(pc, 1.0)

def screen_with_pc(recs, names, alts, window_hours, step_minutes, threshold):
    start = datetime.now(timezone.utc)
    n_steps = int(window_hours*60/step_minutes)+1
    # Track closest approach state per pair
    closest = {}  # (i,j) -> (dist, t_min, r_rel, v_rel)
    for ti in range(n_steps):
        t = start + timedelta(minutes=ti*step_minutes)
        pos, vel = propagate_at(recs, t)
        good = ~np.isnan(pos[:,0])
        idx_map = np.where(good)[0]
        pg = pos[good]
        if len(pg) < 2:
            continue
        tree = KDTree(pg)
        for a, b in tree.query_pairs(threshold):
            i, j = idx_map[a], idx_map[b]
            d = np.linalg.norm(pg[a]-pg[b])
            key = (i, j)
            if key not in closest or d < closest[key][0]:
                r_rel = pos[i]-pos[j]
                v_rel = vel[i]-vel[j]
                closest[key] = (d, ti*step_minutes, r_rel, v_rel)
    return closest

if __name__ == "__main__":
    with open(LOAD_PATH) as f:
        sats = json.load(f)
    recs, names = build_states(sats, SUBSET)

    # Compute altitude of each object once (for sigma assignment)
    start = datetime.now(timezone.utc)
    pos0, _ = propagate_at(recs, start)
    alts = np.linalg.norm(pos0, axis=1) - 6371

    print(f"Screening {len(recs)} objects over {WINDOW_HOURS}h for Pc...")
    closest = screen_with_pc(recs, names, alts, WINDOW_HOURS,
                             STEP_MINUTES, THRESHOLD_KM)

    # Compute Pc for each genuine pair
    results = []
    for (i, j), (d, tmin, r_rel, v_rel) in closest.items():
        if d < EPS_KM:
            continue   # skip self-conjunctions
        s1 = sigma_for_altitude(alts[i])
        s2 = sigma_for_altitude(alts[j])
        pc = collision_probability_2d(r_rel, v_rel, s1, s2, HARD_BODY_R)
        results.append({
            "a": names[i], "b": names[j], "dist": d,
            "time_h": tmin/60.0, "pc": pc
        })

    results.sort(key=lambda r: -r["pc"])

    print(f"\nGenuine pairs assessed: {len(results)}")
    actionable = [r for r in results if r["pc"] >= PC_ACTION]
    print(f"Actionable (Pc >= {PC_ACTION:.0e}): {len(actionable)}")

    print(f"\n{'Object A':>20} | {'Object B':>20} | {'Dist(km)':>8} | {'Pc':>10}")
    print("-"*70)
    for r in results[:10]:
        flag = " <-- ACTION" if r["pc"] >= PC_ACTION else ""
        print(f"{r['a'][:20]:>20} | {r['b'][:20]:>20} | "
              f"{r['dist']:8.2f} | {r['pc']:.2e}{flag}")

    # Save results to JSON for the dashboard
    with open("data/pc_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to data/pc_results.json")

    # Visualization: Pc vs distance
    if results:
        dists = [r["dist"] for r in results]
        pcs   = [max(r["pc"], 1e-20) for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Phase 11 - Collision Probability Assessment", fontsize=13)

        ax = axes[0]
        colors = ["red" if p >= PC_ACTION else "steelblue" for p in pcs]
        ax.scatter(dists, pcs, c=colors, s=40, alpha=0.7)
        ax.axhline(PC_ACTION, color="red", linestyle="--",
                   label=f"Action threshold ({PC_ACTION:.0e})")
        ax.set_yscale("log")
        ax.set_xlabel("Miss distance (km)")
        ax.set_ylabel("Collision probability (Pc)")
        ax.set_title("Pc vs Miss Distance")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[1]
        valid_pcs = [np.log10(p) for p in pcs if p > 1e-20]
        ax.hist(valid_pcs, bins=25, color="coral", edgecolor="white")
        ax.axvline(np.log10(PC_ACTION), color="red", linestyle="--",
                   label="Action threshold")
        ax.set_xlabel("log10(Pc)")
        ax.set_ylabel("Number of pairs")
        ax.set_title("Distribution of Collision Probabilities")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("outputs/phase11_collision_probability.png", dpi=150)
        plt.show()
        print("Saved: outputs/phase11_collision_probability.png")

        
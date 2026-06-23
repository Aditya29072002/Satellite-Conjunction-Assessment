import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta
from scipy.spatial import KDTree

LOAD_PATH    = "data/satellites.json"
SUBSET       = 1500
WINDOW_HOURS = 24
STEP_MINUTES = 5
THRESHOLD_KM = 25.0
EPS_KM       = 0.5
HARD_BODY_R  = 0.020
PC_ACTION    = 1e-4

def sigma_for_altitude(alt_km):
    if alt_km < 2000:
        return np.array([0.5, 1.5, 0.5])
    elif alt_km < 35000:
        return np.array([1.0, 3.0, 1.0])
    else:
        return np.array([1.5, 5.0, 1.5])

def build_states(sats, subset):
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
            pos[i] = r; vel[i] = v
    return pos, vel

def collision_probability_2d(r_rel, v_rel, sigma1, sigma2, hbr):
    vmag = np.linalg.norm(v_rel)
    if vmag < 1e-9:
        return 0.0
    sig = np.sqrt(sigma1**2 + sigma2**2)
    v_hat = v_rel / vmag
    arb = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arb, v_hat)) > 0.9:
        arb = np.array([0.0, 1.0, 0.0])
    u1 = arb - np.dot(arb, v_hat)*v_hat; u1 /= np.linalg.norm(u1)
    u2 = np.cross(v_hat, u1)
    var1 = sum((sig[k]*u1[k])**2 for k in range(3))
    var2 = sum((sig[k]*u2[k])**2 for k in range(3))
    s1, s2 = np.sqrt(max(var1,1e-12)), np.sqrt(max(var2,1e-12))
    x, y = np.dot(r_rel,u1), np.dot(r_rel,u2)
    n = 40; xs = np.linspace(-hbr,hbr,n); dx = xs[1]-xs[0]; pc = 0.0
    for xi in xs:
        for yi in xs:
            if xi*xi+yi*yi <= hbr*hbr:
                pc += (1/(2*np.pi*s1*s2))*np.exp(-0.5*(((x+xi)/s1)**2+((y+yi)/s2)**2))*dx*dx
    return min(pc, 1.0)

def parabolic_tca(t_prev, t_curr, t_next, d_prev, d_curr, d_next):
    """Fit parabola to 3 points, return (refined_time, refined_dist)."""
    denom = (d_prev - 2*d_curr + d_next)
    if abs(denom) < 1e-12:
        return t_curr, d_curr
    # vertex offset in units of step (between -1 and 1)
    delta = 0.5 * (d_prev - d_next) / denom
    delta = max(-1.0, min(1.0, delta))
    step = t_curr - t_prev
    t_ref = t_curr + delta * step
    # refined distance at vertex
    d_ref = d_curr - 0.25 * (d_prev - d_next) * delta
    return t_ref, max(d_ref, 0.0)

def classify(pc, dist):
    if pc >= PC_ACTION:
        return "RED"
    if pc >= 1e-5 or dist < 5.0:
        return "YELLOW"
    return "GREEN"

if __name__ == "__main__":
    with open(LOAD_PATH) as f:
        sats = json.load(f)
    recs, names = build_states(sats, SUBSET)
    start = datetime.now(timezone.utc)
    n_steps = int(WINDOW_HOURS*60/STEP_MINUTES)+1

    pos0, _ = propagate_at(recs, start)
    alts = np.linalg.norm(pos0, axis=1) - 6371

    print(f"Screening {len(recs)} objects over {WINDOW_HOURS}h with TCA refinement...")

    # Store full distance history per pair so we can interpolate around the minimum
    history = {}   # (i,j) -> list of (step_index, dist, pos_i, pos_j, vel_i, vel_j)
    for ti in range(n_steps):
        t = start + timedelta(minutes=ti*STEP_MINUTES)
        pos, vel = propagate_at(recs, t)
        good = ~np.isnan(pos[:,0]); idx_map = np.where(good)[0]; pg = pos[good]
        if len(pg) < 2: continue
        tree = KDTree(pg)
        for a, b in tree.query_pairs(THRESHOLD_KM):
            i, j = idx_map[a], idx_map[b]
            d = np.linalg.norm(pg[a]-pg[b])
            history.setdefault((i,j), []).append(
                (ti, d, pos[i].copy(), pos[j].copy(), vel[i].copy(), vel[j].copy()))

    results = []
    for (i,j), hist in history.items():
        # find sampled minimum
        dists = [h[1] for h in hist]
        kmin = int(np.argmin(dists))
        d_curr = dists[kmin]
        if d_curr < EPS_KM:
            continue  # self-conjunction
        ti_curr = hist[kmin][0]
        t_curr_min = ti_curr * STEP_MINUTES

        # parabolic refinement if we have neighbors in the sampled history
        # (neighbors must be consecutive steps)
        d_ref, t_ref_min = d_curr, t_curr_min
        prev = next_ = None
        for h in hist:
            if h[0] == ti_curr-1: prev = h
            if h[0] == ti_curr+1: next_ = h
        if prev and next_:
            t_ref_min, d_ref = parabolic_tca(
                (ti_curr-1)*STEP_MINUTES, ti_curr*STEP_MINUTES, (ti_curr+1)*STEP_MINUTES,
                prev[1], d_curr, next_[1])

        # Pc using the closest sampled geometry
        _, _, pi, pj, vi, vj = hist[kmin]
        r_rel, v_rel = pi-pj, vi-vj
        s1, s2 = sigma_for_altitude(alts[i]), sigma_for_altitude(alts[j])
        pc = collision_probability_2d(r_rel, v_rel, s1, s2, HARD_BODY_R)

        results.append({
            "a": names[i], "b": names[j],
            "dist_sampled": round(d_curr, 3),
            "dist_refined": round(d_ref, 3),
            "tca_sampled_h": round(t_curr_min/60.0, 3),
            "tca_refined_h": round(t_ref_min/60.0, 3),
            "pc": pc,
            "risk": classify(pc, d_ref),
        })

    results.sort(key=lambda r: -r["pc"])

    reds = [r for r in results if r["risk"]=="RED"]
    yellows = [r for r in results if r["risk"]=="YELLOW"]
    greens = [r for r in results if r["risk"]=="GREEN"]
    print(f"\nAssessed {len(results)} genuine pairs:")
    print(f"  RED:    {len(reds)}")
    print(f"  YELLOW: {len(yellows)}")
    print(f"  GREEN:  {len(greens)}")

    # Show refinement effect on top pairs
    print(f"\n{'Object A':>18} | {'Object B':>18} | {'Samp':>6} | {'Refined':>7} | {'Pc':>9} | Risk")
    print("-"*78)
    for r in results[:10]:
        print(f"{r['a'][:18]:>18} | {r['b'][:18]:>18} | "
              f"{r['dist_sampled']:6.2f} | {r['dist_refined']:7.2f} | "
              f"{r['pc']:.2e} | {r['risk']}")

    # ── Export CSV ───────────────────────────────────────────────────────────
    with open("outputs/conjunction_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Object_A","Object_B","Dist_sampled_km","Dist_refined_km",
                    "TCA_sampled_h","TCA_refined_h","Pc","Risk"])
        for r in results:
            w.writerow([r["a"], r["b"], r["dist_sampled"], r["dist_refined"],
                        r["tca_sampled_h"], r["tca_refined_h"],
                        f"{r['pc']:.3e}", r["risk"]])
    print(f"\nSaved CSV: outputs/conjunction_report.csv")

    # ── Export text report ───────────────────────────────────────────────────
    with open("outputs/conjunction_report.txt", "w") as f:
        f.write("="*60+"\n")
        f.write("  CONJUNCTION ASSESSMENT REPORT\n")
        f.write(f"  Generated: {start.strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write(f"  Window: {WINDOW_HOURS}h | Objects screened: {len(recs)}\n")
        f.write("="*60+"\n\n")
        f.write(f"SUMMARY:  RED={len(reds)}  YELLOW={len(yellows)}  GREEN={len(greens)}\n\n")
        f.write("TOP EVENTS BY COLLISION PROBABILITY:\n")
        f.write("-"*60+"\n")
        for r in results[:15]:
            f.write(f"[{r['risk']:6}] {r['a'][:20]:>20} <-> {r['b'][:20]:<20}\n")
            f.write(f"         miss={r['dist_refined']:.2f} km  "
                    f"TCA={r['tca_refined_h']:.1f}h  Pc={r['pc']:.2e}\n")
    print(f"Saved report: outputs/conjunction_report.txt")

    # Save JSON for dashboard
    with open("data/operational_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: data/operational_results.json")

    # ── Visual: refinement effect + risk breakdown ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase 13 - Operational Screening: TCA Refinement & Risk", fontsize=13)

    samp = [r["dist_sampled"] for r in results]
    refd = [r["dist_refined"] for r in results]
    axes[0].scatter(samp, refd, alpha=0.6, s=30, color="purple")
    lim = max(max(samp), max(refd))*1.05
    axes[0].plot([0,lim],[0,lim], "--", color="gray", label="No change")
    axes[0].set_xlabel("Sampled miss distance (km)")
    axes[0].set_ylabel("Refined miss distance (km)")
    axes[0].set_title("Effect of Parabolic TCA Refinement")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    counts = [len(reds), len(yellows), len(greens)]
    cols = ["#d62728", "#ff7f0e", "#2ca02c"]
    axes[1].bar(["RED","YELLOW","GREEN"], counts, color=cols, edgecolor="white")
    for k, v in enumerate(counts):
        axes[1].text(k, v+0.3, str(v), ha="center", fontweight="bold")
    axes[1].set_title("Conjunctions by Risk Class")
    axes[1].set_ylabel("Number of pairs")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/phase13_operational.png", dpi=150)
    plt.show()
    print("Saved: outputs/phase13_operational.png")

    
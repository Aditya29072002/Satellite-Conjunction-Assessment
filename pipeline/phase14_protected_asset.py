import json
import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta

LOAD_PATH    = "data/satellites.json"
SUBSET       = 1500
WINDOW_HOURS = 24
STEP_MINUTES = 5
THRESHOLD_KM = 50.0      # wider net for a single asset
EPS_KM       = 0.5
HARD_BODY_R  = 0.020
PC_ACTION    = 1e-4

# ── Choose the protected asset by name (substring match) ─────────────────────
PROTECTED = "ISS (ZARYA)"     # change to any satellite name you like

def sigma_for_altitude(alt_km):
    if alt_km < 2000:   return np.array([0.5, 1.5, 0.5])
    elif alt_km < 35000: return np.array([1.0, 3.0, 1.0])
    else:               return np.array([1.5, 5.0, 1.5])

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

def collision_probability_2d(r_rel, v_rel, s1, s2, hbr):
    vmag = np.linalg.norm(v_rel)
    if vmag < 1e-9: return 0.0
    sig = np.sqrt(s1**2 + s2**2)
    v_hat = v_rel/vmag
    arb = np.array([1.0,0,0])
    if abs(np.dot(arb,v_hat))>0.9: arb = np.array([0,1.0,0])
    u1 = arb-np.dot(arb,v_hat)*v_hat; u1/=np.linalg.norm(u1)
    u2 = np.cross(v_hat,u1)
    var1 = sum((sig[k]*u1[k])**2 for k in range(3))
    var2 = sum((sig[k]*u2[k])**2 for k in range(3))
    sa, sb = np.sqrt(max(var1,1e-12)), np.sqrt(max(var2,1e-12))
    x,y = np.dot(r_rel,u1), np.dot(r_rel,u2)
    n=40; xs=np.linspace(-hbr,hbr,n); dx=xs[1]-xs[0]; pc=0.0
    for xi in xs:
        for yi in xs:
            if xi*xi+yi*yi<=hbr*hbr:
                pc += (1/(2*np.pi*sa*sb))*np.exp(-0.5*(((x+xi)/sa)**2+((y+yi)/sb)**2))*dx*dx
    return min(pc,1.0)

def classify(pc, dist):
    if pc >= PC_ACTION: return "RED"
    if pc >= 1e-5 or dist < 5.0: return "YELLOW"
    return "GREEN"

if __name__ == "__main__":
    with open(LOAD_PATH) as f:
        sats = json.load(f)
    recs, names = build_states(sats, SUBSET)

    # Find the protected asset index
    target = None
    for i, nm in enumerate(names):
        if PROTECTED.upper() in nm.upper():
            target = i; break
    if target is None:
        print(f"'{PROTECTED}' not found in first {SUBSET} objects.")
        print("Some available names:", ", ".join(names[:15]))
        raise SystemExit

    print(f"Protected asset: {names[target]} (index {target})")
    print(f"Screening against {len(recs)-1} other objects over {WINDOW_HOURS}h "
          f"(threshold {THRESHOLD_KM} km)...\n")

    start = datetime.now(timezone.utc)
    n_steps = int(WINDOW_HOURS*60/STEP_MINUTES)+1
    pos0, _ = propagate_at(recs, start)
    alts = np.linalg.norm(pos0, axis=1) - 6371

    # Track closest approach of every object to the target
    closest = {}   # other_idx -> (dist, t_min, r_rel, v_rel)
    target_track = []   # (t_min, target_position) for plotting orbit
    for ti in range(n_steps):
        t = start + timedelta(minutes=ti*STEP_MINUTES)
        pos, vel = propagate_at(recs, t)
        if np.isnan(pos[target,0]):
            continue
        target_track.append((ti*STEP_MINUTES, pos[target].copy()))
        diff = pos - pos[target]
        d = np.linalg.norm(diff, axis=1)
        for j in range(len(recs)):
            if j == target or np.isnan(d[j]):
                continue
            if d[j] < THRESHOLD_KM:
                if j not in closest or d[j] < closest[j][0]:
                    closest[j] = (d[j], ti*STEP_MINUTES,
                                  pos[target]-pos[j], vel[target]-vel[j])

    # Assess each threatening object
    threats = []
    s_t = sigma_for_altitude(alts[target])
    for j, (d, tmin, r_rel, v_rel) in closest.items():
        if d < EPS_KM:
            continue
        pc = collision_probability_2d(r_rel, v_rel, s_t,
                                      sigma_for_altitude(alts[j]), HARD_BODY_R)
        threats.append({"obj": names[j], "dist": d, "tca_h": tmin/60.0,
                        "pc": pc, "risk": classify(pc, d)})
    threats.sort(key=lambda r: -r["pc"])

    reds = sum(1 for t in threats if t["risk"]=="RED")
    yellows = sum(1 for t in threats if t["risk"]=="YELLOW")
    print(f"Threats to {names[target]}: {len(threats)} objects within {THRESHOLD_KM} km")
    print(f"  RED={reds}  YELLOW={yellows}  GREEN={len(threats)-reds-yellows}\n")
    print(f"{'Threatening Object':>22} | {'Dist(km)':>8} | {'TCA(h)':>6} | {'Pc':>9} | Risk")
    print("-"*65)
    for t in threats[:12]:
        print(f"{t['obj'][:22]:>22} | {t['dist']:8.2f} | {t['tca_h']:6.1f} | "
              f"{t['pc']:.2e} | {t['risk']}")

    # Save
    out = {"protected": names[target], "threats": threats}
    with open("data/protected_asset_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: data/protected_asset_results.json")

    # ── Visualization ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5.5))
    fig.suptitle(f"Phase 14 - Protected Asset Screening: {names[target]}",
                 fontsize=13)

    # 3D: target orbit + threatening objects' closest points
    ax1 = fig.add_subplot(121, projection="3d")
    track = np.array([p for _, p in target_track])
    ax1.plot(track[:,0], track[:,1], track[:,2],
             color="blue", linewidth=1.5, label=f"{names[target]} orbit")
    for t, (j, (d, tmin, r_rel, v_rel)) in zip(threats, closest.items()):
        pass
    # plot closest approach points of top threats
    for j, (d, tmin, r_rel, v_rel) in list(closest.items())[:40]:
        # target pos at tmin approx = nearest track sample
        col = "red" if d < 10 else ("orange" if d < 25 else "gray")
        # threatening object's position = target_pos - r_rel
        # find target pos at tmin
        ti = int(tmin/STEP_MINUTES)
        if ti < len(track):
            obj_pos = track[ti] - r_rel
            ax1.scatter(*obj_pos, c=col, s=20)
    ax1.scatter([], [], c="red", label="<10 km")
    ax1.scatter([], [], c="orange", label="10-25 km")
    ax1.set_title("Orbit and Close Approaches")
    ax1.legend(fontsize=7)

    # Threat distance vs TCA
    ax2 = fig.add_subplot(122)
    if threats:
        dists = [t["dist"] for t in threats]
        tcas  = [t["tca_h"] for t in threats]
        cols = ["red" if t["risk"]=="RED" else
                ("orange" if t["risk"]=="YELLOW" else "green") for t in threats]
        ax2.scatter(tcas, dists, c=cols, s=40, alpha=0.7)
        ax2.axhline(5, color="red", linestyle="--", alpha=0.5, label="5 km")
        ax2.set_xlabel("Time of closest approach (h)")
        ax2.set_ylabel("Miss distance (km)")
        ax2.set_title(f"Threats to {names[target]}")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/phase14_protected_asset.png", dpi=150)
    plt.show()
    print("Saved: outputs/phase14_protected_asset.png")


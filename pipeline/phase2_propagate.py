import json
import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, jday
from datetime import datetime, timezone
import time

LOAD_PATH = "data/satellites.json"
POS_PATH  = "data/positions.npy"

def propagate_satellites(sats, subset=2000):
    """Convert TLEs into ECI (X,Y,Z) positions in km."""
    sats = sats[:subset]
    now  = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day,
                  now.hour, now.minute, now.second)

    positions = []
    failed    = 0
    t0 = time.time()
    for s in sats:
        try:
            sat = Satrec.twoline2rv(s["line1"], s["line2"])
            e, r, v = sat.sgp4(jd, fr)
            if e == 0 and not any(np.isnan(r)):
                positions.append(r)
            else:
                failed += 1
        except:
            failed += 1
    elapsed = time.time() - t0
    positions = np.array(positions)
    print(f"Propagated {len(positions)} satellites in {elapsed:.2f}s "
          f"({failed} failed)")
    return positions

def plot_positions(positions):
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"Phase 2 - Satellite Positions ({len(positions)} objects)",
                 fontsize=13)

    ax1 = fig.add_subplot(131, projection="3d")
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    R = 6371
    ax1.plot_surface(R*np.cos(u)*np.sin(v),
                     R*np.sin(u)*np.sin(v),
                     R*np.cos(v),
                     color="deepskyblue", alpha=0.3)
    ax1.scatter(positions[:,0], positions[:,1], positions[:,2],
                s=1, c="red", alpha=0.5)
    ax1.set_title("3D Orbital Distribution")
    ax1.set_xlabel("X (km)"); ax1.set_ylabel("Y (km)"); ax1.set_zlabel("Z (km)")

    ax2 = fig.add_subplot(132)
    ax2.scatter(positions[:,0], positions[:,1], s=1, alpha=0.4, color="orange")
    circle = plt.Circle((0,0), 6371, color="blue", fill=False,
                        linewidth=1.5, label="Earth")
    ax2.add_patch(circle)
    ax2.set_aspect("equal")
    ax2.set_title("XY Projection (Equatorial Plane)")
    ax2.set_xlabel("X (km)"); ax2.set_ylabel("Y (km)")
    ax2.legend()

    ax3 = fig.add_subplot(133)
    radii = np.linalg.norm(positions, axis=1)
    alts  = radii - 6371
    valid = alts[(alts > 100) & (alts < 40000)]
    ax3.hist(valid, bins=50, color="mediumseagreen", edgecolor="white")
    ax3.axvline(2000, color="red", linestyle="--", label="LEO/MEO boundary")
    ax3.axvline(35786, color="purple", linestyle="--", label="GEO altitude")
    ax3.set_title("Altitude Distribution")
    ax3.set_xlabel("Altitude (km)"); ax3.set_ylabel("Count")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/phase2_positions.png", dpi=150)
    plt.show()
    print("Saved: outputs/phase2_positions.png")

if __name__ == "__main__":
    with open(LOAD_PATH) as f:
        sats = json.load(f)
    positions = propagate_satellites(sats, subset=2000)
    np.save(POS_PATH, positions)
    print(f"Saved positions to {POS_PATH}")
    plot_positions(positions)
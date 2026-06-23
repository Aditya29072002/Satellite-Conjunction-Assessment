import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_PATH = "data/satellites.json"
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def fetch_tle_data():
    urls = [
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    ]
    lines = []
    for url in urls:
        try:
            print(f"Trying: {url}")
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and "1 " in r.text:
                lines = [l.strip() for l in r.text.strip().splitlines() if l.strip()]
                print(f"Got {len(lines)} lines from {url}")
                break
        except Exception as e:
            print(f"  Failed: {e}")
    if len(lines) < 9:
        print("Live fetch failed. Generating synthetic dataset (500 objects).")
        return generate_synthetic_tles(500)
    return parse_tles(lines)

def generate_synthetic_tles(n=500):
    import math
    sats = []
    rng = np.random.default_rng(42)
    for i in range(n):
        alt_km   = rng.uniform(200, 35786)
        inc_deg  = rng.uniform(0, 110)
        raan_deg = rng.uniform(0, 360)
        ecc      = rng.uniform(0, 0.01)
        argp_deg = rng.uniform(0, 360)
        ma_deg   = rng.uniform(0, 360)
        mu = 398600.4418
        a  = 6371 + alt_km
        n_rev = math.sqrt(mu / a**3) * (86400 / (2*math.pi))
        sats.append({
            "name": f"SYNTH-{i:04d}",
            "line1": f"1 {i+1:05d}U 00000A   24001.00000000  .00000000  00000-0  00000-0 0  0000",
            "line2": f"2 {i+1:05d} {inc_deg:8.4f} {raan_deg:8.4f} {int(ecc*1e7):07d} "
                     f"{argp_deg:8.4f} {ma_deg:8.4f} {n_rev:11.8f}    00",
            "inclination": inc_deg,
            "altitude_km": alt_km,
            "mean_motion": n_rev,
        })
    print(f"Generated {len(sats)} synthetic satellites.")
    return sats

def parse_tles(lines):
    sats = []
    i = 0
    while i < len(lines) - 2:
        name = lines[i].strip()
        l1   = lines[i+1].strip()
        l2   = lines[i+2].strip()
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                inc  = float(l2[8:16])
                mm   = float(l2[52:63])
                sats.append({
                    "name": name,
                    "line1": l1,
                    "line2": l2,
                    "inclination": inc,
                    "mean_motion": mm,
                    "altitude_km": ((398600.4418 /
                                    (mm * 2*3.14159/86400)**2)**(1/3)) - 6371
                })
                i += 3
            except:
                i += 1
        else:
            i += 1
    return sats

def plot_statistics(sats):
    incs = [s["inclination"] for s in sats]
    alts = [s["altitude_km"] for s in sats]
    mms  = [s["mean_motion"] for s in sats]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Phase 1 - Dataset Overview ({len(sats)} Objects)", fontsize=13)
    axes[0].hist(incs, bins=36, color="steelblue", edgecolor="white")
    axes[0].set_title("Orbital Inclination Distribution")
    axes[0].set_xlabel("Inclination (deg)")
    axes[0].set_ylabel("Count")
    valid_alts = [a for a in alts if 0 < a < 40000]
    axes[1].hist(valid_alts, bins=40, color="coral", edgecolor="white")
    axes[1].set_title("Altitude Distribution")
    axes[1].set_xlabel("Altitude (km)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(2000, color="red", linestyle="--", label="LEO/MEO boundary")
    axes[1].axvline(35786, color="purple", linestyle="--", label="GEO")
    axes[1].legend(fontsize=7)
    axes[2].scatter(alts[:500], mms[:500], alpha=0.4, s=8, color="green")
    axes[2].set_title("Mean Motion vs Altitude")
    axes[2].set_xlabel("Altitude (km)")
    axes[2].set_ylabel("Mean Motion (rev/day)")
    plt.tight_layout()
    plt.savefig("outputs/phase1_dataset_overview.png", dpi=150)
    plt.show()
    print("Saved: outputs/phase1_dataset_overview.png")

if __name__ == "__main__":
    sats = fetch_tle_data()
    print(f"\nTotal satellites loaded: {len(sats)}")
    print(f"   Sample: {sats[0]['name']}")
    print(f"   Inclination range: {min(s['inclination'] for s in sats):.1f} - "
          f"{max(s['inclination'] for s in sats):.1f} deg")
    with open(SAVE_PATH, "w") as f:
        json.dump(sats, f)
    print(f"Saved to {SAVE_PATH}")
    plot_statistics(sats)
    
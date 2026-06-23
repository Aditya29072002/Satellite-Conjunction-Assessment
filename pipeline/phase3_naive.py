import numpy as np
import matplotlib.pyplot as plt
import time

POS_PATH = "data/positions.npy"
THRESHOLD_KM = 10.0

def naive_vectorized(positions, threshold=THRESHOLD_KM):
    """O(N^2) using NumPy broadcasting."""
    t0 = time.time()
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    mask = (dists < threshold) & (np.triu(np.ones_like(dists), k=1) > 0)
    i_idx, j_idx = np.where(mask)
    close_pairs = list(zip(i_idx, j_idx, dists[i_idx, j_idx]))
    elapsed = time.time() - t0
    return close_pairs, elapsed

def benchmark_naive(positions):
    sizes  = [50, 100, 200, 300, 500, 750, 1000]
    times  = []
    counts = []
    for n in sizes:
        subset = positions[:n]
        _, t = naive_vectorized(subset)
        pairs = int(n*(n-1)/2)
        times.append(t)
        counts.append(pairs)
        print(f"  N={n:5d} | pairs={pairs:12,d} | time={t:.4f}s")
    return sizes, times, counts

def plot_naive_results(sizes, times, counts):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Phase 3 - Naive O(N^2) Baseline Performance", fontsize=13)
    axes[0].plot(sizes, times, "o-", color="red", linewidth=2)
    axes[0].set_title("Execution Time vs Object Count")
    axes[0].set_xlabel("Number of Satellites (N)")
    axes[0].set_ylabel("Time (seconds)")
    axes[1].plot(sizes, counts, "s-", color="darkorange", linewidth=2)
    axes[1].set_title("Pairs Checked (O(N^2) Growth)")
    axes[1].set_xlabel("Number of Satellites (N)")
    axes[1].set_ylabel("Pairs Checked")
    axes[2].plot(counts, times, "^-", color="purple", linewidth=2)
    axes[2].set_title("Time vs Pairs (Linearity Check)")
    axes[2].set_xlabel("Pairs Checked")
    axes[2].set_ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig("outputs/phase3_naive_benchmark.png", dpi=150)
    plt.show()
    print("Saved: outputs/phase3_naive_benchmark.png")

if __name__ == "__main__":
    positions = np.load(POS_PATH)
    print(f"Loaded {len(positions)} satellite positions")
    print(f"Threshold: {THRESHOLD_KM} km\n")
    print("Benchmarking naive O(N^2) method...")
    sizes, times, counts = benchmark_naive(positions)
    print(f"\nAt N=1000: {counts[-1]:,} pairs, {times[-1]:.3f}s")
    print(f"Projected N=25000: ~{25000*24999//2:,} pairs - impractical!")
    plot_naive_results(sizes, times, counts)


    
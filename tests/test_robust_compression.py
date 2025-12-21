import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainselpy.solution import Solution
from trainselpy.distributional_head import (
    ParticleDistribution,
    compress_top_k,
    compress_resampling,
    compress_kmeans
)

def test_robust_compression():
    # Setup a distribution
    particles = []
    for i in range(10):
        sol = Solution(int_values=[np.array([i])], dbl_values=[])
        particles.append(sol)
    
    weights = np.arange(1, 11) / np.sum(np.arange(1, 11)) # [1/55, 2/55, ..., 10/55]
    dist = ParticleDistribution(particles, weights)
    
    print("\n" + "="*50)
    print("TESTING ROBUST COMPRESSION FUNCTIONS")
    print("="*50)
    
    # --- Test 1: compress_top_k with distribution and 'k' ---
    print("\nTest 1: compress_top_k(dist, k=5)")
    res1 = compress_top_k(dist, k=5)
    print(f"  Result type: {type(res1)}")
    print(f"  Result K: {res1.K}")
    assert isinstance(res1, ParticleDistribution)
    assert res1.K == 5
    # Particles should be the last 5 (highest weights)
    particle_vals = [p.int_values[0][0] for p in res1.particles]
    print(f"  Particle values: {particle_vals}")
    assert all(val in [5, 6, 7, 8, 9] for val in particle_vals)
    
    # --- Test 2: compress_resampling with raw data and 'K' ---
    print("\nTest 2: compress_resampling(particles, weights, K=3)")
    res2_p, res2_w = compress_resampling(particles, weights, K=3)
    print(f"  Result types: {type(res2_p)}, {type(res2_w)}")
    print(f"  Result counts: {len(res2_p)}, {len(res2_w)}")
    assert isinstance(res2_p, list)
    assert isinstance(res2_w, np.ndarray)
    assert len(res2_p) == 3
    
    # --- Test 3: compress_kmeans (delegates to top_k) with 'k' ---
    print("\nTest 3: compress_kmeans(dist, k=2)")
    res3 = compress_kmeans(dist, k=2)
    print(f"  Result K: {res3.K}")
    assert isinstance(res3, ParticleDistribution)
    assert res3.K == 2
    
    # --- Test 4: Regression test for lowercase 'k' in notebook example ---
    print("\nTest 4: Notebook simulation dist_topk = compress_top_k(dist, k=5)")
    try:
        dist_topk = compress_top_k(dist, k=5)
        print("  SUCCESS: compress_top_k(dist, k=5) worked!")
    except TypeError as e:
        print(f"  FAILURE: {e}")
        raise e

    print("\nALL TESTS PASSED!")

if __name__ == "__main__":
    test_robust_compression()

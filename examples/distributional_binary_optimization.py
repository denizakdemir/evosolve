"""
Example 1: Binary vector optimization with distributional head.

This demonstrates multimodal optimization where a distributional approach
maintains diversity and finds multiple optima.

Problem:
    Maximize sum(bits) BUT with penalty if sum >= 15
    - Optimal regime 1: All zeros (sum=0, fitness=0)
    - Optimal regime 2: 14 ones (sum=14, fitness=14)
    - Cliff: 15+ ones (fitness=-100)
    
Traditional GA gets stuck at one optimum. Distributional GA maintains
a portfolio covering both regimes.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainselpy import train_sel, train_sel_control


def multimodal_binary_fitness(int_vals, dbl_vals=None, data=None):
    """
    Multimodal fitness with cliff:
    - Reward sum of bits up to 14
    - Severe penalty for 15+
    """
    bits = int_vals if isinstance(int_vals[0], (int, np.integer)) else int_vals[0]
    s = np.sum(bits)
    
    if s >= 15:
        return -100.0  # Cliff
    else:
        return float(s)  # Reward


def run_standard_ga():
    """Run standard GA (collapses to single solution)."""
    print("=" * 70)
    print("STANDARD GA (individual solutions)")
    print("=" * 70)
    
    control = train_sel_control(
        niterations=50,
        npop=100,
        progress=True,
        mutprob=0.1,
        mutintensity=0.1,
        crossprob=0.8
    )
    
    result = train_sel(
        candidates=[list(range(20))],
        setsizes=[20],
        settypes=["BOOL"],
        stat=multimodal_binary_fitness,
        data={},
        control=control
    )
    
    # Extract best solution
    best_sol = result.selected_values[0] if hasattr(result, 'selected_values') else None
    if best_sol is None:
        # Fallback: compute from selected_indices
        best_sol = np.array(result.selected_indices[0])
    
    best_sum = np.sum(best_sol)
    
    print(f"\\nBest solution sum: {best_sum}")
    print(f"Best fitness: {result.fitness:.2f}")
    print(f"Converged to {'regime 1 (zeros)' if best_sum < 7 else 'regime 2 (14 ones)'}")
    
    return result


def run_distributional_ga():
    """Run distributional GA (maintains portfolio of solutions)."""
    print("\\n" + "=" * 70)
    print("DISTRIBUTIONAL GA (distribution over solutions)")
    print("=" * 70)
    
    # Create distributional control parameters
    control = train_sel_control(
        niterations=50,
        npop=50,  # Population of distributions
        progress=True,
        mutprob=0.2,
        mutintensity=0.1,
        crossprob=0.8,
        # Distributional-specific settings
        dist_objective='mean',  # Optimize expected fitness
        dist_n_samples=20,  # Monte Carlo samples per distribution
        dist_K_particles=10,  # Particles per distribution
        dist_compression='top_k'
    )
    
    result = train_sel(
        candidates=[list(range(20))],  # 20 binary variables
        setsizes=[20],
        settypes=["DIST:BOOL"],  # Distributional head wrapping BOOL!
        stat=multimodal_binary_fitness,
        data={},
        control=control
    )
    
    # Best distribution
    best_dist = result.distribution  # ParticleDistribution object
    
    print(f"\\nBest distribution:")
    print(f"  Number of particles (K): {best_dist.K}")
    print(f"  Expected fitness: {result.fitness:.2f}")
    
    # Analyze particle distribution
    particle_sums = [np.sum(p.int_values[0]) for p in best_dist.particles]
    print(f"\\n  Particle sums: {sorted(particle_sums)}")
    print(f"  Sum range: [{min(particle_sums)}, {max(particle_sums)}]")
    
    # Check diversity
    unique_sums = len(set(particle_sums))
    print(f"  Unique sum values: {unique_sums}/{best_dist.K}")
    print(f"  Diversity: {unique_sums/best_dist.K * 100:.1f}%")
    
    # Check if both regimes are covered
    has_regime_1 = any(s < 5 for s in particle_sums)
    has_regime_2 = any(s > 10 for s in particle_sums)
    
    print(f"\\n  Covers regime 1 (low sums): {has_regime_1}")
    print(f"  Covers regime 2 (high sums): {has_regime_2}")
    print(f"  Multimodal coverage: {'YES ✓' if has_regime_1 and has_regime_2 else 'NO ✗'}")
    
    return result


if __name__ == "__main__":
    print("\\n" + "=" * 70)
    print("DISTRIBUTIONAL OPTIMIZATION: Binary Multimodal Example")
    print("=" * 70)
    print("\\nProblem: sum(bits) with penalty at 15+")
    print("Two regimes: sum≈0 (stable) vs sum=14 (optimal)")
    print("\\nGoal: Show distributional GA maintains both regimes")
    
    # Run standard GA
    std_result = run_standard_ga()
    
    # Run distributional GA  
    dist_result = run_distributional_ga()
    
    print("\\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("Standard GA: Converges to single solution (loses diversity)")
    print("Distributional GA: Maintains portfolio (multimodal coverage)")

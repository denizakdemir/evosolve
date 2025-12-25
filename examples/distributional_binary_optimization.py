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

New Features Demonstrated (Phase 1-4 Improvements):
- Quick-start presets: 'exploratory' preset for multimodal optimization
- Configuration validation with helpful warnings
- Multi-objective combining with dist_use_nsga_means (mean + diversity)
- CMA-ES compatibility checking
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evosolve import train_sel, train_sel_control
from evosolve.core import get_distributional_preset  # NEW: Quick-start presets


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
    # selected_indices holds the BOOL/INT values, selected_values holds DBL values
    if result.selected_values and len(result.selected_values) > 0:
        best_sol = result.selected_values[0]
    elif result.selected_indices and len(result.selected_indices) > 0:
        best_sol = np.array(result.selected_indices[0])
    else:
        best_sol = None
    
    if best_sol is None:
        print("Error: Could not extract best solution")
        return result
    
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

    # NEW: Use 'exploratory' preset for multimodal optimization
    # This preset uses entropy regularization to maintain diversity
    print("\\nUsing 'exploratory' preset (entropy-regularized for multimodal)...")
    preset_config = get_distributional_preset('exploratory')
    print(f"  Preset config: objective={preset_config['dist_objective']}, "
          f"tau={preset_config.get('dist_tau', 'N/A')}")

    # Create distributional control parameters
    control = train_sel_control(
        niterations=50,
        npop=50,  # Population of distributions
        progress=True,
        mutprob=0.2,
        mutintensity=0.1,
        crossprob=0.8,
        # Apply preset configuration
        **preset_config
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


def run_distributional_multiobjective():
    """NEW: Run distributional GA with multi-objective optimization.

    This demonstrates dist_use_nsga_means which combines:
    - Mean objective (performance)
    - Distributional metric (diversity via entropy)
    This creates a Pareto front trading off performance vs diversity.
    """
    print("\n" + "=" * 70)
    print("DISTRIBUTIONAL MULTI-OBJECTIVE (Mean + Diversity)")
    print("=" * 70)

    # Use entropy objective with NSGA means mode
    control = train_sel_control(
        niterations=50,
        npop=50,
        progress=True,
        mutprob=0.2,
        mutintensity=0.1,
        crossprob=0.8,
        use_cma_es=False,  # CMA-ES incompatible with distributional
        # Multi-objective: optimize BOTH mean AND entropy
        dist_objective='entropy',  # Distributional metric
        dist_tau=0.3,  # Entropy weight
        dist_use_nsga_means=True,  # NEW: Combine mean + entropy in Pareto
        dist_K_particles=10,
        dist_n_samples=20,
        dist_maximize=True
    )

    result = train_sel(
        candidates=[list(range(20))],
        setsizes=[20],
        settypes=["DIST:BOOL"],
        stat=multimodal_binary_fitness,
        data={},
        control=control
    )

    best_dist = result.distribution

    print(f"\nBest distribution (Pareto-optimal):")
    print(f"  Number of particles (K): {best_dist.K}")
    print(f"  Combined fitness: {result.fitness:.2f}")
    print(f"  (Optimizes BOTH expected fitness AND diversity)")

    # Analyze particle distribution
    particle_sums = [np.sum(p.int_values[0]) for p in best_dist.particles]
    print(f"\n  Particle sums: {sorted(particle_sums)}")
    unique_sums = len(set(particle_sums))
    print(f"  Diversity: {unique_sums}/{best_dist.K} unique values ({unique_sums/best_dist.K*100:.1f}%)")

    return result


def demonstrate_validation():
    """NEW: Demonstrate configuration validation and error checking."""
    print("\n" + "=" * 70)
    print("CONFIGURATION VALIDATION (Error Prevention)")
    print("=" * 70)

    # Example 1: Incompatible CMA-ES + Distributional
    print("\n1. Testing CMA-ES + Distributional compatibility check...")
    try:
        control = train_sel_control(
            niterations=5,
            use_cma_es=True,  # This is incompatible with distributional
            dist_K_particles=10,
            progress=False
        )
        result = train_sel(
            candidates=[list(range(10))],
            setsizes=[10],
            settypes=["DIST:BOOL"],
            stat=multimodal_binary_fitness,
            data={},
            control=control
        )
    except ValueError as e:
        print(f"  ✓ Caught incompatibility: {e}")

    # Example 2: Mixed schema validation
    print("\n2. Testing mixed schema validation...")
    try:
        from evosolve.distributional_operators import initialize_distributional_population
        control = train_sel_control(dist_K_particles=3)
        pop = initialize_distributional_population(
            candidates=[list(range(10)), list(range(5))],
            setsizes=[5, 3],
            settypes=["DIST:BOOL", "INT"],  # Mixed: distributional + standard
            pop_size=10,
            control=control
        )
    except NotImplementedError as e:
        print(f"  ✓ Caught mixed schema: {e}")

    print("\n  Configuration validation prevents common errors!")


if __name__ == "__main__":
    print("\\n" + "=" * 70)
    print("DISTRIBUTIONAL OPTIMIZATION: Binary Multimodal Example")
    print("=" * 70)
    print("\\nProblem: sum(bits) with penalty at 15+")
    print("Two regimes: sum≈0 (stable) vs sum=14 (optimal)")
    print("\\nGoal: Show distributional GA maintains both regimes")
    
    # Run standard GA
    std_result = run_standard_ga()

    # Run distributional GA with exploratory preset
    dist_result = run_distributional_ga()

    # NEW: Run distributional multi-objective (mean + diversity)
    moo_result = run_distributional_multiobjective()

    # NEW: Demonstrate validation features
    demonstrate_validation()

    print("\\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("Standard GA: Converges to single solution (loses diversity)")
    print("Distributional GA: Maintains portfolio (multimodal coverage)")
    print("NEW: Multi-objective mode optimizes BOTH performance AND diversity")
    print("NEW: Configuration validation prevents common errors")

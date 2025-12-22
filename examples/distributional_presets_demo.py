"""
Comprehensive demonstration of distributional optimization presets.

This example showcases all 4 quick-start presets introduced in Phase 4.3:
1. 'robust' - Conservative CVaR optimization (worst-case focus)
2. 'risk_averse' - Mean-Variance tradeoff (balance performance & stability)
3. 'exploratory' - Entropy-regularized (maximize diversity)
4. 'performance' - Pure mean optimization (maximize expected fitness)

Problem: Portfolio optimization with 10 assets
- Maximize return (mean of selected assets)
- Different presets handle risk/uncertainty differently
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainselpy import train_sel, train_sel_control
from trainselpy.core import get_distributional_preset


# Simulated asset returns (mean, std)
ASSETS = [
    (0.15, 0.25),  # High risk, high return
    (0.12, 0.20),
    (0.10, 0.15),
    (0.08, 0.12),
    (0.06, 0.10),
    (0.05, 0.08),  # Low risk, low return
    (0.04, 0.06),
    (0.03, 0.05),
    (0.02, 0.04),
    (0.01, 0.02),
]


def portfolio_fitness(int_vals, dbl_vals=None, data=None):
    """
    Fitness = mean return of selected assets.

    This is the base fitness. Different distributional objectives
    will aggregate this in different ways:
    - 'robust': Focus on worst-case scenarios (CVaR)
    - 'risk_averse': Penalize variance
    - 'exploratory': Add diversity bonus
    - 'performance': Pure mean
    """
    selected = int_vals if isinstance(int_vals[0], (int, np.integer)) else int_vals[0]

    # Calculate portfolio return
    total_return = 0.0
    for i, is_selected in enumerate(selected):
        if is_selected:
            # Sample from asset's return distribution
            mean_ret, std_ret = ASSETS[i]
            sampled_return = np.random.normal(mean_ret, std_ret)
            total_return += sampled_return

    return float(total_return)


def run_preset_comparison():
    """Compare all 4 presets on the same problem."""
    print("=" * 80)
    print("DISTRIBUTIONAL PRESETS COMPARISON")
    print("=" * 80)
    print("\nProblem: Select 5 assets from 10 to maximize portfolio return")
    print("Assets range from high-risk/high-return to low-risk/low-return\n")

    presets = ['robust', 'risk_averse', 'exploratory', 'performance']
    results = {}

    for preset_name in presets:
        print("-" * 80)
        print(f"PRESET: {preset_name.upper()}")
        print("-" * 80)

        # Get preset configuration
        preset_config = get_distributional_preset(preset_name)

        # Print configuration details
        print(f"\nConfiguration:")
        print(f"  dist_objective: {preset_config['dist_objective']}")
        if 'dist_alpha' in preset_config:
            print(f"  dist_alpha: {preset_config['dist_alpha']} (CVaR quantile)")
        if 'dist_lambda_var' in preset_config:
            print(f"  dist_lambda_var: {preset_config['dist_lambda_var']} (variance penalty)")
        if 'dist_tau' in preset_config:
            print(f"  dist_tau: {preset_config['dist_tau']} (entropy weight)")
        if 'dist_use_nsga_means' in preset_config:
            print(f"  dist_use_nsga_means: {preset_config['dist_use_nsga_means']} (multi-objective)")

        # Run optimization
        control = train_sel_control(
            niterations=30,
            npop=30,
            progress=False,
            mutprob=0.15,
            mutintensity=0.1,
            crossprob=0.8,
            **preset_config  # Apply preset
        )

        result = train_sel(
            candidates=[list(range(10))],
            setsizes=[5],  # Select 5 assets
            settypes=["DIST:BOOL"],
            stat=portfolio_fitness,
            data={},
            control=control
        )

        results[preset_name] = result

        # Analyze results
        best_dist = result.distribution
        print(f"\nResults:")
        print(f"  Fitness: {result.fitness:.4f}")
        print(f"  Particles (K): {best_dist.K}")

        # Analyze particle diversity
        particle_selections = []
        for p in best_dist.particles:
            selected_assets = [i for i, val in enumerate(p.int_values[0]) if val]
            particle_selections.append(tuple(sorted(selected_assets)))

        unique_selections = len(set(particle_selections))
        print(f"  Unique portfolios: {unique_selections}/{best_dist.K}")
        print(f"  Diversity: {unique_selections/best_dist.K * 100:.1f}%")

        # Analyze risk profile (count high-risk assets)
        high_risk_counts = []
        for p in best_dist.particles:
            high_risk = sum(p.int_values[0][:3])  # First 3 are high-risk
            high_risk_counts.append(high_risk)

        avg_high_risk = np.mean(high_risk_counts)
        print(f"  Avg high-risk assets: {avg_high_risk:.2f}/3")

        # Show most common portfolio
        from collections import Counter
        most_common = Counter(particle_selections).most_common(1)[0]
        print(f"  Most common portfolio: {list(most_common[0])} (weight: {most_common[1]/best_dist.K:.2f})")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    print("\nFitness Comparison:")
    for preset_name in presets:
        result = results[preset_name]
        print(f"  {preset_name:15s}: {result.fitness:8.4f}")

    print("\nDiversity Comparison:")
    for preset_name in presets:
        result = results[preset_name]
        best_dist = result.distribution
        particle_selections = []
        for p in best_dist.particles:
            selected = tuple(sorted([i for i, v in enumerate(p.int_values[0]) if v]))
            particle_selections.append(selected)
        unique = len(set(particle_selections))
        print(f"  {preset_name:15s}: {unique}/{best_dist.K} unique ({unique/best_dist.K*100:.0f}%)")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("• 'robust': Highest diversity, conservative portfolios (CVaR optimization)")
    print("• 'risk_averse': Balanced, avoids high variance (Mean-Variance tradeoff)")
    print("• 'exploratory': Maximum diversity, explores all options (Entropy bonus)")
    print("• 'performance': May collapse to single best, highest expected return")
    print("\nChoose preset based on your optimization goals:")
    print("  - Worst-case robustness → 'robust'")
    print("  - Stability/consistency → 'risk_averse'")
    print("  - Exploration/diversity → 'exploratory'")
    print("  - Pure performance → 'performance'")


def demonstrate_preset_customization():
    """Show how to customize presets with overrides."""
    print("\n" + "=" * 80)
    print("PRESET CUSTOMIZATION")
    print("=" * 80)

    print("\nYou can override any preset parameter:")

    # Start with 'robust' but make it more aggressive
    custom_config = get_distributional_preset('robust', dist_alpha=0.5)
    print(f"\nCustom 'robust' with alpha=0.5 (less conservative):")
    print(f"  dist_objective: {custom_config['dist_objective']}")
    print(f"  dist_alpha: {custom_config['dist_alpha']}")

    # Start with 'exploratory' but increase entropy weight
    custom_config2 = get_distributional_preset('exploratory', dist_tau=1.0)
    print(f"\nCustom 'exploratory' with tau=1.0 (more diversity):")
    print(f"  dist_objective: {custom_config2['dist_objective']}")
    print(f"  dist_tau: {custom_config2['dist_tau']}")

    print("\nThis allows you to start with a good baseline and fine-tune!")


def demonstrate_validation():
    """Show configuration validation in action."""
    print("\n" + "=" * 80)
    print("CONFIGURATION VALIDATION")
    print("=" * 80)

    import warnings

    print("\n1. Testing mismatched parameter warning...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Set CVaR alpha but use entropy objective
        control = train_sel_control(
            dist_objective='entropy',
            dist_alpha=0.2,  # This is for CVaR, not entropy!
            dist_K_particles=5,
            niterations=2,
            progress=False
        )
        if len(w) > 0:
            print(f"  ✓ Warning caught: {w[0].message}")

    print("\n2. Testing invalid objective...")
    try:
        control = train_sel_control(
            dist_objective='invalid_objective',
            dist_K_particles=5
        )
    except ValueError as e:
        print(f"  ✓ Error caught: {e}")

    print("\n  Configuration validation helps prevent common mistakes!")


if __name__ == "__main__":
    # Main comparison
    run_preset_comparison()

    # Show customization
    demonstrate_preset_customization()

    # Show validation
    demonstrate_validation()

    print("\n" + "=" * 80)
    print("QUICK-START GUIDE")
    print("=" * 80)
    print("""
To use a preset in your code:

    from trainselpy.core import get_distributional_preset

    # Option 1: Use preset directly
    preset = get_distributional_preset('robust')
    control = train_sel_control(**preset)

    # Option 2: Customize a preset
    preset = get_distributional_preset('exploratory', dist_tau=1.0)
    control = train_sel_control(niterations=100, **preset)

Available presets: 'robust', 'risk_averse', 'exploratory', 'performance'
""")

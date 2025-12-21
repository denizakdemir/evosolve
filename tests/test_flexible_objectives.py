import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainselpy.solution import Solution
from trainselpy.distributional_head import (
    ParticleDistribution,
    mean_objective,
    mean_variance_objective,
    cvar_objective,
    entropy_regularized_objective
)

def test_flexible_objectives():
    # 1. Setup a simple distribution
    # 3 particles with fitness values 10, 20, 30
    p1 = Solution(int_values=[np.array([10])], dbl_values=[])
    p2 = Solution(int_values=[np.array([20])], dbl_values=[])
    p3 = Solution(int_values=[np.array([30])], dbl_values=[])
    
    # Weights: [0.2, 0.5, 0.3]
    dist = ParticleDistribution(particles=[p1, p2, p3], weights=np.array([0.2, 0.5, 0.3]))
    
    # Mock fitness function
    def mock_fitness(int_vals, dbl_vals, data):
        return float(int_vals[0][0])
    
    # Pre-computed fitness array
    fitness_array = np.array([10.0, 20.0, 30.0])
    
    print("\n" + "="*50)
    print("TESTING FLEXIBLE OBJECTIVES")
    print("="*50)
    
    # --- Mean Objective ---
    print("\nMean Objective:")
    mean_callable = mean_objective(dist, mock_fitness, n_samples=1000)
    mean_array = mean_objective(dist, fitness_array)
    expected_mean = 0.2*10 + 0.5*20 + 0.3*30 # 2 + 10 + 9 = 21
    print(f"  Callable (MC): {mean_callable:.2f}")
    print(f"  Array (Exact): {mean_array:.2f} (Expected: {expected_mean:.2f})")
    assert abs(mean_array - expected_mean) < 1e-6
    
    # --- Mean-Variance Objective ---
    print("\nMean-Variance Objective (lambda=1.0):")
    mv_callable = mean_variance_objective(dist, mock_fitness, n_samples=1000, lambda_var=1.0)
    mv_array = mean_variance_objective(dist, fitness_array, lambda_var=1.0)
    # Variance = E[X^2] - E[X]^2
    # E[X^2] = 0.2*100 + 0.5*400 + 0.3*900 = 20 + 200 + 270 = 490
    # Var = 490 - 21^2 = 490 - 441 = 49
    # Expected MV = 21 + 1.0 * 49 = 70
    expected_mv = 70.0
    print(f"  Callable (MC): {mv_callable:.2f}")
    print(f"  Array (Exact): {mv_array:.2f} (Expected: {expected_mv:.2f})")
    assert abs(mv_array - expected_mv) < 1e-6
    
    # --- CVaR Objective ---
    print("\nCVaR Objective (alpha=0.4, maximize=True):")
    # alpha=0.4 means worst 40% (lowest values)
    # Particle 1 (w=0.2, val=10) and PART of particle 2 (w=0.2 from w=0.5, val=20)
    # CVaR = (0.2 * 10 + 0.2 * 20) / 0.4 = (2 + 4) / 0.4 = 6 / 0.4 = 15
    expected_cvar = 15.0
    cvar_callable = cvar_objective(dist, mock_fitness, n_samples=1000, alpha=0.4)
    cvar_array = cvar_objective(dist, fitness_array, alpha=0.4)
    print(f"  Callable (MC): {cvar_callable:.2f}")
    print(f"  Array (Exact): {cvar_array:.2f} (Expected: {expected_cvar:.2f})")
    assert abs(cvar_array - expected_cvar) < 1e-6
    
    # --- Entropy Regularized Objective ---
    print("\nEntropy Regularized Objective (tau=1.0):")
    # Entropy = -(0.2*log(0.2) + 0.5*log(0.5) + 0.3*log(0.3))
    # = - (0.2 * -1.609 + 0.5 * -0.693 + 0.3 * -1.204)
    # = - (-0.322 - 0.347 - 0.361) = 1.03
    entropy = -(0.2*np.log(0.2) + 0.5*np.log(0.5) + 0.3*np.log(0.3))
    expected_ent_reg = expected_mean + 1.0 * entropy
    ent_callable = entropy_regularized_objective(dist, mock_fitness, n_samples=1000, tau=1.0)
    ent_array = entropy_regularized_objective(dist, fitness_array, tau=1.0)
    print(f"  Callable (MC): {ent_callable:.2f}")
    print(f"  Array (Exact): {ent_array:.2f} (Expected: {expected_ent_reg:.2f})")
    assert abs(ent_array - expected_ent_reg) < 1e-6
    
    print("\nALL TESTS PASSED!")

if __name__ == "__main__":
    test_flexible_objectives()

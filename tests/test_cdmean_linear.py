
import numpy as np
import pytest
from trainselpy.core import make_data
from trainselpy.optimization_criteria import cdmean_opt, cdmean_opt_linear

def test_cdmean_linear_dispatch():
    """
    Test that cdmean_opt correctly dispatches to cdmean_opt_linear
    when L is present in the data.
    """
    # Create random G matrix (positive definite)
    np.random.seed(42)
    N = 50
    X_feat = np.random.randn(N, 100)
    G = np.dot(X_feat, X_feat.T) / 100
    # Make strictly positive definite
    G += np.eye(N) * 0.01
    
    # Create L matrix (contrast matrix)
    # 5 contrasts
    L = np.random.randn(5, N)
    
    # Create data with L
    data_with_L = make_data(K=G, L=L, lambda_val=1.0)
    
    # Create data without L
    data_no_L = make_data(K=G, lambda_val=1.0)
    
    # Define a solution (subset of indices)
    soln = list(range(10)) # Select first 10 individuals
    
    # Calculate CDMEAN with L using cdmean_opt (should dispatch)
    val_dispatch = cdmean_opt(soln, data_with_L)
    
    # Calculate CDMEAN with L explicitly
    val_explicit = cdmean_opt_linear(soln, data_with_L)
    
    # Calculate standard CDMEAN (should use different logic)
    val_standard = cdmean_opt(soln, data_no_L)
    
    # Assert dispatch works
    assert val_dispatch == val_explicit
    
    # Assert standard logic produces different result (highly likely)
    assert val_dispatch != val_standard
    
    # Assert value range (0 to 1 ideally, though CD can theoretically be weird if G is not perfect, usually < 1)
    assert val_dispatch > 0
    assert val_dispatch < 1.0
    
    print(f"CDMEAN (Linear): {val_dispatch}")
    print(f"CDMEAN (Standard): {val_standard}")

def test_cdmean_linear_math_consistency():
    """
    Test that cdmean_opt_linear produces consistent results
    mathematically with a small example.
    """
    # Small example
    G = np.eye(4)
    # L selects just the 4th individual (index 3)
    # L = [0, 0, 0, 1]
    L = np.zeros((1, 4))
    L[0, 3] = 1.0
    
    # Solution selects indices 0, 1
    soln = [0, 1]
    
    # lambda = 1
    # V = I_2 + I_2 = 2*I_2
    # V_inv = 0.5 * I_2
    # G = I_4
    # G_L = L @ G = [0, 0, 0, 1]
    # G_L_soln = [0, 0] (columns 0 and 1 of G_L)
    
    # Since G_L_soln is all zeros (because L targets individual 3, but we selected 0 and 1 who are uncorrelated with 3),
    # the prediction should be zero, and reliability derived from prediction variance should be zero?
    # Wait, existing formula:
    # term1_diag = sum(G_L_soln * V_inv_G_L, axis=1)
    # If G_L_soln is 0, term1 is 0.
    # w = G_L_soln @ V_inv_1 = 0
    # term2 is 0.
    # Numerator is 0.
    # Reliability is 0.
    
    data = make_data(K=G, L=L, lambda_val=1.0)
    val = cdmean_opt_linear(soln, data)
    
    assert np.isclose(val, 0.0)
    
    # Now select index 3 (the one L targets)
    soln2 = [3]
    # V = 1 + 1 = 2
    # V_inv = 0.5
    # G_L = [0, 0, 0, 1]
    # G_L_soln = [1]
    # w = 1 * 0.5 = 0.5
    # sum_V_inv = 1 * 0.5 = 0.5
    # V_inv_G_L = 0.5 * 1 = 0.5
    
    # Term 1: 1 * 0.5 = 0.5
    # Term 2: (0.5^2) / 0.5 = 0.5
    # Result: 0.5 - 0.5 = 0.0??
    
    # Wait. The formula includes the mean correction.
    # If we predict just the mean, and the selected individual is perfectly correlated 
    # but we account for the fixed mean effect...
    # Actually, CD reliability usually = 1 - PEV/Var(u).
    # The formula `diag(G_all_soln @ (V_inv - V_inv_2) @ G_all_soln.T)` calculates the variance of the *predicted* values.
    # For a selected individual in the training set, should reliability be 1?
    # In standard CDMEAN, we exclude selected individuals from the mean.
    # In cdmean_opt_linear, we average over L.
    # If L targets an individual that IS in the solution, its reliability should be high.
    
    # Let's trace why it became 0.
    # The term (V_inv - V_inv_2) projects onto the orthogonal complement of the 1 vector in the metric V.
    # If we only have 1 selected individual, we cannot estimate variance + mean?
    # If we fit fixed effect (mean), we lose 1 degree of freedom.
    # With 1 observation, we fit the mean perfectly, so residual is 0.
    # We can't distinguish genetic value from mean?
    # Correct. You need at least 2 observations or prior info to separate mean from genetic effect 
    # if you treat mean as fixed.
    # So reliability for 1 selected individual is indeed 0 if you correct for mean.
    
    # Try selecting 2 individuals: 3 and 2.
    # L targets 3.
    # G = I.
    # soln = [2, 3]
    # L = [0, 0, 0, 1]
    # G_L = [0, 0, 0, 1]
    # G_L_soln = [0, 1] (cols 2, 3)
    
    # V = 2 * I_2.
    # V_inv = 0.5 * I_2.
    # V_inv_1 = 0.5 * [1, 1] = [0.5, 0.5]
    # sum_V_inv = 1.0
    
    # w = G_L_soln @ V_inv_1 = [0, 1] @ [0.5, 0.5] = 0.5
    
    # V_inv_G_L = V_inv @ [0, 1].T = [0, 0.5].T
    # Term 1 (diag of G_L_soln @ V_inv @ G_L_soln.T)
    # = [0, 1] dot [0, 0.5] = 0.5
    
    # Term 2 = w^2 / sum_V_inv = 0.5^2 / 1.0 = 0.25
    
    # Result = 0.5 - 0.25 = 0.25.
    
    # Var(target) = LGL' = 1.
    # Reliability = 0.25.
    
    val2 = cdmean_opt_linear([2, 3], data)
    assert np.isclose(val2, 0.25)
    print(f"CDMEAN (Linear, small example): {val2}")

if __name__ == "__main__":
    test_cdmean_linear_dispatch()
    test_cdmean_linear_math_consistency()

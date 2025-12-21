
"""
Benchmark for Convex Relaxations in TrainSelPy.

Compares the performance (Time and Solution Quality) of:
1. Frank-Wolfe (FW) Solver
2. SLSQP Solver (Scipy)
3. Random Uniform Baseline (Reference)

Criteria Benchmarked:
- D-Optimality
- A-Optimality
- CDMean-Optimality
- PEV-Optimality
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from trainselpy.relaxations import (
    DOptimality, 
    AOptimality, 
    CDMeanOptimality, 
    PEVOptimality, 
    ConvexRelaxationSolver
)

def generate_data(n_samples: int = 100, n_features: int = 10):
    """Generate synthetic data for benchmarking."""
    np.random.seed(42)
    # Feature matrix X
    X = np.random.randn(n_samples, n_features)
    
    # Relationship matrix G (assume Identity + some structure for realism)
    # G = XX' + I (simple PD kernel)
    G = X @ X.T + 0.1 * np.eye(n_samples)
    # Normalize G diagonal to be approx 1 like genomic relationships
    d = np.diag(G)
    G = G / np.mean(d)
    
    return X, G

def run_benchmark(n_samples: int = 200, k: int = 50, n_features: int = 10):
    print(f"\n--- Benchmarking N={n_samples}, K={k}, Features={n_features} ---")
    
    X, G = generate_data(n_samples, n_features)
    
    results = []
    
    # Define benchmarks: (Name, Criterion_Class, Data_Arg, Solver_Methods)
    benchmarks = [
        ("D-Optimality", DOptimality, X, ["FW", "SLSQP"]),
        ("A-Optimality", AOptimality, X, ["FW", "SLSQP"]),
        ("CDMean", CDMeanOptimality, G, ["FW"]), # SLSQP often too slow/unstable for custom gradients on large matrices
        ("PEV", PEVOptimality, G, ["FW"])
    ]
    
    baseline_w = np.ones(n_samples) * (k / n_samples)
    
    for name, CritClass, data, methods in benchmarks:
        print(f"\nRunning {name}...")
        
        # Initialize Criterion
        if name in ["CDMean", "PEV"]:
            crit = CritClass(G_matrix=data, lambda_val=1.0)
            eval_data = None # X not needed for evaluate
        else:
            crit = CritClass()
            eval_data = data # X needed for evaluate
            
        # 1. Baseline (Uniform)
        start = time.time()
        val_unif = crit.evaluate(baseline_w, eval_data)
        baseline_time = time.time() - start
        
        results.append({
            "Criterion": name,
            "Method": "Uniform (Baseline)",
            "Time (s)": baseline_time,
            "Objective": val_unif,
            "Improvement vs Random": 0.0
        })
        
        # 2. Solvers
        for method in methods:
            print(f"  -> Method: {method}")
            solver = ConvexRelaxationSolver(crit, method=method)
            
            start = time.time()
            try:
                # For CDMean/PEV, X is ignored by solve's gradient calls but required by signature, passing None is fine or dummy
                # The generic solve() takes X.
                # Only D/A opt use X in evaluate/gradient locally.
                # CDMean stores G internally.
                solve_data = data if name in ["D-Optimality", "A-Optimality"] else None
                
                w_opt = solver.solve(solve_data, k, w_init=None)
                elapsed = time.time() - start
                
                val_opt = crit.evaluate(w_opt, eval_data)
                
                # Check constraints roughly
                sum_w = np.sum(w_opt)
                min_w = np.min(w_opt)
                max_w = np.max(w_opt)
                
                # Improvement: 
                # Recall these criteria return "Minimization" objectives (-logdet, -trace, -CDMean).
                # So Lower is Better.
                # Example: Uniform gave -100. Opt gave -150. Improvement is beneficial.
                
                # For display, let's show raw values.
                
                results.append({
                    "Criterion": name,
                    "Method": method,
                    "Time (s)": elapsed,
                    "Objective": val_opt,
                    "Improvement vs Random": (val_unif - val_opt) # Positive means we reduced the minimization objective
                })
                
            except Exception as e:
                print(f"     Failed: {e}")
                results.append({
                    "Criterion": name,
                    "Method": method,
                    "Time (s)": np.nan,
                    "Objective": np.nan,
                    "Error": str(e)
                })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Small run for verification
    df_small = run_benchmark(n_samples=100, k=25, n_features=5)
    print("\nResults (N=100):")
    print(df_small.to_string())
    
    # Larger run
    df_large = run_benchmark(n_samples=300, k=50, n_features=10)
    print("\nResults (N=300):")
    print(df_large.to_string())
    
    # Save
    df_large.to_csv("examples/benchmarks/benchmark_relaxations_results.csv", index=False)
    print("\nSaved results to examples/benchmarks/benchmark_relaxations_results.csv")


import numpy as np
import matplotlib.pyplot as plt
from evosolve.algorithms import genetic_algorithm
import time
import os

# --- Problem Definitions ---

def trap_function_5(bits):
    """
    Deceptive Trap Function of order 5.
    Unitation u = number of 1s.
    If u = 5 -> fitness = 5 (Global Peak)
    If u < 5 -> fitness = 4 - u (Local Peak at u=0 with fitness 4)
    """
    u = sum(bits)
    if u == 5:
        return 5.0
    else:
        return 4.0 - u

def evaluate_mixed_fitness_moo(int_vals, dbl_vals, data):
    """
    Mixed Problem MOO:
    Obj 1: Favors Trap=1s, Perm=Sorted, Cont=0.
    Obj 2: Favors Trap=0s, Perm=Unsorted, Cont=2.
    
    This creates a true Pareto surface trading off all three variable types.
    """
    
    # --- 1. Binary (Trap) ---
    binary_vec = int_vals[0]
    # Trap score (favors blocks of 1s, specifically 11111)
    trap_score = 0
    block_size = 5
    for i in range(0, len(binary_vec), block_size):
        chunk = binary_vec[i:i+block_size]
        if len(chunk) == block_size:
            trap_score += trap_function_5(chunk)
            
    # Count Zeros (Conflict for Obj 2)
    zeros_score = len(binary_vec) - sum(binary_vec)
            
    # --- 2. Permutation ---
    perm_vec = int_vals[1]
    # Distance from Identity (0, 1, 2...)
    perm_dist = sum(abs(v - i) for i, v in enumerate(perm_vec))
    
    # --- 3. Continuous (Rastrigin vs Shifted Sphere) ---
    vals = dbl_vals[0]
    if hasattr(vals, '__len__'):
        x = np.array(vals)
    else:
        x = np.array([vals])
    n = len(x)
    
    # Rastrigin (Global min at 0). Obj1 wants to Minimize this (Maximize Negative)
    rast_val = 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    # Distance to 2.0. Obj2 wants to Minimize this (Maximize Negative)
    dist_to_2 = np.sum((x - 2.0)**2)
    
    # Objectives
    # Obj 1: "System Quality" -> High 1s, Sorted, x close to 0
    obj1 = (10.0 * trap_score) - (1.0 * perm_dist) - (1.0 * rast_val)
    
    # Obj 2: "Alternative Config" -> High 0s, Unsorted, x close to 2
    obj2 = (2.0 * zeros_score) + (1.0 * perm_dist) - (10.0 * dist_to_2)
    
    return [obj1, obj2]

def run_benchmark():
    print("--- Running Difficult Mixed-Type MOO Benchmark ---")
    
    cand_bin = list(range(30))
    cand_perm = list(range(10))
    cand_dbl = [] 
    
    candidates = [cand_bin, cand_perm, cand_dbl]
    settypes = ["BOOL", "UOS", "DBL"]
    setsizes = [30, 10, 10]
    
    # Enable MOO with n_stat=2
    n_stat = 2
    data = {"debug_print": False}
    
    # Common Control
    npop = 200
    ngen = 100 
    base_control = {
        "npop": npop,
        "niterations": ngen,
        "progress": True,
        "sigma": 0.5,
        "mutation_rate": 0.1,
        "verbose": False
    }
    
    # --- Run 1: Baseline (NSGA-II) ---
    print("\n[Run 1] Baseline Multi-Objective GA (NSGA-II)...")
    start_time = time.time()
    control_base = base_control.copy()
    control_base["use_vae"] = False
    control_base["use_gan"] = False
    
    res_base = genetic_algorithm(
        data, candidates, setsizes, settypes, evaluate_mixed_fitness_moo, 
        control=control_base, n_stat=n_stat
    )
    time_base = time.time() - start_time
    
    # In MOO, fitness is sum of objectives (for display), but we care about Pareto Front.
    # res['pareto_front'] contains list of [obj1, obj2] for non-dominated sol.
    pf_base = res_base.get('pareto_front', [])
    print(f"Baseline Time: {time_base:.2f}s")
    print(f"Baseline Pareto Front Size: {len(pf_base)}")
    # Print some points
    if pf_base:
        # Sort by obj1 for display
        sorted_pf = sorted(pf_base, key=lambda x: x[0])
        print("Sample Pareto Points (Obj1, Obj2):")
        for i in range(0, len(sorted_pf), max(1, len(sorted_pf)//5)):
            print(f"  {sorted_pf[i]}")
    
    # --- Run 2: With VAE + GAN ---
    print("\n[Run 2] Neural-Enhanced Multi-Objective GA (VAE+GAN)...")
    start_time = time.time()
    control_nn = base_control.copy()
    control_nn["use_vae"] = True
    control_nn["use_gan"] = True
    control_nn["vae_latent_dim"] = 8
    control_nn["gan_noise_dim"] = 8
    control_nn["nn_update_freq"] = 5
    control_nn["nn_start_gen"] = 5
    control_nn["nn_offspring_ratio"] = 0.3
    control_nn["nn_epochs"] = 50
    
    res_nn = genetic_algorithm(
        data, candidates, setsizes, settypes, evaluate_mixed_fitness_moo, 
        control=control_nn, n_stat=n_stat
    )
    time_nn = time.time() - start_time
    
    pf_nn = res_nn.get('pareto_front', [])
    print(f"NN-Enhanced Time: {time_nn:.2f}s")
    print(f"NN-Enhanced Pareto Front Size: {len(pf_nn)}")
    if pf_nn:
        sorted_pf = sorted(pf_nn, key=lambda x: x[0])
        print("Sample Pareto Points (Obj1, Obj2):")
        for i in range(0, len(sorted_pf), max(1, len(sorted_pf)//5)):
            print(f"  {sorted_pf[i]}")

    # --- Comparison (Hypervolume approximation) ---
    print("\n--- Summary ---")
    
    def simple_hv(front, ref):
        # Very simple HV: sum of areas of rectangles formed by point and reference
        # Assumes sorted front (descending obj1, ascending obj2 for maximization??)
        # For maximization, we want VOLUME ABOVE reference.
        # Let's assume minimization for standard HV code usually.
        # But we maximized. Ref point should be "worse" than all points.
        # Ref point e.g. [-50, -1000].
        # Just summing up raw fitness isn't great.
        # Let's just compare bestObj1 and bestObj2 found.
        best_obj1 = max([p[0] for p in front]) if front else -999
        best_obj2 = max([p[1] for p in front]) if front else -9999
        return best_obj1, best_obj2
        
    b1_base, b2_base = simple_hv(pf_base, None)
    b1_nn, b2_nn = simple_hv(pf_nn, None)
    
    print(f"Baseline Best Obj1: {b1_base:.2f}, Best Obj2: {b2_base:.2f}")
    print(f"NN-Enh   Best Obj1: {b1_nn:.2f}, Best Obj2: {b2_nn:.2f}")

if __name__ == "__main__":
    run_benchmark()

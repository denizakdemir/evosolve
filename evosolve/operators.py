"""
Genetic algorithm operators for TrainSelPy.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from evosolve.solution import Solution

def _get_valid_replacement(
    current_val: Optional[int],
    current_set: Set[int],
    candidates: List[int],
    set_type: str,
    max_attempts: int = 20
) -> int:
    """
    Select a valid replacement value using rejection sampling with fallback.
    """
    # For multisets (UOMS, OMS), any candidate is valid
    if set_type in ["UOMS", "OMS"]:
        return random.choice(candidates)

    # For unique sets (UOS, OS), we need a value not in current_set and not current_val
    # Rejection sampling
    n_candidates = len(candidates)
    n_taken = len(current_set)
    n_free = n_candidates - n_taken
    
    if n_free <= 0:
        return current_val if current_val is not None else random.choice(candidates)

    # Dynamic adjustment of attempts based on density
    if n_free < n_candidates:
        prob = n_free / n_candidates
        dynamic_limit = int(10.0 / prob)
        runs = min(dynamic_limit, max(200, n_candidates // 10))
    else:
        runs = 20
        
    for _ in range(runs):
        val = random.choice(candidates)
        if val != current_val and val not in current_set:
            return val
            
    # Fallback
    available = [c for c in candidates if c != current_val and c not in current_set]
    if not available:
        return current_val if current_val is not None else random.choice(candidates)
        
    return random.choice(available)


def _pmx_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partially Mapped Crossover (PMX) for ordered sets (permutations).
    Operates on numpy arrays.
    """
    size = len(parent1)
    
    cx_point1 = random.randint(0, size - 1)
    cx_point2 = random.randint(0, size - 1)
    
    if cx_point1 > cx_point2:
        cx_point1, cx_point2 = cx_point2, cx_point1
    
    # Initialize offspring as copies of parents
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Create mapping
    mapping1 = {}
    mapping2 = {}
    
    # Python loop is fine here as PMX is complex
    p1_segment = parent1[cx_point1:cx_point2+1]
    p2_segment = parent2[cx_point1:cx_point2+1]

    for i in range(len(p1_segment)):
        val1 = p1_segment[i]
        val2 = p2_segment[i]
        
        # Swap segment in children
        child1[cx_point1 + i] = val2
        child2[cx_point1 + i] = val1
        
        mapping1[val2] = val1
        mapping2[val1] = val2
    
    # Fix conflicts
    indices = list(range(0, cx_point1)) + list(range(cx_point2 + 1, size))
    for i in indices:
        val = child1[i]
        while val in mapping1:
            val = mapping1[val]
        child1[i] = val
        
        val = child2[i]
        while val in mapping2:
            val = mapping2[val]
        child2[i] = val
    
    return child1, child2


def _crossover_single_int_array(
    arr1: np.ndarray,
    arr2: np.ndarray,
    crossintensity: float,
    stype: str,
    candidates: Optional[List[int]]
) -> None:
    """Helper to crossover a single integer array."""
    size = len(arr1)
    if size <= 1:
        return

    if stype == "OS":
        # PMX Crossover
        # PMX returns new arrays, so update in place
        new1, new2 = _pmx_crossover(arr1, arr2)
        arr1[:] = new1
        arr2[:] = new2
        return

    # One-point or Multi-point crossover
    n_points = max(1, int(size * crossintensity))
    n_points = min(n_points, size - 1)
    if n_points <= 0:
        return

    points = np.sort(
        np.random.choice(np.arange(1, size, dtype=int), size=n_points, replace=False)
    )

    for k, point in enumerate(points):
        if k % 2 != 0:
            continue
        start = points[k - 1] if k > 0 else 0
        end = point
        
        # Swap
        tmp = arr1[start:end].copy()
        arr1[start:end] = arr2[start:end]
        arr2[start:end] = tmp

    # Post-processing / Repair
    if stype in ["UOS", "UOMS"]:
        arr1.sort()
        arr2.sort()

    if stype == "UOS" and candidates:
        # Repair duplicates for UOS
        for arr in (arr1, arr2):
            if len(arr) != len(np.unique(arr)):
                unique_values = []
                seen = set()
                for v in arr:
                    if v not in seen:
                        unique_values.append(v)
                        seen.add(v)
                
                current_set = seen
                n_missing = len(arr) - len(unique_values)
                
                for _ in range(n_missing):
                    new_val = _get_valid_replacement(None, current_set, candidates, stype)
                    unique_values.append(new_val)
                    current_set.add(new_val)
                
                arr[:] = sorted(unique_values)
    
    elif stype == "INT" and candidates:
        # INT: Clip to bounds after crossover
        if len(candidates) >= 2:
            min_val, max_val = candidates[0], candidates[1]
        elif len(candidates) == 1:
            min_val, max_val = 0, candidates[0]
        else:
            min_val, max_val = 0, 100
        
        arr1[:] = np.clip(arr1, min_val, max_val)
        arr2[:] = np.clip(arr2, min_val, max_val)

def _crossover_single_dbl_array(
    arr1: np.ndarray,
    arr2: np.ndarray,
    crossintensity: float,
    stype: str
) -> None:
    """Helper to crossover a single double array."""
    size = len(arr1)
    if size <= 1:
        return

    # Standard crossover for all continuous types including SPD, GRAPH_W
    # (Mixing parts of matrices is valid crossover strategy)
    n_points = max(1, int(size * crossintensity))
    n_points = min(n_points, size - 1)
    if n_points <= 0:
        return

    points = np.sort(
        np.random.choice(np.arange(1, size, dtype=int), size=n_points, replace=False)
    )

    for k, point in enumerate(points):
        if k % 2 != 0:
            continue
        start = points[k - 1] if k > 0 else 0
        end = point
        
        tmp = arr1[start:end].copy()
        arr1[start:end] = arr2[start:end]
        arr2[start:end] = tmp
    
    # Special repair for Manifold types if crossover breaks constraints?
    # SIMPLEX: Normalization might be broken if we swap arbitrary segments.
    # SPD: Positive definiteness might be broken.
    
    if stype == "SIMPLEX":
        # Repair normalization
        for arr in (arr1, arr2):
            # Ensure non-negative
            np.maximum(arr, 1e-9, out=arr)
            total = np.sum(arr)
            if total > 0:
                arr /= total
            else:
                arr[:] = 1.0 / size

    elif stype == "SPD":
        # Repair SPD property
        for arr in (arr1, arr2):
             # Reshape to NxN
            n_cols = len(arr)
            n = int(np.sqrt(n_cols))
            if n*n == n_cols:
                mat = arr.reshape(n, n)
                # Symmetrize
                mat = 0.5 * (mat + mat.T)
                # Fix PD
                try:
                    np.linalg.cholesky(mat)
                except np.linalg.LinAlgError:
                    vals, vecs = np.linalg.eigh(mat)
                    vals[vals < 1e-6] = 1e-6
                    mat = vecs @ np.diag(vals) @ vecs.T
                arr[:] = mat.flatten()


def crossover(
    parents: List[Solution],
    crossprob: float,
    crossintensity: float,
    settypes: List[str] = None,
    candidates: List[List[int]] = None
) -> List[Solution]:
    """
    Perform crossover on the population.
    Correctly handles mixed integer and double variable types.
    """
    offspring = []
    n_parents = len(parents)
    
    # Define continuous types
    dbl_types = ["DBL", "GRAPH_W", "SPD", "SIMPLEX"]
    
    for i in range(0, n_parents, 2):
        if i + 1 < n_parents:
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            if random.random() < crossprob:
                child1 = parent1.copy()
                child2 = parent2.copy()
                
                int_idx = 0
                dbl_idx = 0
                
                if settypes:
                    for j, stype in enumerate(settypes):
                        cand = candidates[j] if candidates else None
                        
                        if stype in dbl_types:
                            if dbl_idx < len(child1.dbl_values):
                                _crossover_single_dbl_array(
                                    child1.dbl_values[dbl_idx], 
                                    child2.dbl_values[dbl_idx],
                                    crossintensity, stype
                                )
                                dbl_idx += 1
                        else:
                            if int_idx < len(child1.int_values):
                                _crossover_single_int_array(
                                    child1.int_values[int_idx],
                                    child2.int_values[int_idx],
                                    crossintensity, stype, cand
                                )
                                int_idx += 1
                else:
                    # Fallback for backward compatibility or missing settypes
                    # Crossover all available arrays as generic types
                    # This relies on the old bulk functions if they still existed, 
                    # OR we just iterate all remaining arrays as generic DBL/UOS
                    pass 
                    # (For now assuming settypes is always provided in TrainSel context)

                child1.invalidate_hash()
                child2.invalidate_hash()
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parent1.copy())
                offspring.append(parent2.copy())
        else:
            offspring.append(parents[i].copy())
    
    return offspring


def mutation(
    population: List[Solution],
    candidates: List[List[int]],
    settypes: List[str],
    mutprob: float,
    mutintensity: float
) -> None:
    """
    Perform mutation on the population using vectorized operations where possible.
    Correctly handles mixed integer and double variable types.
    """
    if not population:
        return

    pop_size = len(population)
    
    # Initialize indices for int and dbl values
    # We assume all solutions have the same structure
    int_idx = 0
    dbl_idx = 0
    
    # Define continuous types
    dbl_types = ["DBL", "GRAPH_W", "SPD", "SIMPLEX"]
    
    for i, stype in enumerate(settypes):
        if stype in dbl_types:
            # ----- Continuous Variable Mutation -----
            if not population[0].dbl_values or dbl_idx >= len(population[0].dbl_values):
                continue
                
            # Get dimensions
            # We assume all solutions have the same size for this gene
            n_cols = len(population[0].dbl_values[dbl_idx])
            
            # Generate mutation mask and deltas
            mask = np.random.rand(pop_size, n_cols) < mutprob
            deltas = np.random.normal(0, mutintensity, size=(pop_size, n_cols))
            
            for idx, sol in enumerate(population):
                row_mask = mask[idx]
                if not np.any(row_mask):
                    continue
                
                row = sol.dbl_values[dbl_idx]
                
                if stype in ["DBL", "GRAPH_W"]:
                    # Standard continuous mutation
                    row[row_mask] += deltas[idx][row_mask]
                    np.clip(row, 0.0, 1.0, out=row)
                    
                elif stype == "SPD":
                    # Symmetric Positive Definite Mutation
                    # Add noise then project back to SPD
                    row[row_mask] += deltas[idx][row_mask]
                    # Reshape to NxN (assuming square)
                    n = int(np.sqrt(n_cols))
                    if n*n == n_cols:
                        mat = row.reshape(n, n)
                        # Symmetrize
                        mat = 0.5 * (mat + mat.T)
                        # Ensure PD (add identity if eigenvalues <= 0)
                        # Fast check/fix: value = U S U.T -> U max(S, eps) U.T
                        # Or simple diagonal boosting
                        try:
                            # Attempt Cholesky
                            np.linalg.cholesky(mat)
                        except np.linalg.LinAlgError:
                            # Fix
                            vals, vecs = np.linalg.eigh(mat)
                            vals[vals < 1e-6] = 1e-6
                            mat = vecs @ np.diag(vals) @ vecs.T
                        row[:] = mat.flatten()
                        
                elif stype == "SIMPLEX":
                    # Add noise then project to simplex (sum = 1, non-negative)
                    row[row_mask] += deltas[idx][row_mask]
                    # Clip negatives
                    np.maximum(row, 1e-9, out=row)
                    # Normalize
                    total = np.sum(row)
                    if total > 0:
                        row /= total
                    else:
                        # Reset to uniform if degenerate
                        row[:] = 1.0 / n_cols
                
                sol.invalidate_hash()
            
            dbl_idx += 1
            
        else:
            # ----- Integer Variable Mutation -----
            if not population[0].int_values or int_idx >= len(population[0].int_values):
                continue
                
            cand = candidates[i] if candidates and i < len(candidates) else []
            n_cols = len(population[0].int_values[int_idx])
            
            # Bulk generate masks
            mask = np.random.rand(pop_size, n_cols) < mutprob
            
            if stype == "OS":
                # Swap mutation
                for idx, sol in enumerate(population):
                    row = sol.int_values[int_idx]
                    pos_indices = np.where(mask[idx])[0]
                    if pos_indices.size == 0:
                        continue
                    
                    swap_pos = np.random.randint(0, n_cols, size=pos_indices.size)
                    for pos, sp in zip(pos_indices, swap_pos):
                        if pos != sp:
                            tmp = row[pos]
                            row[pos] = row[sp]
                            row[sp] = tmp
                    sol.invalidate_hash()
                    
            elif stype in ["BOOL", "GRAPH_U"]:
                # Bit flip (XOR)
                for idx, sol in enumerate(population):
                    row_mask = mask[idx]
                    if not np.any(row_mask):
                        continue
                    row = sol.int_values[int_idx]
                    row[row_mask] = 1 - row[row_mask]
                    sol.invalidate_hash()
                    
            elif stype == "PARTITION":
                # Partition Mutation: Merge/Split/Move
                # Since we treat partition as array of group IDs [0..K-1],
                # "Move" is efficiently implemented as changing value to another random group ID.
                # "Merge" would be complex (all x -> y). 
                # "Split" would be selecting subset of group x -> new group z.
                # For now, we implement "Move" (reassign element to different group).
                
                # We need to know max group ID? 
                # Assuming cand contains group IDs OR we infer from current max?
                # Let's assume cand contains allowed group IDs.
                
                valid_groups = np.array(cand) if cand else np.arange(int(np.sqrt(n_cols))) # Fallback
                if len(valid_groups) == 0: valid_groups = np.array([0])
                
                for idx, sol in enumerate(population):
                    row_mask = mask[idx]
                    if not np.any(row_mask):
                        continue
                        
                    row = sol.int_values[int_idx]
                    # Assign new random group from candidates
                    new_groups = np.random.choice(valid_groups, size=np.sum(row_mask))
                    row[row_mask] = new_groups
                    sol.invalidate_hash()
                    
            elif stype == "INT":
                # INT: add small integer deltas with boundary clipping  
                for idx, sol in enumerate(population):
                    row_mask = mask[idx]
                    if not np.any(row_mask):
                        continue
                        
                    row = sol.int_values[int_idx]
                    # Get bounds from candidates
                    if len(cand) >= 2:
                        min_val, max_val = cand[0], cand[1]
                    elif len(cand) == 1:
                        min_val, max_val = 0, cand[0]
                    else:
                        min_val, max_val = 0, 100
                    
                    # Add integer deltas scaled by range
                    range_size = max_val - min_val
                    sigma = max(1, int(mutintensity * range_size))
                    deltas = np.random.randint(-sigma, sigma + 1, size=np.sum(row_mask))
                    row[row_mask] += deltas
                    # Clip to bounds
                    row[:] = np.clip(row, min_val, max_val)
                    sol.invalidate_hash()
                     
            else:
                # Standard Set Mutation (UOS, UOMS, OMS)
                for idx, sol in enumerate(population):
                    row_mask = mask[idx]
                    if not np.any(row_mask):
                        continue
                        
                    row = sol.int_values[int_idx]
                    if stype in ["UOS", "OS"]:
                        current_set = set(row)
                    else:
                        current_set = set() # Not needed
                        
                    positions = np.where(row_mask)[0]
                    for pos in positions:
                        old_val = row[pos]
                        new_val = _get_valid_replacement(old_val, current_set, cand, stype)
                        row[pos] = new_val
                        if stype in ["UOS", "OS"]:
                            current_set.discard(old_val)
                            current_set.add(new_val)
                            
                    if stype in ["UOS", "UOMS"]:
                        row.sort()
                        
                    sol.invalidate_hash()
            
            int_idx += 1

"""
Solution class and helpers for EvoSolve.
"""

import numpy as np
from typing import List, Dict, Callable, Union, Optional, Any, Tuple

class Solution:
    """
    Class to represent a solution in the genetic algorithm.

    Attributes
    ----------
    int_values : List[np.ndarray]
        Integer-valued decision variables (for UOS, OS, UOMS, OMS, BOOL set types)
    dbl_values : List[np.ndarray]
        Continuous decision variables (for DBL set type)
    fitness : float
        Scalar fitness value (sum of objectives for multi-objective)
    multi_fitness : List[float]
        Individual objective values for multi-objective optimization
    """
    def __init__(
        self,
        int_values: List[Any] = None,
        dbl_values: List[Any] = None,
        fitness: float = float('-inf'),
        multi_fitness: List[float] = None
    ):
        # Normalize inputs to lists of numpy arrays
        if int_values is None:
            self.int_values = []
        else:
            self.int_values = [
                np.asarray(x, dtype=int) for x in int_values
            ]

        if dbl_values is None:
            self.dbl_values = []
        else:
            self.dbl_values = [
                np.asarray(x, dtype=float) for x in dbl_values
            ]

        self.fitness = float(fitness) if fitness is not None else float('-inf')
        self.multi_fitness = list(multi_fitness) if multi_fitness is not None else []

        # Cache for hash value to avoid repeated computation (performance optimization)
        self._hash_cache = None
        self._hash_valid = False

    def copy(self):
        """Create a deep copy of the solution."""
        # Copy integer values (numpy arrays need explicit copy)
        int_copy = [x.copy() for x in self.int_values]
        # Copy double values
        dbl_copy = [x.copy() for x in self.dbl_values]
        # Always copy multi_fitness as a list (even if empty)
        multi_fit_copy = self.multi_fitness.copy() if self.multi_fitness else []
        new_sol = Solution(int_copy, dbl_copy, self.fitness, multi_fit_copy)

        # Copy cached hash if it's valid (performance optimization)
        if self._hash_valid:
            new_sol._hash_cache = self._hash_cache
            new_sol._hash_valid = True

        return new_sol

    def get_hash(self):
        """
        Get a hash of the solution for caching.

        Uses cached hash value for performance. Hash is invalidated when solution
        is modified (mutation/crossover create new solutions, so hash remains valid).
        """
        if self._hash_valid and self._hash_cache is not None:
            return self._hash_cache

        # Create hash from integer and double values
        # Convert arrays to tuples for hashing
        # Ensure it works even if stored as list (defensive coding)
        int_tuple = tuple(tuple(np.asarray(iv).tolist()) for iv in self.int_values)
        
        # Round doubles to avoid precision issues
        # Using 8 decimals should be safe for caching within a run
        # Convert to rounded tuple
        dbl_tuple = tuple(tuple(np.round(np.asarray(dv), 8).tolist()) for dv in self.dbl_values)

        self._hash_cache = hash((int_tuple, dbl_tuple))
        self._hash_valid = True
        return self._hash_cache

    def invalidate_hash(self):
        """Invalidate cached hash (call after in-place modification)."""
        self._hash_valid = False
        self._hash_cache = None

    def __lt__(self, other):
        """Comparison for sorting (by fitness)."""
        return self.fitness < other.fitness


def flatten_dbl_values(dbl_values: List[Any]) -> np.ndarray:
    """Flatten double values into a single numpy array."""
    if not dbl_values:
        return np.array([])
    return np.concatenate([np.asarray(x) for x in dbl_values])


def unflatten_dbl_values(flat_values: np.ndarray, template: List[Any]) -> List[np.ndarray]:
    """Unflatten numpy array back into list of numpy arrays structure."""
    result = []
    idx = 0
    for sublist in template:
        # sublist is expected to be array-like, so len() works if it's correct
        # or size if it's numpy array. Use len(sublist) safely.
        size = len(sublist)
        result.append(flat_values[idx:idx+size].copy()) # .copy() to own memory?
        idx += size
    return result

"""
Genetic algorithm implementation for TrainSelPy.
This module is deprecated and re-exports functions from:
- evosolve.algorithms
- evosolve.operators
- evosolve.selection
"""

from evosolve.algorithms import (
    genetic_algorithm,
    island_model_ga,
    initialize_population,
    evaluate_fitness,
    simulated_annealing
)

from evosolve.operators import (
    crossover,
    mutation
)

from evosolve.selection import (
    selection,
    fast_non_dominated_sort,
    calculate_crowding_distance
)

# Re-export for backward compatibility
__all__ = [
    "genetic_algorithm",
    "island_model_ga",
    "initialize_population",
    "evaluate_fitness",
    "simulated_annealing",
    "crossover",
    "mutation",
    "selection",
    "fast_non_dominated_sort",
    "calculate_crowding_distance"
]
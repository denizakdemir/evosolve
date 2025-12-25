"""
EvoSolve: A General Purpose Evolutionary Optimization Framework
"""

from evosolve.core import (
    make_data,
    evolve,
    evolve_control,
    set_control_default,
    time_estimation
)

from evosolve.optimization_criteria import (
    dopt,
    maximin_opt,
    pev_opt,
    cdmean_opt,
    cdmean_opt_target,
    fun_opt_prop,
    aopt,
    eopt,
    coverage_opt
)

from evosolve.utils import (
    r_data_to_python,
    create_distance_matrix,
    calculate_relationship_matrix,
    create_mixed_model_data,
    plot_optimization_progress,
    plot_pareto_front
)

# Import data module
from evosolve.data import wheat_data


# Import relaxations module
from evosolve.relaxations import (
    DOptimality,
    AOptimality,
    CDMeanOptimality,
    PEVOptimality,
    ConvexRelaxationSolver,
    discretize
)

__version__ = "0.1.1"
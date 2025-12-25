"""
Core module implementing the main functionality of EvoSolve.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Union, Optional, Any, TypedDict, Tuple
from dataclasses import dataclass
import time
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.linalg import det, solve
from scipy.spatial.distance import pdist, squareform
import warnings
import random

from evosolve.optimization_criteria import cdmean_opt
from evosolve.algorithms import (
    genetic_algorithm,
    island_model_ga
)


class EvoData(TypedDict, total=False):
    G: Union[np.ndarray, pd.DataFrame]
    R: Union[np.ndarray, pd.DataFrame]
    lambda_val: float
    labels: pd.DataFrame
    Nind: int
    class_name: str
    X: Optional[np.ndarray]
    L: Optional[Union[np.ndarray, pd.DataFrame]]
    G_L: Optional[np.ndarray]
    L_G_L_diag: Optional[np.ndarray]



class ControlParams(TypedDict, total=False):
    size: str
    niterations: int
    minitbefstop: int
    nEliteSaved: int
    nelite: int
    npop: int
    mutprob: float
    mutintensity: float
    crossprob: float
    crossintensity: float
    niterSANN: int
    tempini: float
    tempfin: float
    dynamicNelite: bool
    progress: bool
    parallelizable: bool
    mc_cores: int
    nislands: int
    niterIslands: int
    minitbefstopIslands: int
    nEliteSavedIslands: int
    neliteIslands: int
    npopIslands: int
    niterSANNislands: int
    use_surrogate: bool
    surrogate_start_gen: int
    surrogate_update_freq: int
    surrogate_prescreen_factor: int
    use_surrogate_objective: bool
    surrogate_generation_prob: float
    use_nsga3: bool
    use_cma_es: bool
    cma_es_sigma: float
    # Optional hooks and evaluation controls
    repair_func: Callable[..., None]
    callback: Callable[[Dict[str, Any]], None]
    vectorized_stat: bool
    # Neural Network Parameters
    use_vae: bool
    use_gan: bool
    nn_epochs: int
    nn_update_freq: int
    nn_start_gen: int
    vae_lr: float
    gan_lr: float
    gan_lambda_gp: float
    gan_n_critic: int
    # Distributional Optimization Parameters
    dist_objective: str
    dist_n_samples: int
    dist_K_particles: int
    dist_compression: str
    dist_lambda_var: float
    dist_alpha: float
    dist_tau: float
    dist_weight_mutation_prob: float
    dist_support_mutation_prob: float


def make_data(
    M: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    Vk: Optional[np.ndarray] = None,
    Ve: Optional[np.ndarray] = None,
    lambda_val: Optional[float] = None,
    X: Optional[np.ndarray] = None,
    L: Optional[Union[np.ndarray, pd.DataFrame]] = None
) -> EvoData:
    """
    Create a data structure for EvoSolve optimization.
    
    Parameters
    ----------
    M : ndarray, optional
        Features matrix for samples.
    K : ndarray, optional
        Relationship matrix for samples.
    R : ndarray, optional
        Relationship matrix for errors of samples.
    Vk : ndarray, optional
        Relationship matrix blocks.
    Ve : ndarray, optional
        Relationship matrix errors of blocks.
    lambda_val : float, optional
        Ratio of Ve to Vk.
    X : ndarray, optional
        Design matrix.
    L : ndarray or DataFrame, optional
        Linear contrast matrix. If provided, CDMEAN optimization
        will target the reliability of Lu (linear combinations)
        instead of u.
        
    Returns
    -------
    Dict[str, Any]
        Data structure for EvoSolve optimization.
    """
    if M is None and K is None:
        raise ValueError("At least one of M (features) or K (similarity matrix) must be provided.")
    
    if M is None and K is not None:
        # Use SVD decomposition to obtain a feature matrix from K.
        U, _, _ = np.linalg.svd(K)
        M = U
        if hasattr(K, 'index'):
            M = pd.DataFrame(M, index=K.index)
    
    # Determine names from M
    if hasattr(M, 'index'):
        names_in_M = M.index
    else:
        names_in_M = np.arange(M.shape[0])
        M = pd.DataFrame(M, index=names_in_M)
    
    # Compute K from M if not provided
    if K is None:
        K = np.dot(M, M.T) / M.shape[1]
        K = make_positive_definite(K / np.mean(np.diag(K)))
    
    if not hasattr(K, 'index'):
        K = pd.DataFrame(K, index=names_in_M, columns=names_in_M)
    
    if R is None:
        R = np.eye(K.shape[0])
        R = pd.DataFrame(R, index=K.index, columns=K.columns)
    
    if Vk is not None and Ve is not None:
        if lambda_val is None:
            lambda_val = 1
        
        if not hasattr(Vk, 'index'):
            Vk = pd.DataFrame(Vk, index=np.arange(Vk.shape[0]))
        
        if not hasattr(Ve, 'index'):
            Ve = pd.DataFrame(Ve, index=Vk.index)
        
        # Compute the Kronecker product for block matrices.
        big_K = np.kron(Vk.values, K.values)
        big_R = np.kron(lambda_val * Ve.values, R.values)
        
        # Create combined labels.
        k_idx = K.index
        vk_idx = Vk.index
        combined_names = [f"{a}_{b}" for a in vk_idx for b in k_idx]
        
        labels = pd.DataFrame({
            'intlabel': np.arange(len(combined_names)),
            'names': combined_names
        })
        
        big_K = pd.DataFrame(big_K, index=combined_names, columns=combined_names)
        big_R = pd.DataFrame(big_R, index=combined_names, columns=combined_names)
    else:
        if lambda_val is None:
            lambda_val = 1
        
        big_K = K
        big_R = lambda_val * R
        
        labels = pd.DataFrame({
            'intlabel': np.arange(len(big_K.index)),
            'names': big_K.index
        })
    
    result: EvoData = {
        'G': big_K,
        'R': big_R,
        'lambda_val': lambda_val,
        'lambda': lambda_val,
        'labels': labels,
        'Nind': K.shape[0],
        'class_name': "EvoSolve_Data"
    }
    
    if X is not None:
        result['X'] = X
        
    if L is not None:
        # Precompute matrices for linear contrast optimization
        from evosolve.optimization_criteria import _ensure_numpy
        L_arr = _ensure_numpy(L)
        G_arr = _ensure_numpy(big_K)
        
        # Check dimensions
        if L_arr.shape[1] != G_arr.shape[0]:
            raise ValueError(f"L matrix columns ({L_arr.shape[1]}) must match G matrix rows ({G_arr.shape[0]})")
            
        # Compute G_L = L @ G (Used in numerator)
        G_L = L_arr @ G_arr
        
        # Compute diag(L @ G @ L.T) (Used in denominator)
        # Efficient computation: sum((L @ G) * L, axis=1)
        L_G_L_diag = np.sum(G_L * L_arr, axis=1)
        
        result['L'] = L
        result['G_L'] = G_L
        result['L_G_L_diag'] = L_G_L_diag
    
    return result


def evolve_control(
    size: str = "free",
    niterations: int = 2000,
    minitbefstop: int = 500,
    nEliteSaved: int = 10,
    nelite: int = 200,
    npop: int = 1000,
    mutprob: float = 0.01,
    mutintensity: float = 0.1,
    crossprob: float = 0.5,
    crossintensity: float = 0.75,
    niterSANN: int = 200,
    tempini: float = 100.0,
    tempfin: float = 0.1,
    dynamicNelite: bool = True,
    progress: bool = True,
    parallelizable: bool = False,
    mc_cores: int = 1,
    nislands: int = 1,
    niterIslands: int = 200,
    minitbefstopIslands: int = 20,
    nEliteSavedIslands: int = 3,
    neliteIslands: int = 50,
    npopIslands: int = 200,
    niterSANNislands: int = 30,
    trace: bool = False,
    use_surrogate: bool = False,
    surrogate_start_gen: int = 10,
    surrogate_update_freq: int = 5,
    surrogate_prescreen_factor: int = 5,
    use_surrogate_objective: bool = False,
    surrogate_generation_prob: float = 0.0,
    use_nsga3: bool = False,
    use_cma_es: bool = True,
    cma_es_sigma: float = 0.2,
    repair_func: Optional[Callable[..., None]] = None,
    vectorized_stat: bool = False,
    # Neural Network Parameters
    use_vae: bool = False,
    use_gan: bool = False,
    nn_epochs: int = 10,
    nn_update_freq: int = 5,
    nn_start_gen: int = 10,
    vae_lr: float = 1e-3,
    gan_lr: float = 1e-4,
    gan_lambda_gp: float = 10.0,
    gan_n_critic: int = 5,
    # Distributional Optimization Parameters
    dist_objective: str = "mean",
    dist_n_samples: int = 20,
    dist_K_particles: int = 10,
    dist_compression: str = "top_k",
    dist_lambda_var: float = 1.0,
    dist_alpha: float = 0.1,
    dist_tau: float = 0.1,
    dist_maximize: bool = True,
    dist_weight_mutation_prob: float = 0.5,
    dist_support_mutation_prob: float = 0.5,
    dist_birth_death_prob: float = 0.0,
    dist_birth_rate: float = 0.1,
    dist_death_rate: float = 0.1,
    dist_hv_ref: Optional[Tuple[float, float]] = None,
    dist_eval_mode: str = "sample",
    dist_use_nsga_means: bool = False,
) -> ControlParams:
    """
    Create a control object for the EvoSolve function.
    
    Parameters
    ----------
    size : str
        Size of the problem (e.g., "free").
    niterations : int
        Maximum number of iterations.
    minitbefstop : int
        Minimum number of iterations before stopping.
    nEliteSaved : int
        Number of elite solutions to save.
    nelite : int
        Number of elite solutions to carry to the next generation.
    npop : int
        Population size.
    mutprob : float
        Mutation probability.
    mutintensity : float
        Mutation intensity.
    crossprob : float
        Crossover probability.
    crossintensity : float
        Crossover intensity.
    niterSANN : int
        Number of simulated annealing iterations.
    tempini : float
        Initial temperature for simulated annealing.
    tempfin : float
        Final temperature for simulated annealing.
    dynamicNelite : bool
        Whether to adjust the number of elites dynamically.
    progress : bool
        Whether to display progress.
    parallelizable : bool
        Whether to use parallelization.
    mc_cores : int
        Number of cores to use for parallelization.
    nislands : int
        Number of islands for the island model.
    niterIslands : int
        Maximum number of iterations for the island model.
    minitbefstopIslands : int
        Minimum number of iterations before stopping for the island model.
    nEliteSavedIslands : int
        Number of elite solutions to save for the island model.
    neliteIslands : int
        Number of elite solutions to carry to the next generation for the island model.
    npopIslands : int
        Population size for the island model.
    niterSANNislands : int
        Number of simulated annealing iterations for the island model.
    trace : bool
        Whether to save the trace of the optimization.
    use_surrogate : bool
        Whether to use surrogate-assisted optimization.
    surrogate_start_gen : int
        Generation to start using surrogate.
    surrogate_update_freq : int
        Frequency of surrogate model updates.
    surrogate_prescreen_factor : int
        Factor for generating extra offspring for pre-screening.
    use_surrogate_objective : bool
        Whether to use the surrogate model as the objective function.
    surrogate_generation_prob : float
        Probability of generating offspring using surrogate optimization.
    use_nsga3 : bool
        Whether to use NSGA-III selection for multi-objective optimization.
    use_cma_es : bool
        Whether to use CMA-ES for continuous optimization (DBL variables).
    cma_es_sigma : float
        Initial step size (sigma) for CMA-ES optimizer. Default 0.2.
        Larger values increase exploration, smaller values increase exploitation.
    repair_func : Callable, optional
        Optional repair operator applied to offspring before fitness evaluation.
    callback : Callable[[Dict[str, Any]], None], optional
        Optional callback function invoked at the end of each generation.
        Receives a dictionary with keys: 'generation', 'population',
        'best_solution', 'fitness_history', 'pareto_front' (if multi-objective),
        'control'. Useful for checkpointing, visualization, and custom logging.
    vectorized_stat : bool
        When True, the objective function is assumed to support batched
        (vectorized) inputs and will be called once per generation with all
        solutions instead of once per solution.
    dist_objective : str
        Distributional objective type: 'mean', 'mean_variance', 'cvar', 'entropy'.
    dist_n_samples : int
        Number of Monte Carlo samples for distributional objectives.
    dist_K_particles : int
        Number of particles per distribution.
    dist_compression : str
        Compression strategy: 'top_k', 'resampling', 'kmeans'.
    dist_lambda_var : float
        Variance weight for mean-variance objective.
    dist_alpha : float
        Quantile level for CVaR objective.
    dist_tau : float
        Entropy regularization weight.
    dist_weight_mutation_prob : float
        Probability of mutating distribution weights.
    dist_support_mutation_prob : float
        Probability of mutating distribution support (particles).
    dist_birth_death_prob : float
        Probability of applying birth/death mutation to refresh particles.
    dist_birth_rate : float
        Fraction of particles to add during birth mutation.
    dist_death_rate : float
        Fraction of particles to remove during death mutation.
    dist_hv_ref : Tuple[float, float], optional
        Reference point for hypervolume-based objectives (maximization).
    dist_eval_mode : str
        How distributional objectives are aggregated: "sample" (Monte Carlo) or
        "weighted" (deterministic particle-weighted evaluation).
    dist_use_nsga_means : bool
        If True, expose distribution mean objectives for NSGA selection even when
        running with scalarized distributional objectives.
        
    Returns
    -------
    Dict[str, Any]
        Control object for the EvoSolve function.
    """
    control: ControlParams = {
        "size": size,
        "niterations": niterations,
        "minitbefstop": minitbefstop,
        "nEliteSaved": nEliteSaved,
        "nelite": nelite,
        "npop": npop,
        "mutprob": mutprob,
        "mutintensity": mutintensity,
        "crossprob": crossprob,
        "crossintensity": crossintensity,
        "niterSANN": niterSANN,
        "tempini": tempini,
        "tempfin": tempfin,
        "dynamicNelite": dynamicNelite,
        "progress": progress,
        "parallelizable": parallelizable,
        "mc_cores": mc_cores,
        "nislands": nislands,
        "niterIslands": niterIslands,
        "minitbefstopIslands": minitbefstopIslands,
        "nEliteSavedIslands": nEliteSavedIslands,
        "neliteIslands": neliteIslands,
        "npopIslands": npopIslands,
        "niterSANNislands": niterSANNislands,
        "trace": trace,
        "use_surrogate": use_surrogate,
        "surrogate_start_gen": surrogate_start_gen,
        "surrogate_update_freq": surrogate_update_freq,
        "surrogate_prescreen_factor": surrogate_prescreen_factor,
        "use_surrogate_objective": use_surrogate_objective,
        "surrogate_generation_prob": surrogate_generation_prob,
        "use_nsga3": use_nsga3,
        "use_cma_es": use_cma_es,
        "cma_es_sigma": cma_es_sigma,
        "repair_func": repair_func,
        "vectorized_stat": vectorized_stat,
        "use_vae": use_vae,
        "use_gan": use_gan,
        "nn_epochs": nn_epochs,
        "nn_update_freq": nn_update_freq,
        "nn_start_gen": nn_start_gen,
        "vae_lr": vae_lr,
        "gan_lr": gan_lr,
        "gan_lambda_gp": gan_lambda_gp,
        "gan_n_critic": gan_n_critic,
        "dist_objective": dist_objective,
        "dist_n_samples": dist_n_samples,
        "dist_K_particles": dist_K_particles,
        "dist_compression": dist_compression,
        "dist_lambda_var": dist_lambda_var,
        "dist_alpha": dist_alpha,
        "dist_tau": dist_tau,
        "dist_maximize": dist_maximize,
        "dist_weight_mutation_prob": dist_weight_mutation_prob,
        "dist_support_mutation_prob": dist_support_mutation_prob,
        "dist_birth_death_prob": dist_birth_death_prob,
        "dist_birth_rate": dist_birth_rate,
        "dist_death_rate": dist_death_rate,
        "dist_hv_ref": dist_hv_ref,
        "dist_eval_mode": dist_eval_mode,
        "dist_use_nsga_means": dist_use_nsga_means
    }

    # Configuration Validation for Distributional Optimization
    _validate_distributional_config(control)

    return control


def _validate_distributional_config(control: ControlParams) -> None:
    """
    Validate distributional optimization configuration parameters.

    Issues warnings for likely misconfigurations and raises errors for
    incompatible combinations.

    Parameters
    ----------
    control : ControlParams
        Control parameters dictionary to validate
    """
    import warnings

    dist_obj = control.get("dist_objective", "mean")
    dist_alpha = control.get("dist_alpha", 0.1)
    dist_tau = control.get("dist_tau", 0.1)
    dist_lambda_var = control.get("dist_lambda_var", 1.0)
    dist_compression = control.get("dist_compression", "top_k")

    # Validate dist_objective value
    valid_objectives = ["mean", "mean_variance", "cvar", "entropy", "hypervolume"]
    if dist_obj not in valid_objectives:
        raise ValueError(
            f"Invalid dist_objective '{dist_obj}'. "
            f"Must be one of: {', '.join(valid_objectives)}"
        )

    # Warn if CVaR-specific parameter is set but objective is not CVaR
    if dist_alpha != 0.1 and dist_obj != "cvar":
        warnings.warn(
            f"dist_alpha={dist_alpha} is set but dist_objective='{dist_obj}' "
            "(not 'cvar'). The alpha parameter only affects CVaR objectives.",
            UserWarning
        )

    # Warn if entropy-specific parameter is set but objective is not entropy
    if dist_tau != 0.1 and dist_obj != "entropy":
        warnings.warn(
            f"dist_tau={dist_tau} is set but dist_objective='{dist_obj}' "
            "(not 'entropy'). The tau parameter only affects entropy-regularized objectives.",
            UserWarning
        )

    # Warn if mean-variance-specific parameter is set but objective is not mean_variance
    if dist_lambda_var != 1.0 and dist_obj != "mean_variance":
        warnings.warn(
            f"dist_lambda_var={dist_lambda_var} is set but dist_objective='{dist_obj}' "
            "(not 'mean_variance'). The lambda_var parameter only affects mean-variance objectives.",
            UserWarning
        )

    # Validate dist_compression value
    valid_compressions = ["top_k", "resampling", "kmeans"]
    if dist_compression not in valid_compressions:
        warnings.warn(
            f"Unknown dist_compression '{dist_compression}'. "
            f"Valid options are: {', '.join(valid_compressions)}. "
            "Defaulting to 'top_k'.",
            UserWarning
        )

    # Validate alpha range for CVaR
    if dist_obj == "cvar":
        if not (0 < dist_alpha <= 1):
            raise ValueError(
                f"dist_alpha must be in (0, 1] for CVaR objective, got {dist_alpha}"
            )

    # Validate dist_eval_mode
    dist_eval_mode = control.get("dist_eval_mode", "sample")
    valid_eval_modes = ["sample", "weighted"]
    if dist_eval_mode not in valid_eval_modes:
        warnings.warn(
            f"Unknown dist_eval_mode '{dist_eval_mode}'. "
            f"Valid options are: {', '.join(valid_eval_modes)}. "
            "Defaulting to 'sample'.",
            UserWarning
        )

    # Validate K_particles
    K_particles = control.get("dist_K_particles", 10)
    if K_particles < 2:
        raise ValueError(
            f"dist_K_particles must be >= 2 for meaningful distributions, got {K_particles}"
        )


# =============================================================================
# Distributional Optimization Presets
# =============================================================================

DISTRIBUTIONAL_PRESETS = {
    "robust": {
        "dist_objective": "cvar",
        "dist_alpha": 0.2,
        "dist_K_particles": 15,
        "dist_maximize": False,
        "dist_compression": "kmeans",
        "dist_use_nsga_means": True,
        "use_cma_es": False,  # CMA-ES incompatible with distributional
        "__description__": "Conservative optimization focused on worst-case scenarios. "
                          "Uses CVaR with α=0.2 (bottom 20% tail) to minimize risk."
    },
    "risk_averse": {
        "dist_objective": "mean_variance",
        "dist_lambda_var": 2.0,
        "dist_K_particles": 12,
        "dist_maximize": True,
        "dist_compression": "top_k",
        "dist_use_nsga_means": True,
        "use_cma_es": False,  # CMA-ES incompatible with distributional
        "__description__": "Balances performance and stability. "
                          "Penalizes high variance solutions (λ=2.0) to prefer consistent outcomes."
    },
    "exploratory": {
        "dist_objective": "entropy",
        "dist_tau": 0.5,
        "dist_K_particles": 20,
        "dist_maximize": True,
        "dist_compression": "resampling",
        "dist_weight_mutation_prob": 0.7,
        "dist_support_mutation_prob": 0.7,
        "use_cma_es": False,  # CMA-ES incompatible with distributional
        "__description__": "Encourages diversity and exploration. "
                          "Uses entropy regularization (τ=0.5) to maintain diverse particle distributions."
    },
    "performance": {
        "dist_objective": "mean",
        "dist_K_particles": 10,
        "dist_maximize": True,
        "dist_compression": "top_k",
        "dist_use_nsga_means": False,
        "use_cma_es": False,  # CMA-ES incompatible with distributional
        "__description__": "Pure performance optimization. "
                          "Optimizes expected value without risk considerations."
    }
}


def get_distributional_preset(preset_name: str, **overrides) -> dict:
    """
    Get a predefined distributional optimization configuration.

    Presets provide sensible defaults for common use cases:

    - **robust**: Conservative, worst-case optimization (CVaR α=0.2)
    - **risk_averse**: Balanced performance + stability (Mean-Variance λ=2.0)
    - **exploratory**: Maximum diversity and exploration (Entropy τ=0.5)
    - **performance**: Pure mean performance optimization

    Parameters
    ----------
    preset_name : str
        Name of the preset: "robust", "risk_averse", "exploratory", or "performance"
    **overrides
        Additional parameters to override preset defaults

    Returns
    -------
    dict
        Configuration parameters suitable for evolve_control(**config)

    Examples
    --------
    >>> # Use robust preset with custom K
    >>> config = get_distributional_preset("robust", dist_K_particles=20)
    >>> control = train_sel_control(**config)

    >>> # Use risk-averse preset
    >>> config = get_distributional_preset("risk_averse")
    >>> result = evolve(stat=fitness_fn, candidates=...,
    ...                    settypes=["DIST:BOOL"],
    ...                    control=evolve_control(**config))
    """
    if preset_name not in DISTRIBUTIONAL_PRESETS:
        available = ", ".join(DISTRIBUTIONAL_PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )

    # Get preset config (excluding description)
    config = {k: v for k, v in DISTRIBUTIONAL_PRESETS[preset_name].items()
              if not k.startswith("__")}

    # Apply overrides
    config.update(overrides)

    return config


def set_control_default(size: str = "free", **kwargs) -> ControlParams:
    """
    Set default parameters for the EvoSolve control object.
    
    Parameters
    ----------
    size : str
        Size of the problem (default is "free").
    **kwargs
        Additional control parameters.
        
    Returns
    -------
    Dict[str, Any]
        Control object with default parameters.
    """
    # Use the provided size parameter
    defaults = {
        "niterations": 2000,
        "minitbefstop": 500,
        "nEliteSaved": 10,
        "nelite": 200,
        "npop": 1000,
        "mutprob": 0.01,
        "mutintensity": 0.1,
        "crossprob": 0.5,
        "crossintensity": 0.75,
        "niterSANN": 200,
        "tempini": 100.0,
        "tempfin": 0.1,
        "dynamicNelite": True,
        "progress": True,
        "parallelizable": False,
        "mc_cores": 1,
        "nislands": 1,
    }
    
    # Update defaults with provided kwargs, allowing user to override defaults
    defaults.update(kwargs)

    return evolve_control(
        size=size,
        **defaults
    )


@dataclass
class EvoResult:
    """
    Class to hold the results of the EvoSolve optimization.
    """
    selected_indices: List[List[int]]  # Selected indices for each set.
    selected_values: List[Any]         # Selected values for each set.
    fitness: float                     # Fitness value.
    fitness_history: List[float]       # History of fitness values.
    execution_time: float              # Execution time in seconds.
    pareto_front: Optional[List[List[float]]] = None  # Pareto front for multi-objective optimization.
    pareto_solutions: Optional[List[Dict[str, Any]]] = None  # Pareto solutions for multi-objective optimization.
    # Distributional optimization fields
    distribution: Optional[Any] = None  # Best ParticleDistribution for distributional optimization
    particle_solutions: Optional[List[Any]] = None  # Particle solutions
    particle_weights: Optional[np.ndarray] = None  # Particle weights


def evolve(
    data: Optional[EvoData] = None,
    candidates: Optional[List[List[int]]] = None,
    setsizes: Optional[List[int]] = None,
    ntotal: Optional[int] = None,
    settypes: Optional[List[str]] = None,
    stat: Optional[Callable] = None,
    n_stat: int = 1,
    target: Optional[List[int]] = None,
    control: Optional[ControlParams] = None,
    init_sol: Optional[Dict[str, Any]] = None,
    packages: List[str] = [],
    n_jobs: int = 1,
    verbose: bool = True
) -> EvoResult:
    """
    Optimize the selection of training populations.
    
    Parameters
    ----------
    data : EvoData, optional
        Data structure for EvoSolve optimization.
    candidates : List[List[int]], optional
        List of candidate index sets.
    setsizes : List[int], optional
        List of set sizes to select.
    ntotal : int, optional
        Total number of elements to select.
    settypes : List[str], optional
        List of set types. Options include "UOS", "OS", "UOMS", "OMS", "BOOL", "DBL".
    stat : Callable, optional
        Fitness function. If None, the CDMEAN criterion is used.
    n_stat : int, optional
        Number of objectives for multi-objective optimization.
    target : List[int], optional
        List of target indices.
    control : Dict[str, Any], optional
        Control object for the EvoSolve function.
    init_sol : Dict[str, Any], optional
        Initial solution.
    packages : List[str], optional
        List of packages to import in parallel.
    n_jobs : int, optional
        Number of jobs for parallelization.
    verbose : bool, optional
        Whether to display progress messages.
    verbose : bool, optional
        Whether to display progress messages.
        
    Returns
    -------
    EvoSolveResult
        Results of the optimization.
    """
    if verbose:
        print("Starting EvoSolve optimization")
    
    start_time = time.time()
    
    # Set defaults
    if ntotal is None:
        ntotal = 0
    
    if target is None:
        target = []
    
    # Use CDMEAN if no alternative fitness function is provided.
    if stat is None:
        stat = lambda sol, d=data: cdmean_opt(sol, d)
    
    if stat is None and data is None:
        raise ValueError("No data provided for CDMEAN optimization")
    
    # Set control parameters if not provided.
    if control is None:
        control = set_control_default()
    
    nislands = control.get("nislands", 1)
    parallelizable = control.get("parallelizable", False)
    n_cores = control.get("mc_cores", 1)
    
    # Validate candidates and setsizes.
    if candidates is None or setsizes is None:
        raise ValueError("Candidates and setsizes must be provided")
    
    if len(candidates) != len(setsizes):
        raise ValueError("Candidates and setsizes must have the same length")
    
    # Validate settypes.
    if settypes is None:
        settypes = ["UOS"] * len(candidates)
    
    if len(settypes) != len(candidates):
        raise ValueError("Settypes and candidates must have the same length")
    
    # Count the number of double variables.
    # GRAPH_W, SPD, SIMPLEX are treated as continuous (double) variables
    dbl_types = ["DBL", "GRAPH_W", "SPD", "SIMPLEX"]
    n_dbl = sum(1 for st in settypes if st in dbl_types)
    
    # Parallel processing setup.
    if parallelizable and n_stat > 1 and nislands == 1:
        raise ValueError("Parallelization for multi-objective optimization is only supported when nislands > 1")
    
    if ntotal is not None and ntotal > 0 and parallelizable:
        raise ValueError("ntotal is not supported when working in parallel")
    
    # Run the optimization.
    if nislands == 1:
        # Single island optimization.
        if parallelizable:
            if n_stat > 1:
                raise ValueError("Parallelization for multi-objective optimization is only supported when nislands > 1")
            
            # Define parallel fitness evaluation function.
            if n_dbl == 0 or n_dbl == len(settypes):
                def parallel_stat(solutions):
                    return Parallel(n_jobs=n_jobs)(
                        delayed(stat)(solution, data) for solution in solutions
                    )
            else:
                def parallel_stat(int_solutions, dbl_solutions):
                    return Parallel(n_jobs=n_jobs)(
                        delayed(stat)(int_sol, dbl_sol, data) 
                        for int_sol, dbl_sol in zip(int_solutions, dbl_solutions)
                    )
            
            result = genetic_algorithm(
                data=data,
                candidates=candidates,
                setsizes=setsizes,
                settypes=settypes,
                stat_func=parallel_stat,
                target=target,
                control=control,
                init_sol=init_sol,
                n_stat=n_stat,
                is_parallel=True
            )
        else:
            result = genetic_algorithm(
                data=data,
                candidates=candidates,
                setsizes=setsizes,
                settypes=settypes,
                stat_func=stat,
                target=target,
                control=control,
                init_sol=init_sol,
                n_stat=n_stat,
                is_parallel=False
            )
    else:
        # Island model optimization.
        inner_control = control.copy()
        inner_control["niterations"] = control.get("niterIslands", 200)
        inner_control["minitbefstop"] = control.get("minitbefstopIslands", 20)
        inner_control["niterSANN"] = control.get("niterSANNislands", 30)
        inner_control["nEliteSaved"] = control.get("nEliteSavedIslands", 3)
        inner_control["nelite"] = control.get("neliteIslands", 50)
        inner_control["npop"] = control.get("npopIslands", 200)
        
        if parallelizable:
            result = island_model_ga(
                data=data,
                candidates=candidates,
                setsizes=setsizes,
                settypes=settypes,
                stat_func=stat,
                n_stat=n_stat,
                target=target,
                control=inner_control,
                init_sol=init_sol,
                n_islands=nislands,
                n_jobs=n_jobs
            )
        else:
            result = island_model_ga(
                data=data,
                candidates=candidates,
                setsizes=setsizes,
                settypes=settypes,
                stat_func=stat,
                n_stat=n_stat,
                target=target,
                control=inner_control,
                init_sol=init_sol,
                n_islands=nislands,
                n_jobs=1
            )
    
    execution_time = time.time() - start_time
    
    sel_result = EvoResult(
        selected_indices=result["selected_indices"],
        selected_values=result["selected_values"],
        fitness=result["fitness"],
        fitness_history=result["fitness_history"],
        execution_time=execution_time,
        pareto_front=result.get("pareto_front", None),
        pareto_solutions=result.get("pareto_solutions", None),
        distribution=result.get("distribution", None),
        particle_solutions=result.get("particle_solutions", None),
        particle_weights=result.get("particle_weights", None)
    )
    
    if verbose:
        print(f"Optimization completed in {execution_time:.2f} seconds")
        print(f"Final fitness: {sel_result.fitness}")
    
    return sel_result


def time_estimation(
    nind: int, 
    nsel: int, 
    niter: int = 100, 
    control: Optional[ControlParams] = None
) -> float:
    """
    Estimate the time required for optimization.
    
    Parameters
    ----------
    nind : int
        Number of individuals.
    nsel : int
        Number of individuals to select.
    niter : int, optional
        Number of iterations.
    control : Dict[str, Any], optional
        Control object for the EvoSolve function.
        
    Returns
    -------
    float
        Estimated time in seconds.
    """
    if control is None:
        control = set_control_default()
    
    npop = control.get("npop", 1000)
    nislands = control.get("nislands", 1)
    
    base_time = 0.001  # Base time per evaluation.
    complexity_factor = 1 + (nsel / nind) * 10  # Complexity increases with the selection ratio.
    
    estimated_time = base_time * npop * niter * complexity_factor
    
    if nislands > 1:
        estimated_time = estimated_time * 0.8 * (1 + 0.1 * nislands)
    
    return estimated_time


def make_positive_definite(matrix, epsilon=1e-6):
    """
    Make a matrix positive definite by adding a small value to the diagonal.
    
    Parameters
    ----------
    matrix : ndarray
        Input matrix.
    epsilon : float, optional
        Small value to add to the diagonal.
        
    Returns
    -------
    ndarray
        Positive definite matrix.
    """
    try:
        np.linalg.cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        n = matrix.shape[0]
        return matrix + np.eye(n) * epsilon

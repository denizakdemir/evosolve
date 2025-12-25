"""
Distributional Optimization Head for TrainSelPy.

This module implements distributional optimization where the search operates
over distributions of solutions rather than individual solutions.

The core representation is particle distributions (weighted empirical measures):
    μ_θ = Σ_{i=1}^K w_i δ_{x_i}

where θ = (w_1,...,w_K, x_1,...,x_K) includes both weights and support points.
"""

import numpy as np
from typing import List, Dict, Callable, Union, Optional, Any, Tuple
from evosolve.solution import Solution
import copy


class ParticleDistribution:
    """
    Represents a distribution as weighted particles (empirical measure).
    
    This is the universal distribution representation that can wrap
    any decision variable type (BOOL, INT, DBL, OS, UOS, etc.).
    
    Attributes
    ----------
    particles : List[Solution]
        K support points (particles)
    weights : np.ndarray
        K weights (probabilities), sum to 1
    K : int
        Number of particles
    """
    
    def __init__(self, particles: List[Solution], weights: np.ndarray):
        """
        Create a particle distribution.

        Parameters
        ----------
        particles : List[Solution]
            Support points
        weights : np.ndarray
            Particle weights (will be normalized)
        """
        # Input validation
        if len(particles) == 0:
            raise ValueError("Particle list cannot be empty")

        self.particles = particles
        self.K = len(particles)

        # Normalize weights
        weights = np.asarray(weights, dtype=float)

        # Check for NaN or infinite weights
        if np.any(np.isnan(weights)):
            raise ValueError("Weights contain NaN values")
        if np.any(np.isinf(weights)):
            raise ValueError("Weights contain infinite values")

        # Handle zero-sum weights with warning
        if weights.sum() == 0:
            import warnings
            warnings.warn("All weights are zero; defaulting to uniform distribution")
            weights = np.ones(len(weights))

        self.weights = weights / weights.sum()

        assert len(self.particles) == len(self.weights), \
            "Number of particles must match number of weights"
    
    def sample(self, n: int, copy: bool = True) -> List[Solution]:
        """
        Sample n solutions from the distribution.

        Parameters
        ----------
        n : int
            Number of samples
        copy : bool, optional
            If True (default), return independent copies of the particles.
            If False, return references to the original particles.
            Set to False when samples will NOT be modified (e.g., for evaluation only).
            Set to True when samples will be modified (e.g., before mutation).

        Returns
        -------
        List[Solution]
            Sampled solutions (copies if copy=True, references if copy=False)

        Notes
        -----
        **Performance Optimization**: When sampling for read-only operations (e.g.,
        fitness evaluation), use `copy=False` to avoid expensive deep copying of
        large or complex solutions. Only use `copy=True` when the samples will be
        modified, such as before applying mutation operators.

        Examples
        --------
        >>> # For evaluation (read-only), use copy=False for better performance
        >>> samples = dist.sample(100, copy=False)
        >>> fitness_vals = [eval_fitness(s) for s in samples]
        >>>
        >>> # For mutation (will modify), use copy=True
        >>> samples = dist.sample(10, copy=True)
        >>> mutate(samples, ...)  # Safe to modify
        """
        # Sample particle indices according to weights
        indices = np.random.choice(self.K, size=n, p=self.weights)

        # Return copies or references based on flag
        if copy:
            samples = [self.particles[idx].copy() for idx in indices]
        else:
            samples = [self.particles[idx] for idx in indices]

        return samples
    
    def get_base_structure(self) -> Dict[str, Any]:
        """
        Extract information about the base decision variable structure.
        
        Returns
        -------
        Dict[str, Any]
            Structure metadata including:
            - has_int: whether particles have integer values
            - has_dbl: whether particles have double values
            - int_shapes: shapes of integer arrays
            - dbl_shapes: shapes of double arrays
        """
        if len(self.particles) == 0:
            return {
                'has_int': False,
                'has_dbl': False,
                'int_shapes': [],
                'dbl_shapes': []
            }
        
        # Use first particle as template
        template = self.particles[0]
        
        return {
            'has_int': len(template.int_values) > 0,
            'has_dbl': len(template.dbl_values) > 0,
            'int_shapes': [arr.shape for arr in template.int_values],
            'dbl_shapes': [arr.shape for arr in template.dbl_values]
        }


class DistributionalSolution:
    """
    Wrapper for representing a distribution as a solution in the GA.
    
    This allows distributional heads to integrate seamlessly with
    existing GA infrastructure.
    
    Attributes
    ----------
   distribution : ParticleDistribution
        The distribution this solution represents
    fitness : float
        Distributional objective value
    multi_fitness : List[float]
        For multi-objective optimization
    """
    
    def __init__(
        self,
        distribution: ParticleDistribution,
        fitness: float = float('-inf'),
        multi_fitness: List[float] = None
    ):
        self.distribution = distribution
        self.fitness = float(fitness)
        self.multi_fitness = list(multi_fitness) if multi_fitness is not None else []
    
    def copy(self):
        """Create a deep copy of the distributional solution."""
        # Deep copy the distribution
        new_particles = [p.copy() for p in self.distribution.particles]
        new_weights = self.distribution.weights.copy()
        new_dist = ParticleDistribution(new_particles, new_weights)
        
        # Copy fitness values
        multi_fit_copy = self.multi_fitness.copy() if self.multi_fitness else []
        
        return DistributionalSolution(new_dist, self.fitness, multi_fit_copy)
    
    def __lt__(self, other):
        """Comparison for sorting (by fitness)."""
        return self.fitness < other.fitness
    
    def get_hash(self) -> int:
        """
        Get a hash of the distributional solution.
        
        Combines hashes of all particles and their weights.
        """
        # Get hashes of all particles
        particle_hashes = [p.get_hash() for p in self.distribution.particles]
        
        # Combine with weight hash
        weights_hash = hash(self.distribution.weights.tobytes())
        
        # Return composite hash
        return hash(tuple(particle_hashes + [weights_hash]))


# =============================================================================
# Objective Functionals
# =============================================================================

def mean_objective(
    dist: ParticleDistribution,
    base_fitness_fn: Union[Callable, np.ndarray, List[float]],
    n_samples: int = 100,
    data: Dict[str, Any] = None
) -> float:
    """
    Mean objective functional: E[f(x)]
    
    Parameters
    ----------
    dist : ParticleDistribution
        Distribution to evaluate
    base_fitness_fn : Callable or array-like
        Base fitness function f(int_vals, dbl_vals, data) OR pre-computed fitness for each particle
    n_samples : int
        Number of Monte Carlo samples (only if base_fitness_fn is callable)
    data : Dict
        Data dictionary for fitness function
        
    Returns
    -------
    float
        Expected fitness
    """
    # Case 1: Pre-computed fitness values provided
    if not callable(base_fitness_fn):
        fitness_values = np.asarray(base_fitness_fn, dtype=float)
        if len(fitness_values) != dist.K:
            raise ValueError(f"Fitness array length ({len(fitness_values)}) must match number of particles ({dist.K})")
        # Exact expected value for discrete distribution
        return float(np.sum(dist.weights * fitness_values))

    # Case 2: Fitness function provided (Monte Carlo estimation)
    if data is None:
        data = {}
    
    # Sample from distribution (read-only evaluation, so no copy needed)
    samples = dist.sample(n_samples, copy=False)

    # Evaluate each sample
    fitness_values = []
    for sol in samples:
        f = base_fitness_fn(sol.int_values, sol.dbl_values, data)
        fitness_values.append(f)

    # Return mean
    return float(np.mean(fitness_values))


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure objective arrays are 2D (n_samples x n_obj)."""
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def evaluate_particle_objectives(
    dist: ParticleDistribution,
    base_fitness_fn: Callable,
    data: Dict[str, Any] = None
) -> np.ndarray:
    """
    Evaluate the base fitness function once per particle.

    Returns
    -------
    np.ndarray
        Array of shape (K, n_obj)
    """
    if data is None:
        data = {}

    def _call_base_fitness(sol: Solution) -> Any:
        """Align call signature with TrainSel convention (data last)."""
        has_int = bool(sol.int_values)
        has_dbl = bool(sol.dbl_values)
        if has_int and has_dbl:
            int_arg = sol.int_values if len(sol.int_values) > 1 else sol.int_values[0]
            dbl_arg = sol.dbl_values if len(sol.dbl_values) > 1 else sol.dbl_values[0]
            return base_fitness_fn(int_arg, dbl_arg, data)
        elif has_int:
            int_arg = sol.int_values if len(sol.int_values) > 1 else sol.int_values[0]
            return base_fitness_fn(int_arg, data)
        else:
            dbl_arg = sol.dbl_values if len(sol.dbl_values) > 1 else sol.dbl_values[0]
            return base_fitness_fn(dbl_arg, data)

    values = [
        _call_base_fitness(sol)
        for sol in dist.particles
    ]
    return _ensure_2d(np.asarray(values, dtype=float))


def sample_distribution_objectives(
    dist: ParticleDistribution,
    base_fitness_fn: Callable,
    n_samples: int,
    data: Dict[str, Any] = None
) -> np.ndarray:
    """
    Evaluate the base fitness function on Monte Carlo samples from the distribution.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_obj)
    """
    if data is None:
        data = {}
    # Sample for evaluation only (read-only), so no copy needed
    samples = dist.sample(n_samples, copy=False)

    def _call_base_fitness(sol: Solution) -> Any:
        # Handle stray DistributionalSolution particles by unwrapping first particle
        if hasattr(sol, "distribution") and hasattr(sol.distribution, "particles"):
            inner = sol.distribution.particles[0] if sol.distribution.particles else None
            if inner is not None:
                sol = inner
        has_int = bool(getattr(sol, "int_values", []))
        has_dbl = bool(getattr(sol, "dbl_values", []))
        if has_int and has_dbl:
            int_arg = sol.int_values if len(sol.int_values) > 1 else sol.int_values[0]
            dbl_arg = sol.dbl_values if len(sol.dbl_values) > 1 else sol.dbl_values[0]
            return base_fitness_fn(int_arg, dbl_arg, data)
        elif has_int:
            int_arg = sol.int_values if len(sol.int_values) > 1 else sol.int_values[0]
            return base_fitness_fn(int_arg, data)
        else:
            dbl_arg = sol.dbl_values if len(sol.dbl_values) > 1 else sol.dbl_values[0]
            return base_fitness_fn(dbl_arg, data)

    values = [_call_base_fitness(sol) for sol in samples]
    return _ensure_2d(np.asarray(values, dtype=float))


def compute_hv_2d(
    values: np.ndarray,
    hv_ref: Optional[Tuple[float, float]] = None,
    maximize: bool = True,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute a simple 2D hypervolume for maximization problems.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (n_points, 2)
    hv_ref : Tuple[float, float], optional
        Reference point (assumed dominated by all points). If None, uses a loose
        reference based on minimum values.
    maximize : bool
        If False, will flip signs to treat as maximization.
    weights : np.ndarray, optional
        Sample weights (if provided, will weight points before Pareto filtering)
    """
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 2 or vals.shape[1] != 2 or vals.size == 0:
        return 0.0
    # Convert to maximization
    if not maximize:
        vals = -vals
    # Weighted Pareto filter (approximate): keep points with highest weight for identical coords
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        # simple tie-breaker by weight
        order = np.lexsort((weights, vals[:, 0], vals[:, 1]))
        vals = vals[order]
    # Pareto prune
    vals = vals[np.argsort(-vals[:, 0])]
    pruned = []
    best_y = -np.inf
    for x, y in vals:
        if y > best_y:
            pruned.append((x, y))
            best_y = y
    if not pruned:
        return 0.0
    pruned = np.array(pruned)
    if hv_ref is None:
        ref_x = pruned[:, 0].min() - 0.1 * max(1.0, abs(pruned[:, 0].min()))
        ref_y = pruned[:, 1].min() - 0.1 * max(1.0, abs(pruned[:, 1].min()))
    else:
        ref_x, ref_y = hv_ref
        if not maximize:
            ref_x, ref_y = -ref_x, -ref_y
    hv = 0.0
    for i, (x, y) in enumerate(pruned):
        next_x = pruned[i + 1, 0] if i + 1 < len(pruned) else ref_x
        width = max(0.0, x - next_x)
        height = max(0.0, y - ref_y)
        hv += width * height
    return float(hv)


def _aggregate_cvar(
    values: np.ndarray,
    weights: Optional[np.ndarray],
    alpha: float,
    maximize: bool
) -> np.ndarray:
    """Compute CVaR per objective for weighted or unweighted samples."""
    n_samples, n_obj = values.shape
    results = []
    for j in range(n_obj):
        col = values[:, j]
        if weights is None:
            tail_size = max(1, int(alpha * n_samples))
            idx = np.argsort(col)
            if not maximize:
                idx = idx[::-1]
            tail_vals = col[idx[:tail_size]]
            results.append(float(np.mean(tail_vals)))
        else:
            idx = np.argsort(col)
            if not maximize:
                idx = idx[::-1]
            sorted_vals = col[idx]
            sorted_weights = weights[idx]
            cum_w = np.cumsum(sorted_weights)
            tail_mask = cum_w <= alpha + 1e-12
            if not np.any(tail_mask):
                results.append(float(sorted_vals[0]))
                continue
            tail_vals = sorted_vals[tail_mask]
            tail_weights = sorted_weights[tail_mask]
            weight_sum = float(np.sum(tail_weights))
            next_idx = tail_mask.sum()
            if weight_sum < alpha and next_idx < len(sorted_vals):
                tail_vals = np.append(tail_vals, sorted_vals[next_idx])
                tail_weights = np.append(tail_weights, alpha - weight_sum)
                weight_sum = alpha
            results.append(float(np.sum(tail_vals * tail_weights) / weight_sum))
    return np.asarray(results, dtype=float)


def aggregate_distributional_objectives(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    objective_type: str = "mean",
    lambda_var: float = 1.0,
    alpha: float = 0.1,
    tau: float = 0.1,
    maximize: bool = True,
    entropy_weights: Optional[np.ndarray] = None,
    hv_ref: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Aggregate sampled or per-particle objective values into a single vector.

    Parameters
    ----------
    values : np.ndarray
        Objective values with shape (n_samples, n_obj)
    weights : np.ndarray, optional
        Sample weights aligned with the first dimension of values
    objective_type : str
        Aggregation type ('mean', 'mean_variance', 'cvar', 'entropy', 'max_components')
    lambda_var : float
        Variance trade-off for mean_variance objective
    alpha : float
        Tail probability for CVaR
    tau : float
        Entropy weight for entropy-regularized objective
    maximize : bool
        Whether objectives are maximized (affects CVaR tail definition)
    entropy_weights : np.ndarray, optional
        Weights used for entropy calculation; defaults to provided weights or uniform
    hv_ref : Tuple[float, float], optional
        Reference point for hypervolume-based objectives (maximization). If None,
        uses a loose reference based on the data.
    """
    vals = _ensure_2d(np.asarray(values, dtype=float))
    if vals.size == 0:
        return np.array([])

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or len(w) != vals.shape[0]:
            raise ValueError("weights must be 1D and aligned with values")
        if w.sum() == 0:
            w = None
        else:
            w = w / w.sum()

    if objective_type == "mean":
        return np.mean(vals, axis=0) if w is None else np.sum(vals * w[:, None], axis=0)
    elif objective_type == "mean_variance":
        if w is None:
            mean = np.mean(vals, axis=0)
            var = np.var(vals, axis=0)
        else:
            mean = np.sum(vals * w[:, None], axis=0)
            var = np.sum(w[:, None] * (vals - mean) ** 2, axis=0)
        return mean + lambda_var * var
    elif objective_type == "cvar":
        return _aggregate_cvar(vals, w, alpha, maximize)
    elif objective_type == "entropy":
        base_mean = np.mean(vals, axis=0) if w is None else np.sum(vals * w[:, None], axis=0)
        ent_w = entropy_weights if entropy_weights is not None else w
        if ent_w is None:
            ent_w = np.ones(vals.shape[0]) / vals.shape[0]
        ent_w = np.asarray(ent_w, dtype=float)
        ent_w = ent_w[ent_w > 0]
        entropy = -np.sum(ent_w * np.log(ent_w))
        return base_mean + tau * entropy
    elif objective_type == "entropy_cvar":
        # Combine mean, entropy bonus, and tail emphasis
        base_mean = np.mean(vals, axis=0) if w is None else np.sum(vals * w[:, None], axis=0)
        ent_w = entropy_weights if entropy_weights is not None else w
        if ent_w is None:
            ent_w = np.ones(vals.shape[0]) / vals.shape[0]
        ent_w = np.asarray(ent_w, dtype=float)
        ent_w = ent_w[ent_w > 0]
        entropy = -np.sum(ent_w * np.log(ent_w))
        cvar_vals = _aggregate_cvar(vals, w, alpha, maximize)
        # lambda_var acts as weight on the cvar deviation from mean
        return base_mean + lambda_var * (cvar_vals - base_mean) + tau * entropy
    elif objective_type == "hypervolume_entropy":
        # Two-objective hypervolume + entropy bonus (returns 2 components)
        if vals.shape[1] != 2:
            raise ValueError("hypervolume_entropy requires exactly 2 objectives")
        hv = compute_hv_2d(vals, hv_ref=hv_ref, maximize=maximize, weights=w)
        ent_w = entropy_weights if entropy_weights is not None else w
        if ent_w is None:
            ent_w = np.ones(vals.shape[0]) / vals.shape[0]
        ent_w = np.asarray(ent_w, dtype=float)
        ent_w = ent_w[ent_w > 0]
        entropy = -np.sum(ent_w * np.log(ent_w))
        return np.array([hv, entropy], dtype=float)
    elif objective_type == "max_components":
        # Component-wise maxima across samples/particles (envelope view)
        return np.max(vals, axis=0)
    else:
        # Default to mean
        return np.mean(vals, axis=0) if w is None else np.sum(vals * w[:, None], axis=0)


def mean_variance_objective(
    dist: ParticleDistribution,
    base_fitness_fn: Union[Callable, np.ndarray, List[float]],
    n_samples: int = 100,
    lambda_var: float = 1.0,
    data: Dict[str, Any] = None
) -> float:
    """
    Mean-variance objective: E[f(x)] + λ·Var(f(x))
    
    Parameters
    ----------
    dist : ParticleDistribution
        Distribution to evaluate
    base_fitness_fn : Callable or array-like
        Base fitness function OR pre-computed fitness values
    n_samples : int
        Number of Monte Carlo samples (if callable)
    lambda_var : float
        Variance weight (positive for risk-seeking, negative for risk-averse)
    data : Dict
        Data dictionary
        
    Returns
    -------
    float
        Mean-variance objective value
    """
    # Case 1: Pre-computed fitness values provided
    if not callable(base_fitness_fn):
        fitness_values = np.asarray(base_fitness_fn, dtype=float)
        if len(fitness_values) != dist.K:
            raise ValueError(f"Fitness array length ({len(fitness_values)}) must match number of particles ({dist.K})")
            
        # Exact mean and variance
        mean_fit = np.sum(dist.weights * fitness_values)
        var_fit = np.sum(dist.weights * (fitness_values - mean_fit)**2)
        return float(mean_fit + lambda_var * var_fit)

    # Case 2: Fitness function provided (Monte Carlo estimation)
    if data is None:
        data = {}
    
    # Sample and evaluate (read-only, so no copy needed)
    samples = dist.sample(n_samples, copy=False)
    fitness_values = np.array([
        base_fitness_fn(sol.int_values, sol.dbl_values, data)
        for sol in samples
    ])

    mean_fitness = np.mean(fitness_values)
    var_fitness = np.var(fitness_values)

    return float(mean_fitness + lambda_var * var_fitness)


def cvar_objective(
    dist: ParticleDistribution,
    base_fitness_fn: Union[Callable, np.ndarray, List[float]],
    n_samples: int = 100,
    alpha: float = 0.1,
    maximize: bool = True,
    data: Dict[str, Any] = None
) -> float:
    """
    Conditional Value at Risk (CVaR) objective.
    
    CVaR_α = E[f(x) | f(x) in worst α-quantile]
    
    Parameters
    ----------
    dist : ParticleDistribution
        Distribution to evaluate
    base_fitness_fn : Callable or array-like
        Base fitness function OR pre-computed fitness values
    n_samples : int
        Number of Monte Carlo samples (if callable)
    alpha : float
        Quantile level (e.g., 0.1 for worst 10%)
    maximize : bool
        If True, worst means lowest values. If False, worst means highest.
    data : Dict
        Data dictionary
        
    Returns
    -------
    float
        CVaR value
    """
    # Case 1: Pre-computed fitness values provided (Exact discrete CVaR)
    if not callable(base_fitness_fn):
        fv = np.asarray(base_fitness_fn, dtype=float)
        if len(fv) != dist.K:
            raise ValueError(f"Fitness array length ({len(fv)}) must match number of particles ({dist.K})")
            
        # Sort by fitness
        indices = np.argsort(fv)
        if not maximize:
            indices = indices[::-1]  # Worst = highest values
            
        sorted_fv = fv[indices]
        sorted_weights = dist.weights[indices]
        
        # Accumulate weights until we reach alpha
        cum_weights = np.cumsum(sorted_weights)
        tail_mask = cum_weights <= alpha
        
        # If first particle is already > alpha, just take first particle
        if not np.any(tail_mask):
            return float(sorted_fv[0])
            
        # Expected value over the tail
        tail_fv = sorted_fv[tail_mask]
        tail_w = sorted_weights[tail_mask]
        
        # Add partial contribution from the next particle to reach exactly alpha
        if cum_weights[len(tail_w) - 1] < alpha and len(tail_w) < len(sorted_fv):
            next_idx = len(tail_w)
            partial_w = alpha - cum_weights[next_idx - 1]
            total_val = np.sum(tail_fv * tail_w) + sorted_fv[next_idx] * partial_w
            return float(total_val / alpha)
        
        return float(np.sum(tail_fv * tail_w) / np.sum(tail_w))

    # Case 2: Fitness function provided (Monte Carlo estimation)
    if data is None:
        data = {}

    # Sample and evaluate (read-only, so no copy needed)
    samples = dist.sample(n_samples, copy=False)
    fitness_values = np.array([
        base_fitness_fn(sol.int_values, sol.dbl_values, data)
        for sol in samples
    ])
    
    # Determine tail
    if maximize:
        # Worst = lowest values
        tail_size = max(1, int(alpha * len(fitness_values)))
        tail_indices = np.argpartition(fitness_values, tail_size)[:tail_size]
    else:
        # Worst = highest values
        tail_size = max(1, int(alpha * len(fitness_values)))
        tail_indices = np.argpartition(fitness_values, -tail_size)[-tail_size:]
    
    tail_values = fitness_values[tail_indices]
    
    return float(np.mean(tail_values))


def entropy_regularized_objective(
    dist: ParticleDistribution,
    base_fitness_fn: Union[Callable, np.ndarray, List[float]],
    n_samples: int = 100,
    tau: float = 0.1,
    data: Dict[str, Any] = None
) -> float:
    """
    Entropy-regularized objective: E[f(x)] + τ·H(μ)
    
    Encourages diversity by adding entropy bonus.
    
    Parameters
    ----------
    dist : ParticleDistribution
        Distribution to evaluate
    base_fitness_fn : Callable or array-like
        Base fitness function OR pre-computed fitness values
    n_samples : int
        Number of Monte Carlo samples (if callable)
    tau : float
        Entropy weight (positive encourages diversity)
    data : Dict
        Data dictionary
        
    Returns
    -------
    float
        Entropy-regularized objective value
    """
    # Compute entropy H(μ) = -Σ w_i log w_i
    w = dist.weights
    # Avoid log(0)
    w_safe = w[w > 0]
    entropy = -np.sum(w_safe * np.log(w_safe))

    # Compute expected fitness
    if not callable(base_fitness_fn):
        fv = np.asarray(base_fitness_fn, dtype=float)
        if len(fv) != dist.K:
            raise ValueError(f"Fitness array length ({len(fv)}) must match number of particles ({dist.K})")
        e_fit = np.sum(dist.weights * fv)
    else:
        e_fit = mean_objective(dist, base_fitness_fn, n_samples, data)

    return float(e_fit + tau * entropy)


# =============================================================================
# Compression Strategies
# =============================================================================

def compress_top_k(
    particles: Union[List[Solution], ParticleDistribution],
    weights: Optional[np.ndarray] = None,
    K: Optional[int] = None,
    **kwargs
) -> Union[Tuple[List[Solution], np.ndarray], ParticleDistribution]:
    """
    Keep top K particles by weight.
    
    Parameters
    ----------
    particles : List[Solution] or ParticleDistribution
        Original particles or distribution object
    weights : np.ndarray, optional
        Original weights (if particles is a list)
    K : int, optional
        Target number of particles. Can also use 'k' as keyword argument.
        
    Returns
    -------
    Tuple[List[Solution], np.ndarray] or ParticleDistribution
        Compressed result (same type as input)
    """
    # Handle keyword argument 'k'
    if K is None:
        K = kwargs.get('k')
    if K is None:
        raise ValueError("Target size 'K' or 'k' must be specified")

    # Handle ParticleDistribution input
    is_dist = isinstance(particles, ParticleDistribution)
    if is_dist:
        dist = particles
        p_list = dist.particles
        w_arr = dist.weights
    else:
        p_list = particles
        w_arr = weights
        if w_arr is None:
            raise ValueError("Weights must be provided if particles is a list")

    # Keep top K
    K_clamped = min(K, len(p_list))
    top_indices = np.argpartition(w_arr, -K_clamped)[-K_clamped:]
    
    # Sort them in descending order of weight
    top_indices = top_indices[np.argsort(w_arr[top_indices])[::-1]]
    
    new_particles = [p_list[i].copy() for i in top_indices]
    new_weights = w_arr[top_indices]
    new_weights = new_weights / new_weights.sum()
    
    if is_dist:
        return ParticleDistribution(new_particles, new_weights)
    return new_particles, new_weights


def compress_resampling(
    particles: Union[List[Solution], ParticleDistribution],
    weights: Optional[np.ndarray] = None,
    K: Optional[int] = None,
    **kwargs
) -> Union[Tuple[List[Solution], np.ndarray], ParticleDistribution]:
    """
    Compress via resampling K particles according to weights.
    
    Parameters
    ----------
    particles : List[Solution] or ParticleDistribution
        Original particles or distribution
    weights : np.ndarray, optional
        Original weights (if particles is a list)
    K : int, optional
        Target number of particles. Can also use 'k' as keyword argument.
        
    Returns
    -------
    Tuple[List[Solution], np.ndarray] or ParticleDistribution
        Resampled result (same type as input)
    """
    # Handle keyword argument 'k'
    if K is None:
        K = kwargs.get('k')
    if K is None:
        raise ValueError("Target size 'K' or 'k' must be specified")

    # Handle ParticleDistribution input
    is_dist = isinstance(particles, ParticleDistribution)
    if is_dist:
        dist = particles
        p_list = dist.particles
        w_arr = dist.weights
    else:
        p_list = particles
        w_arr = weights
        if w_arr is None:
            raise ValueError("Weights must be provided if particles is a list")

    # Resample indices
    indices = np.random.choice(len(p_list), size=K, p=w_arr)
    
    # Extract particles
    new_particles = [p_list[i].copy() for i in indices]
    
    # Uniform weights after resampling
    new_weights = np.ones(K) / K
    
    if is_dist:
        return ParticleDistribution(new_particles, new_weights)
    return new_particles, new_weights


def compress_kmeans(
    particles: Union[List[Solution], ParticleDistribution],
    weights: Optional[np.ndarray] = None,
    K: Optional[int] = None,
    **kwargs
) -> Union[Tuple[List[Solution], np.ndarray], ParticleDistribution]:
    """
    Compress via K-means clustering in the decision space.

    Attempts to use scikit-learn's KMeans if available. If not available,
    falls back to compress_top_k with a warning.

    Parameters
    ----------
    particles : List[Solution] or ParticleDistribution
        Original particles or distribution object
    weights : np.ndarray, optional
        Original weights (if particles is a list)
    K : int, optional
        Target number of particles (clusters). Can also use 'k' as keyword argument.

    Returns
    -------
    Tuple[List[Solution], np.ndarray] or ParticleDistribution
        Compressed result (same type as input)
    """
    # Handle keyword argument 'k'
    if K is None:
        K = kwargs.get('k')
    if K is None:
        raise ValueError("Target size 'K' or 'k' must be specified")

    # Handle ParticleDistribution input
    is_dist = isinstance(particles, ParticleDistribution)
    if is_dist:
        dist = particles
        p_list = dist.particles
        w_arr = dist.weights
    else:
        p_list = particles
        w_arr = weights
        if w_arr is None:
            raise ValueError("Weights must be provided if particles is a list")

    # If K >= number of particles, no compression needed
    if K >= len(p_list):
        if is_dist:
            return dist
        return p_list, w_arr

    # Try to use scikit-learn KMeans
    try:
        from sklearn.cluster import KMeans

        # Extract features from particles for clustering
        # We'll use a simple approach: flatten all int and dbl values
        features = []
        for p in p_list:
            feat = []
            for int_arr in p.int_values:
                feat.extend(int_arr.flatten().tolist())
            for dbl_arr in p.dbl_values:
                feat.extend(dbl_arr.flatten().tolist())
            features.append(feat)

        features_arr = np.array(features, dtype=float)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_arr, sample_weight=w_arr)

        # For each cluster, select the particle closest to the centroid
        new_particles = []
        new_weights = []

        for cluster_idx in range(K):
            cluster_mask = labels == cluster_idx
            if not np.any(cluster_mask):
                # Empty cluster, skip
                continue

            cluster_particles = [p_list[i] for i in range(len(p_list)) if cluster_mask[i]]
            cluster_weights = w_arr[cluster_mask]
            cluster_features = features_arr[cluster_mask]

            # Find particle closest to centroid
            centroid = kmeans.cluster_centers_[cluster_idx]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_idx = np.argmin(distances)

            # Add the closest particle with aggregated weight
            new_particles.append(cluster_particles[closest_idx].copy())
            new_weights.append(cluster_weights.sum())

        # Normalize weights
        new_weights = np.array(new_weights)
        new_weights = new_weights / new_weights.sum()

        if is_dist:
            return ParticleDistribution(new_particles, new_weights)
        return new_particles, new_weights

    except ImportError:
        # Scikit-learn not available, fall back to top_k
        import warnings
        warnings.warn(
            "scikit-learn not available for K-means clustering. "
            "Falling back to compress_top_k instead.",
            RuntimeWarning
        )
        return compress_top_k(particles, weights, K, **kwargs)


# =============================================================================
# Distribution Operators
# =============================================================================

def crossover_particle_mixture(
    dist1: Union[ParticleDistribution, DistributionalSolution],
    dist2: Union[ParticleDistribution, DistributionalSolution],
    alpha: float = 0.5,
    K_target: Optional[int] = None,
    **kwargs
) -> Union[ParticleDistribution, DistributionalSolution]:
    """
    Crossover by forming mixture of two distributions and compressing.
    
    μ_child = α·μ_1 + (1-α)·μ_2
    
    Parameters
    ----------
    dist1 : ParticleDistribution or DistributionalSolution
        First parent distribution
    dist2 : ParticleDistribution or DistributionalSolution
        Second parent distribution
    alpha : float
        Mixture weight for dist1. Can also use 'crossintensity'.
    K_target : int, optional
        Target number of particles (default: max of parents' K)
        
    Returns
    -------
    ParticleDistribution or DistributionalSolution
        Child distribution (type matches dist1)
    """
    # Handle parameter alias
    if alpha == 0.5:  # Default value
        alpha = kwargs.get(
            'crossintensity',
            kwargs.get('mix_prob', kwargs.get('mixprob', alpha))
        )

    # Handle DistributionalSolution input
    is_sol = isinstance(dist1, DistributionalSolution)
    d1 = dist1.distribution if is_sol else dist1
    d2 = dist2.distribution if isinstance(dist2, DistributionalSolution) else dist2
    
    # Form mixture
    mixed_particles = d1.particles + d2.particles
    mixed_weights = np.concatenate([
        alpha * d1.weights,
        (1 - alpha) * d2.weights
    ])
    
    # Compress if needed
    if K_target is None:
        K_target = max(d1.K, d2.K)
    
    if len(mixed_particles) > K_target:
        compressed_particles, compressed_weights = compress_top_k(
            mixed_particles, mixed_weights, K_target
        )
    else:
        compressed_particles = mixed_particles
        compressed_weights = mixed_weights / mixed_weights.sum()
    
    child_dist = ParticleDistribution(compressed_particles, compressed_weights)
    
    if is_sol:
        return DistributionalSolution(child_dist)
    return child_dist


def mutate_weights(
    dist: Union[ParticleDistribution, DistributionalSolution],
    weight_intensity: float = 0.1,
    **kwargs
) -> Union[ParticleDistribution, DistributionalSolution]:
    """
    Mutate distribution weights via logit perturbation.
    
    logit(w_i) ← logit(w_i) + σ·ε_i
    
    Parameters
    ----------
    dist : ParticleDistribution or DistributionalSolution
        Distribution to mutate
    weight_intensity : float
        Perturbation scale. Can also use 'mutintensity'.
        
    Returns
    -------
    ParticleDistribution or DistributionalSolution
        Mutated distribution (type matches input)
    """
    # Handle parameter alias
    if weight_intensity == 0.1:  # Default value
        weight_intensity = kwargs.get(
            'mutation_strength',
            kwargs.get('mutintensity', weight_intensity)
        )

    # Handle DistributionalSolution input
    is_sol = isinstance(dist, DistributionalSolution)
    d = dist.distribution if is_sol else dist

    # Convert to logits
    eps = 1e-10
    weights = np.clip(d.weights, eps, 1 - eps)
    logits = np.log(weights / (1 - weights))
    
    # Add noise
    noise = np.random.normal(0, weight_intensity, size=len(logits))
    new_logits = logits + noise
    
    # Convert back to probabilities and normalize
    new_weights = 1 / (1 + np.exp(-new_logits))
    new_weights = new_weights / new_weights.sum()
    
    # Create new distribution with same particles but new weights
    new_particles = [p.copy() for p in d.particles]
    
    new_dist = ParticleDistribution(new_particles, new_weights)
    
    if is_sol:
        return DistributionalSolution(new_dist)
    return new_dist


def mutate_support(
    dist: Union[ParticleDistribution, DistributionalSolution],
    base_mutate_fn: Optional[Callable],
    candidates: List[List[int]],
    settypes: List[str],
    support_prob: float = 0.5,
    mutintensity: float = 0.1,
    **kwargs
) -> Union[ParticleDistribution, DistributionalSolution]:
    """
    Mutate support points using base mutation operator.
    
    Parameters
    ----------
    dist : ParticleDistribution or DistributionalSolution
        Distribution to mutate
    base_mutate_fn : Callable, optional
        Base mutation function (e.g., from operators.py). Defaults to evosolve.operators.mutation.
    candidates : List[List[int]]
        Candidates for base decision variables
    settypes : List[str]
        Set types for base variables
    support_prob : float
        Probability of mutating each particle. Can also use 'mutprob' or 'mutation_prob'.
    mutintensity : float
        Mutation intensity
        
    Returns
    -------
    ParticleDistribution or DistributionalSolution
        Mutated distribution (type matches input)
    """
    if base_mutate_fn is None:
        # Default to the standard mutation operator for convenience/backward compatibility
        from evosolve.operators import mutation as base_mutate_fn  # Local import to avoid cycles

    # Handle parameter aliases
    if support_prob == 0.5:  # Default value
        support_prob = kwargs.get('mutprob', kwargs.get('mutation_prob', support_prob))
    # Handle DistributionalSolution input
    is_sol = isinstance(dist, DistributionalSolution)
    d = dist.distribution if is_sol else dist

    # Copy particles
    new_particles = [p.copy() for p in d.particles]
    
    # Apply base mutation to particles
    base_mutate_fn(
        new_particles,
        candidates,
        settypes,
        mutprob=support_prob,
        mutintensity=mutintensity
    )
    
    # Keep same weights
    new_weights = d.weights.copy()
    
    new_dist = ParticleDistribution(new_particles, new_weights)
    
    if is_sol:
        return DistributionalSolution(new_dist)
    return new_dist


def birth_death_mutation(
    dist: Union[ParticleDistribution, DistributionalSolution],
    candidates: List[List[int]],
    setsizes: List[int],
    settypes: List[str],
    birth_rate: float = 0.1,
    death_rate: float = 0.1,
    **kwargs
) -> Union[ParticleDistribution, DistributionalSolution]:
    """
    Birth-death mutation: add and remove particles.
    
    Parameters
    ----------
    dist : ParticleDistribution or DistributionalSolution
        Distribution to mutate
    candidates : List[List[int]]
        Candidates for new particles
    setsizes : List[int]
        Set sizes for new particles
    settypes : List[str]
        Set types for new particles
    birth_rate : float
        Fraction of particles to add
    death_rate : float
        Fraction of particles to remove
        
    Returns
    -------
    ParticleDistribution or DistributionalSolution
        Mutated distribution (type matches input)
    """
    # Handle parameter aliases if any (none common yet but added for consistency)
    birth_rate = kwargs.get('birth_prob', birth_rate)
    death_rate = kwargs.get('death_prob', death_rate)
    from evosolve.algorithms import initialize_population
    
    # Handle DistributionalSolution input
    is_sol = isinstance(dist, DistributionalSolution)
    d = dist.distribution if is_sol else dist

    # Death: remove lowest-weight particles
    n_death = max(0, int(death_rate * d.K))
    if n_death > 0 and n_death < d.K:
        # Keep top (K - n_death) particles
        keep_indices = np.argpartition(d.weights, -d.K + n_death)[-d.K + n_death:]
        new_particles = [d.particles[i] for i in keep_indices]
        new_weights = d.weights[keep_indices]
        new_weights = new_weights / new_weights.sum()
    else:
        new_particles = [p.copy() for p in d.particles]
        new_weights = d.weights.copy()
    
    # Birth: add new random particles
    n_birth = max(0, int(birth_rate * d.K))
    if n_birth > 0:
        # Generate new particles
        new_born = initialize_population(candidates, setsizes, settypes, pop_size=n_birth)
        
        # Add to distribution with small uniform weight
        total_new_weight = 0.1  # Allocate 10% weight to new particles
        birth_weight = total_new_weight / n_birth
        
        # Renormalize existing weights
        new_weights = new_weights * (1 - total_new_weight)
        
        # Combine
        new_particles.extend(new_born)
        new_weights = np.concatenate([new_weights, np.full(n_birth, birth_weight)])
    
    new_dist = ParticleDistribution(new_particles, new_weights)
    
    if is_sol:
        return DistributionalSolution(new_dist)
    return new_dist

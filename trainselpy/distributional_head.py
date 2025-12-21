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
from trainselpy.solution import Solution
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
        self.particles = particles
        self.K = len(particles)
        
        # Normalize weights
        weights = np.asarray(weights, dtype=float)
        if weights.sum() == 0:
            weights = np.ones(len(weights))
        self.weights = weights / weights.sum()
        
        assert len(self.particles) == len(self.weights), \
            "Number of particles must match number of weights"
    
    def sample(self, n: int) -> List[Solution]:
        """
        Sample n solutions from the distribution.
        
        Parameters
        ----------
        n : int
            Number of samples
            
        Returns
        -------
        List[Solution]
            Sampled solutions (independent copies)
        """
        # Sample particle indices according to weights
        indices = np.random.choice(self.K, size=n, p=self.weights)
        
        # Return copies of the selected particles
        samples = [self.particles[idx].copy() for idx in indices]
        
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


# =============================================================================
# Objective Functionals
# =============================================================================

def mean_objective(
    dist: ParticleDistribution,
    base_fitness_fn: Callable,
    n_samples: int = 100,
    data: Dict[str, Any] = None
) -> float:
    """
    Mean objective functional: E[f(x)]
    
    Parameters
    ----------
    dist : ParticleDistribution
        Distribution to evaluate
    base_fitness_fn : Callable
        Base fitness function f(int_vals, dbl_vals, data)
    n_samples : int
        Number of Monte Carlo samples
    data : Dict
        Data dictionary for fitness function
        
    Returns
    -------
    float
        Expected fitness
    """
    if data is None:
        data = {}
    
    # Sample from distribution
    samples = dist.sample(n_samples)
    
    # Evaluate each sample
    fitness_values = []
    for sol in samples:
        f = base_fitness_fn(sol.int_values, sol.dbl_values, data)
        fitness_values.append(f)
    
    # Return mean
    return float(np.mean(fitness_values))


def mean_variance_objective(
    dist: ParticleDistribution,
    base_fitness_fn: Callable,
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
    base_fitness_fn : Callable
        Base fitness function
    n_samples : int
        Number of Monte Carlo samples
    lambda_var : float
        Variance weight (positive for risk-seeking, negative for risk-averse)
    data : Dict
        Data dictionary
        
    Returns
    -------
    float
        Mean-variance objective value
    """
    if data is None:
        data = {}
    
    # Sample and evaluate
    samples = dist.sample(n_samples)
    fitness_values = np.array([
        base_fitness_fn(sol.int_values, sol.dbl_values, data)
        for sol in samples
    ])
    
    mean_fitness = np.mean(fitness_values)
    var_fitness = np.var(fitness_values)
    
    return float(mean_fitness + lambda_var * var_fitness)


def cvar_objective(
    dist: ParticleDistribution,
    base_fitness_fn: Callable,
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
    base_fitness_fn : Callable
        Base fitness function
    n_samples : int
        Number of Monte Carlo samples
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
    if data is None:
        data = {}
    
    # Sample and evaluate
    samples = dist.sample(n_samples)
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
    base_fitness_fn: Callable,
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
    base_fitness_fn : Callable
        Base fitness function
    n_samples : int
        Number of Monte Carlo samples
    tau : float
        Entropy weight (positive encourages diversity)
    data : Dict
        Data dictionary
        
    Returns
    -------
    float
        Entropy-regularized objective value
    """
    if data is None:
        data = {}
    
    # Compute mean fitness
    mean_fit = mean_objective(dist, base_fitness_fn, n_samples, data)
    
    # Compute entropy H(μ) = -Σ w_i log(w_i)
    # Handle zero weights
    weights = dist.weights
    log_weights = np.log(weights + 1e-10)
    entropy = -np.sum(weights * log_weights)
    
    return float(mean_fit + tau * entropy)


# =============================================================================
# Compression Strategies
# =============================================================================

def compress_top_k(
    particles: List[Solution],
    weights: np.ndarray,
    K: int
) -> Tuple[List[Solution], np.ndarray]:
    """
    Keep top K particles by weight.
    
    Parameters
    ----------
    particles : List[Solution]
        Original particles
    weights : np.ndarray
        Original weights
    K : int
        Target number of particles
        
    Returns
    -------
    Tuple[List[Solution], np.ndarray]
        Compressed particles and renormalized weights
    """
    if len(particles) <= K:
        return particles, weights / weights.sum()
    
    # Get indices of top K weights
    top_indices = np.argpartition(weights, -K)[-K:]
    top_indices = top_indices[np.argsort(-weights[top_indices])]  # Sort descending
    
    # Extract and renormalize
    new_particles = [particles[i] for i in top_indices]
    new_weights = weights[top_indices]
    new_weights = new_weights / new_weights.sum()
    
    return new_particles, new_weights


def compress_resampling(
    particles: List[Solution],
    weights: np.ndarray,
    K: int
) -> Tuple[List[Solution], np.ndarray]:
    """
    Compress via resampling K particles according to weights.
    
    Parameters
    ----------
    particles : List[Solution]
        Original particles
    weights : np.ndarray
        Original weights
    K : int
        Target number of particles
        
    Returns
    -------
    Tuple[List[Solution], np.ndarray]
        Resampled particles with uniform weights
    """
    # Resample indices
    indices = np.random.choice(len(particles), size=K, p=weights)
    
    # Extract particles
    new_particles = [particles[i].copy() for i in indices]
    
    # Uniform weights after resampling
    new_weights = np.ones(K) / K
    
    return new_particles, new_weights


def compress_kmeans(
    particles: List[Solution],
    weights: np.ndarray,
    K: int
) -> Tuple[List[Solution], np.ndarray]:
    """
    Compress via clustering (simplified implementation).
    
    Currently implemented as top-K for robustness.
    Full k-means clustering would require distance metric on solution space.
    
    Parameters
    ----------
    particles : List[Solution]
        Original particles
    weights : np.ndarray
        Original weights
    K : int
        Target number of particles
        
    Returns
    -------
    Tuple[List[Solution], np.ndarray]
        Compressed particles and weights
    """
    # Simplified: use top-K
    # TODO: Implement true clustering if distance metric is available
    return compress_top_k(particles, weights, K)


# =============================================================================
# Distribution Operators
# =============================================================================

def crossover_particle_mixture(
    dist1: ParticleDistribution,
    dist2: ParticleDistribution,
    alpha: float = 0.5,
    K_target: Optional[int] = None
) -> ParticleDistribution:
    """
    Crossover by forming mixture of two distributions and compressing.
    
    μ_child = α·μ_1 + (1-α)·μ_2
    
    Parameters
    ----------
    dist1 : ParticleDistribution
        First parent distribution
    dist2 : ParticleDistribution
        Second parent distribution
    alpha : float
        Mixture weight for dist1
    K_target : int, optional
        Target number of particles (default: max of parents' K)
        
    Returns
    -------
    ParticleDistribution
        Child distribution
    """
    # Form mixture
    mixed_particles = dist1.particles + dist2.particles
    mixed_weights = np.concatenate([
        alpha * dist1.weights,
        (1 - alpha) * dist2.weights
    ])
    
    # Compress if needed
    if K_target is None:
        K_target = max(dist1.K, dist2.K)
    
    if len(mixed_particles) > K_target:
        compressed_particles, compressed_weights = compress_top_k(
            mixed_particles, mixed_weights, K_target
        )
    else:
        compressed_particles = mixed_particles
        compressed_weights = mixed_weights / mixed_weights.sum()
    
    return ParticleDistribution(compressed_particles, compressed_weights)


def mutate_weights(
    dist: ParticleDistribution,
    weight_intensity: float = 0.1
) -> ParticleDistribution:
    """
    Mutate distribution weights via logit perturbation.
    
    logit(w_i) ← logit(w_i) + σ·ε_i
    
    Parameters
    ----------
    dist : ParticleDistribution
        Distribution to mutate
    weight_intensity : float
        Perturbation scale
        
    Returns
    -------
    ParticleDistribution
        Mutated distribution (new object)
    """
    # Convert to logits
    eps = 1e-10
    weights = np.clip(dist.weights, eps, 1 - eps)
    logits = np.log(weights / (1 - weights))
    
    # Add noise
    noise = np.random.normal(0, weight_intensity, size=len(logits))
    new_logits = logits + noise
    
    # Convert back to probabilities and normalize
    new_weights = 1 / (1 + np.exp(-new_logits))
    new_weights = new_weights / new_weights.sum()
    
    # Create new distribution with same particles but new weights
    new_particles = [p.copy() for p in dist.particles]
    
    return ParticleDistribution(new_particles, new_weights)


def mutate_support(
    dist: ParticleDistribution,
    base_mutate_fn: Callable,
    candidates: List[List[int]],
    settypes: List[str],
    support_prob: float = 0.5,
    mutintensity: float = 0.1
) -> ParticleDistribution:
    """
    Mutate support points using base mutation operator.
    
    Parameters
    ----------
    dist : ParticleDistribution
        Distribution to mutate
    base_mutate_fn : Callable
        Base mutation function (e.g., from operators.py)
    candidates : List[List[int]]
        Candidates for base decision variables
    settypes : List[str]
        Set types for base variables
    support_prob : float
        Probability of mutating each particle
    mutintensity : float
        Mutation intensity
        
    Returns
    -------
    ParticleDistribution
        Mutated distribution
    """
    # Copy particles
    new_particles = [p.copy() for p in dist.particles]
    
    # Apply base mutation to particles
    base_mutate_fn(
        new_particles,
        candidates,
        settypes,
        mutprob=support_prob,
        mutintensity=mutintensity
    )
    
    # Keep same weights
    new_weights = dist.weights.copy()
    
    return ParticleDistribution(new_particles, new_weights)


def birth_death_mutation(
    dist: ParticleDistribution,
    candidates: List[List[int]],
    setsizes: List[int],
    settypes: List[str],
    birth_rate: float = 0.1,
    death_rate: float = 0.1
) -> ParticleDistribution:
    """
    Birth-death mutation: add and remove particles.
    
    Parameters
    ----------
    dist : ParticleDistribution
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
    ParticleDistribution
        Mutated distribution
    """
    from trainselpy.algorithms import initialize_population
    
    # Death: remove lowest-weight particles
    n_death = max(0, int(death_rate * dist.K))
    if n_death > 0 and n_death < dist.K:
        # Keep top (K - n_death) particles
        keep_indices = np.argpartition(dist.weights, -dist.K + n_death)[-dist.K + n_death:]
        new_particles = [dist.particles[i] for i in keep_indices]
        new_weights = dist.weights[keep_indices]
        new_weights = new_weights / new_weights.sum()
    else:
        new_particles = [p.copy() for p in dist.particles]
        new_weights = dist.weights.copy()
    
    # Birth: add new random particles
    n_birth = max(0, int(birth_rate * dist.K))
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
    
    return ParticleDistribution(new_particles, new_weights)

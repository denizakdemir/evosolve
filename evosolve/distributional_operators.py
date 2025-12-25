"""
Distributional Operators for GA Integration.

This module provides thin wrappers around the distributional head primitives
to integrate them with EvoSolve's GA infrastructure.
"""

import numpy as np
from typing import List, Dict, Callable, Optional, Any
from evosolve.solution import Solution
from evosolve.distributional_head import (
    ParticleDistribution,
    DistributionalSolution,
    crossover_particle_mixture,
    mutate_weights,
    mutate_support,
    birth_death_mutation
)


def distributional_crossover(
    parents: List[DistributionalSolution],
    crossprob: float,
    crossintensity: float,
    settypes: List[str],
    candidates: List[List[int]],
    control: Dict[str, Any]
) -> List[DistributionalSolution]:
    """
    Crossover operator for distributional solutions.
    
    Applies mixture crossover to pairs of parent distributions.
    
    Parameters
    ----------
    parents : List[DistributionalSolution]
        Parent distributions
    crossprob : float
        Probability of performing crossover
    crossintensity : float
        Crossover intensity (used as alpha for mixture)
    settypes : List[str]
        Set types (should contain "DIST:TYPE")
    candidates : List[List[int]]
        Candidate sets
    control : Dict[str, Any]
        Control parameters
        
    Returns
    -------
    List[DistributionalSolution]
        Offspring distributions
    """
    offspring = []
    K_target = control.get("dist_K_particles", 10)
    
    # Process pairs
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents) and np.random.rand() < crossprob:
            p1, p2 = parents[i], parents[i + 1]
            
            # Mixture crossover
            alpha = crossintensity  # Use crossintensity as mixture weight
            child_dist = crossover_particle_mixture(
                p1.distribution,
                p2.distribution,
                alpha=alpha,
                K_target=K_target
            )
            
            # Create new distributional solution
            child = DistributionalSolution(child_dist)
            offspring.append(child)
            
            # Create second child with reversed alpha
            child_dist2 = crossover_particle_mixture(
                p2.distribution,
                p1.distribution,
                alpha=alpha,
                K_target=K_target
            )
            child2 = DistributionalSolution(child_dist2)
            offspring.append(child2)
        else:
            # No crossover, copy parents
            offspring.append(parents[i].copy())
            if i + 1 < len(parents):
                offspring.append(parents[i + 1].copy())
    
    return offspring


def distributional_mutation(
    offspring: List[DistributionalSolution],
    candidates: List[List[int]],
    settypes: List[str],
    setsizes: List[int],
    mutprob: float,
    mutintensity: float,
    control: Dict[str, Any]
) -> None:
    """
    Mutation operator for distributional solutions.
    
    Combines weight mutation and support mutation.
    
    Parameters
    ----------
    offspring : List[DistributionalSolution]
        Offspring distributions to mutate (modified in-place)
    candidates : List[List[int]]
        Candidate sets
    settypes : List[str]
        Set types (should contain "DIST:TYPE")
    mutprob : float
        Mutation probability
    mutintensity : float
        Mutation intensity
    control : Dict[str, Any]
        Control parameters
    """
    from evosolve.operators import mutation
    
    # Extract base type from DIST:TYPE
    base_type = settypes[0].split(":")[1] if ":" in settypes[0] else "BOOL"
    base_settypes = [base_type]
    
    # Get mutation probabilities from control
    weight_mut_prob = control.get("dist_weight_mutation_prob", 0.5)
    support_mut_prob = control.get("dist_support_mutation_prob", 0.5)
    
    birth_death_prob = control.get("dist_birth_death_prob", 0.0)
    birth_rate = control.get("dist_birth_rate", 0.1)
    death_rate = control.get("dist_death_rate", 0.1)

    for sol in offspring:
        # Mutate weights
        if np.random.rand() < weight_mut_prob:
            sol.distribution = mutate_weights(
                sol.distribution,
                weight_intensity=mutintensity
            )
        
        # Mutate support (particles)
        if np.random.rand() < support_mut_prob:
            sol.distribution = mutate_support(
                sol.distribution,
                base_mutate_fn=mutation,
                candidates=candidates,
                settypes=base_settypes,
                support_prob=mutprob,
                mutintensity=mutintensity
            )

        # Optional birth-death mutation to refresh supports
        if birth_death_prob > 0 and np.random.rand() < birth_death_prob:
            sol.distribution = birth_death_mutation(
                sol.distribution,
                candidates=candidates,
                setsizes=setsizes,
                settypes=base_settypes,
                birth_rate=birth_rate,
                death_rate=death_rate
            )


def initialize_distributional_population(
    candidates: List[List[int]],
    setsizes: List[int],
    settypes: List[str],
    pop_size: int,
    control: Dict[str, Any]
) -> List[DistributionalSolution]:
    """
    Initialize population of distributional solutions.

    Parameters
    ----------
    candidates : List[List[int]]
        Candidate sets
    setsizes : List[int]
        Set sizes
    settypes : List[str]
        Set types (should contain "DIST:TYPE")
    pop_size : int
        Population size
    control : Dict[str, Any]
        Control parameters

    Returns
    -------
    List[DistributionalSolution]
        Initial population of distributions
    """
    from evosolve.algorithms import initialize_population

    # Check for mixed schemas (distributional + non-distributional types)
    has_dist = any("DIST:" in st for st in settypes)
    has_non_dist = any("DIST:" not in st for st in settypes)

    if has_dist and has_non_dist:
        raise NotImplementedError(
            "Mixed schemas with both distributional (DIST:X) and non-distributional types "
            "are not currently supported. All settypes must be either distributional or "
            "non-distributional, but not a mixture."
        )

    # Check that all distributional types have the same base type
    if has_dist:
        base_types = []
        for st in settypes:
            if "DIST:" in st:
                base_type = st.split(":")[1] if ":" in st else "BOOL"
                base_types.append(base_type)

        if len(set(base_types)) > 1:
            raise NotImplementedError(
                f"Multiple distributional base types {set(base_types)} are not currently supported. "
                "All DIST: types must have the same base type."
            )

    # Extract base type from DIST:TYPE
    base_type = settypes[0].split(":")[1] if ":" in settypes[0] else "BOOL"
    base_settypes = [base_type]
    
    # Get number of particles per distribution
    K_particles = control.get("dist_K_particles", 10)
    
    # Initialize population of distributions
    population = []
    for _ in range(pop_size):
        # Initialize K particles of base type
        particles = initialize_population(
            candidates=candidates,
            setsizes=setsizes,
            settypes=base_settypes,
            pop_size=K_particles
        )
        
        # Create uniform distribution over particles
        weights = np.ones(K_particles) / K_particles
        dist = ParticleDistribution(particles, weights)
        
        # Wrap in DistributionalSolution
        dsol = DistributionalSolution(dist)
        population.append(dsol)
    
    return population

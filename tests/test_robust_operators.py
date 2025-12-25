"""
Tests for Robust Distributional Operators.

This script verifies that distributional operators handle:
1. Both ParticleDistribution and DistributionalSolution objects.
2. Parameter aliases (mutintensity, crossintensity).
3. Correct return types.
"""

import numpy as np
import pytest
from evosolve.solution import Solution
from evosolve.distributional_head import (
    ParticleDistribution,
    DistributionalSolution,
    crossover_particle_mixture,
    mutate_weights,
    mutate_support,
    birth_death_mutation
)

# Mock mutation function
def mock_mutate(particles, candidates, settypes, mutprob=0.1, mutintensity=0.1):
    for p in particles:
        if len(p.int_values) > 0:
            p.int_values[0] = p.int_values[0].copy()
            p.int_values[0][0] += 1
    return particles

def create_mock_dist(k=5):
    particles = []
    for i in range(k):
        sol = Solution()
        sol.int_values = [np.array([i])]
        particles.append(sol)
    weights = np.ones(k) / k
    return ParticleDistribution(particles, weights)

def test_crossover_robustness():
    d1 = create_mock_dist(k=5)
    d2 = create_mock_dist(k=5)
    s1 = DistributionalSolution(d1)
    s2 = DistributionalSolution(d2)
    
    # Test ParticleDistribution input
    res_d = crossover_particle_mixture(d1, d2, alpha=0.3)
    assert isinstance(res_d, ParticleDistribution)
    
    # Test DistributionalSolution input
    res_s = crossover_particle_mixture(s1, s2, crossintensity=0.3)
    assert isinstance(res_s, DistributionalSolution)
    
    # Test mixed input
    res_m = crossover_particle_mixture(d1, s2, alpha=0.5)
    assert isinstance(res_m, ParticleDistribution)

def test_crossover_respects_mix_prob_alias():
    d1 = create_mock_dist(k=2)
    d2 = create_mock_dist(k=2)

    res = crossover_particle_mixture(d1, d2, mix_prob=0.8, K_target=4)

    # First two weights come from d1 and should sum to the mix probability
    assert res.K == 4
    assert pytest.approx(res.weights[: d1.K].sum(), rel=1e-6) == 0.8

def test_mutate_weights_robustness():
    d = create_mock_dist(k=5)
    s = DistributionalSolution(d)
    
    # Test ParticleDistribution input
    res_d = mutate_weights(d, weight_intensity=0.2)
    assert isinstance(res_d, ParticleDistribution)
    assert not np.array_equal(res_d.weights, d.weights)
    
    # Test DistributionalSolution input
    res_s = mutate_weights(s, mutintensity=0.2)
    assert isinstance(res_s, DistributionalSolution)
    assert not np.array_equal(res_s.distribution.weights, s.distribution.weights)

def test_mutate_weights_mutation_strength_alias():
    d = create_mock_dist(k=4)
    # Zero strength should leave weights unchanged if alias is honored
    res = mutate_weights(d, mutation_strength=0.0)
    assert np.allclose(res.weights, d.weights)

def test_mutate_support_robustness():
    d = create_mock_dist(k=5)
    s = DistributionalSolution(d)
    
    # Test DistributionalSolution input
    res_s = mutate_support(
        s, 
        base_mutate_fn=mock_mutate,
        candidates=[[0, 1, 2]],
        settypes=["INT"],
        support_prob=1.0
    )
    assert isinstance(res_s, DistributionalSolution)
    # Check if mutated (int values increased by 1)
    assert res_s.distribution.particles[0].int_values[0][0] == d.particles[0].int_values[0][0] + 1

    # Test parameter alias 'mutation_prob'
    res_alias = mutate_support(
        s,
        base_mutate_fn=mock_mutate,
        candidates=[[0, 1, 2]],
        settypes=["INT"],
        mutation_prob=1.0
    )
    assert isinstance(res_alias, DistributionalSolution)
    
    # Test parameter alias 'mutprob'
    res_alias2 = mutate_support(
        s,
        base_mutate_fn=mock_mutate,
        candidates=[[0, 1, 2]],
        settypes=["INT"],
        mutprob=1.0
    )
    assert isinstance(res_alias2, DistributionalSolution)

def test_mutate_support_defaults_to_standard_mutation():
    # Use UOS structure to ensure mutation alters elements
    particles = [Solution(int_values=[np.array([0, 1, 2])]) for _ in range(3)]
    weights = np.ones(3) / 3
    dist = ParticleDistribution(particles, weights)

    mutated = mutate_support(
        dist,
        base_mutate_fn=None,
        candidates=[list(range(6))],
        settypes=["UOS"],
        mutation_prob=1.0,
        mutintensity=0.5
    )

    changed = any(
        not np.array_equal(orig.int_values[0], new.int_values[0])
        for orig, new in zip(dist.particles, mutated.particles)
    )
    assert changed

def test_birth_death_robustness():
    d = create_mock_dist(k=10)
    s = DistributionalSolution(d)
    
    # Test DistributionalSolution input
    # birth_death uses initialize_population, which we'll mock or let fail if it tries to call real GA
    # Actually initialize_population is imported inside birth_death_mutation
    
    # We need to provide valid candidates etc. if birth_rate > 0
    res_s = birth_death_mutation(
        s,
        candidates=[[1, 2, 3]],
        setsizes=[1],
        settypes=["INT"],
        birth_rate=0.2,
        death_rate=0.2
    )
    assert isinstance(res_s, DistributionalSolution)
    # K might change depending on birth/death rates
    # K_orig = 10, death = 2, K_after_death = 8, birth = 2, K_final = 10
    assert len(res_s.distribution.particles) == 10

if __name__ == "__main__":
    pytest.main([__file__])

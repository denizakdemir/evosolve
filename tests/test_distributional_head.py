"""
Tests for distributional optimization head.

Following TDD methodology - tests written before implementation.
"""

import pytest
import numpy as np
from evosolve.solution import Solution
from evosolve.algorithms import (
    initialize_population,
    evaluate_fitness,
    _select_next_generation,
)
from evosolve.selection import fast_non_dominated_sort
from evosolve.core import evolve_control

# Tests will be written for distributional_head module
try:
    from evosolve.distributional_head import (
        ParticleDistribution,
        DistributionalSolution,
        mean_objective,
        mean_variance_objective,
        cvar_objective,
        entropy_regularized_objective,
        crossover_particle_mixture,
        mutate_support,
        mutate_weights,
        birth_death_mutation,
        compress_top_k,
        compress_kmeans,
        compress_resampling,
    )
    HAS_DIST_HEAD = True
except ImportError:
    HAS_DIST_HEAD = False


class TestParticleDistribution:
    """Test particle distribution representation and basic operations."""
    
    def test_initialization_basic(self):
        """Test creating a particle distribution with K particles."""
        # Create some simple particles (binary solutions)
        particles = []
        for i in range(10):
            sol = Solution(
                int_values=[np.array([0, 1, 0, 1, 1])],
                dbl_values=[],
                fitness=float(i)
            )
            particles.append(sol)
        
        weights = np.ones(10) / 10  # Uniform weights
        
        dist = ParticleDistribution(particles, weights)
        
        assert dist.K == 10
        assert len(dist.particles) == 10
        assert np.allclose(dist.weights.sum(), 1.0)
        assert np.allclose(dist.weights, weights)
    
    def test_weight_normalization(self):
        """Test that weights are automatically normalized."""
        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(5)
        ]
        weights = np.array([1, 2, 3, 4, 5])  # Not normalized
        
        dist = ParticleDistribution(particles, weights)
        
        assert np.allclose(dist.weights.sum(), 1.0)
        assert np.allclose(dist.weights, weights / weights.sum())
    
    def test_sampling_basic(self):
        """Test sampling solutions from the distribution."""
        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(5)
        ]
        weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        
        dist = ParticleDistribution(particles, weights)
        
        # Sample 100 solutions
        samples = dist.sample(100)
        
        assert len(samples) == 100
        assert all(isinstance(s, Solution) for s in samples)
        
        # Check that sampling respects weights (roughly)
        # First particle should appear ~50 times
        first_particle_count = sum(
            1 for s in samples 
            if np.array_equal(s.int_values[0], particles[0].int_values[0])
        )
        assert 30 < first_particle_count < 70  # With high probability
    
    def test_sampling_deterministic_replication(self):
        """Test sampling with replacement creates independent copies."""
        particle = Solution(int_values=[np.array([1, 2, 3])], fitness=1.0)
        dist = ParticleDistribution([particle], [1.0])
        
        samples = dist.sample(5)
        
        # Modify one sample
        samples[0].int_values[0][0] = 999
        
        # Others should be unchanged
        assert samples[1].int_values[0][0] == 1
        assert particle.int_values[0][0] == 1  # Original unchanged
    
    def test_get_base_structure(self):
        """Test extracting base decision variable structure."""
        particles = [
            Solution(
                int_values=[np.array([0, 1, 0])],
                dbl_values=[np.array([0.5, 0.7])],
                fitness=1.0
            )
            for _ in range(3)
        ]
        weights = np.ones(3) / 3
        
        dist = ParticleDistribution(particles, weights)
        structure = dist.get_base_structure()
        
        assert 'has_int' in structure
        assert 'has_dbl' in structure
        assert structure['has_int'] == True
        assert structure['has_dbl'] == True
        assert structure['int_shapes'] == [(3,)]
        assert structure['dbl_shapes'] == [(2,)]


class TestCompressionStrategies:
    """Test different compression strategies for particle distributions."""
    
    def test_compress_top_k_basic(self):
        """Test keeping top K particles by weight."""
        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(10)
        ]
        weights = np.array([0.3, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.005, 0.005])
        
        new_particles, new_weights = compress_top_k(particles, weights, K=5)
        
        assert len(new_particles) == 5
        assert len(new_weights) == 5
        assert np.allclose(new_weights.sum(), 1.0)
        
        # Should keep first 5 particles (highest weights)
        for i in range(5):
            assert new_particles[i].int_values[0][0] == i
    
    def test_compress_top_k_renormalization(self):
        """Test that weights are properly renormalized after compression."""
        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(10)
        ]
        weights = np.linspace(0.2, 0.02, 10)
        weights = weights / weights.sum()
        
        _, new_weights = compress_top_k(particles, weights, K=3)
        
        assert np.allclose(new_weights.sum(), 1.0)
        # New weights should maintain relative proportions
        assert new_weights[0] > new_weights[1] > new_weights[2]
    
    def test_compress_resampling(self):
        """Test resampling-based compression."""
        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(20)
        ]
        weights = np.ones(20) / 20
        
        new_particles, new_weights = compress_resampling(particles, weights, K=10)
        
        assert len(new_particles) == 10
        assert len(new_weights) == 10
        assert np.allclose(new_weights.sum(), 1.0)
        # Resampling gives uniform weights
        assert np.allclose(new_weights, 1.0/10)


class TestObjectiveFunctionals:
    """Test distributional objective functionals."""
    
    def simple_fitness(self, int_vals, dbl_vals, data=None):
        """Simple fitness function: sum of integer values."""
        if int_vals:
            return float(np.sum(int_vals[0]))
        return 0.0
    
    def test_mean_objective_basic(self):
        """Test mean objective functional."""
        # Create distribution with known mean
        p1 = Solution(int_values=[np.array([1, 0, 0])], fitness=0)
        p2 = Solution(int_values=[np.array([1, 1, 0])], fitness=0)
        p3 = Solution(int_values=[np.array([1, 1, 1])], fitness=0)
        
        particles = [p1, p2, p3]
        weights = np.array([0.5, 0.3, 0.2])
        
        dist = ParticleDistribution(particles, weights)
        
        # Expected mean: 0.5*1 + 0.3*2 + 0.2*3 = 0.5 + 0.6 + 0.6 = 1.7
        fitness = mean_objective(
            dist, 
            self.simple_fitness, 
            n_samples=1000,
            data={}
        )
        
        assert 1.5 < fitness < 1.9  # Should be close to 1.7
    
    def test_mean_variance_objective(self):
        """Test mean-variance objective functional."""
        # High variance distribution
        p1 = Solution(int_values=[np.array([0, 0, 0])], fitness=0)
        p2 = Solution(int_values=[np.array([1, 1, 1, 1, 1])], fitness=0)
        
        particles = [p1, p2]
        weights = np.array([0.5, 0.5])
        
        dist = ParticleDistribution(particles, weights)
        
        # Mean only
        fitness_mean = mean_objective(dist, self.simple_fitness, n_samples=500, data={})
        
        # Mean - variance (penalty)
        fitness_mv = mean_variance_objective(
            dist, 
            self.simple_fitness,
            n_samples=500,
            lambda_var=-1.0,  # Penalty on variance
            data={}
        )
        
        # With variance penalty, fitness should be lower
        assert fitness_mv < fitness_mean
    
    def test_cvar_objective_tail_risk(self):
        """Test CVaR objective captures tail risk."""
        # Distribution with outlier
        particles = [
            Solution(int_values=[np.array([i])], fitness=0)
            for i in [1, 2, 3, 4, 100]  # 100 is outlier
        ]
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        dist = ParticleDistribution(particles, weights)
        
        # CVaR with α=0.2 should focus on worst 20% (the 100)
        fitness_cvar_worst = cvar_objective(
            dist,
            self.simple_fitness,
            n_samples=1000,
            alpha=0.2,
            maximize=False,  # Lower is worse
            data={}
        )
        
        # Regular mean
        fitness_mean = mean_objective(dist, self.simple_fitness, n_samples=1000, data={})
        
        # CVaR should be closer to tail value
        # Mean ≈ (1+2+3+4+100)/5 = 22
        # CVaR_0.2 ≈ 100 (worst 20%)
        assert fitness_cvar_worst > fitness_mean
    
    def test_entropy_regularization(self):
        """Test entropy-regularized objective."""
        # High entropy (uniform) vs low entropy (peaked) distributions
        particles_uniform = [
            Solution(int_values=[np.array([i])], fitness=0)
            for i in range(10)
        ]
        weights_uniform = np.ones(10) / 10
        
        particles_peaked = [
            Solution(int_values=[np.array([i])], fitness=0)
            for i in range(10)
        ]
        weights_peaked = np.array([0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        
        dist_uniform = ParticleDistribution(particles_uniform, weights_uniform)
        dist_peaked = ParticleDistribution(particles_peaked, weights_peaked)
        
        # With positive entropy weight, uniform should score higher
        fitness_uniform = entropy_regularized_objective(
            dist_uniform,
            self.simple_fitness,
            n_samples=500,
            tau=0.5,  # Encourage entropy
            data={}
        )
        
        fitness_peaked = entropy_regularized_objective(
            dist_peaked,
            self.simple_fitness,
            n_samples=500,
            tau=0.5,
            data={}
        )
        
        # Uniform has higher entropy, should have higher fitness
        assert fitness_uniform > fitness_peaked


class TestDistributionalOperators:
    """Test crossover and mutation operators for distributions."""
    
    def test_crossover_mixture_basic(self):
        """Test mixture crossover of two distributions."""
        p1_particles = [
            Solution(int_values=[np.array([0, 0])], fitness=0),
            Solution(int_values=[np.array([1, 0])], fitness=1),
        ]
        p1_weights = np.array([0.6, 0.4])
        
        p2_particles = [
            Solution(int_values=[np.array([0, 1])], fitness=2),
            Solution(int_values=[np.array([1, 1])], fitness=3),
        ]
        p2_weights = np.array([0.7, 0.3])
        
        dist1 = ParticleDistribution(p1_particles, p1_weights)
        dist2 = ParticleDistribution(p2_particles, p2_weights)
        
        # Mixture with α=0.5, compress to K=3
        child_dist = crossover_particle_mixture(dist1, dist2, alpha=0.5, K_target=3)
        
        assert child_dist.K == 3
        assert len(child_dist.particles) == 3
        assert np.allclose(child_dist.weights.sum(), 1.0)
    
    def test_mutate_weights_preserves_normalization(self):
        """Test weight mutation preserves normalization."""
        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(5)
        ]
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        dist = ParticleDistribution(particles, weights)
        original_weights = dist.weights.copy()
        
        mutated_dist = mutate_weights(dist, weight_intensity=0.1)
        
        # Weights should have changed
        assert not np.allclose(mutated_dist.weights, original_weights)
        
        # But still sum to 1
        assert np.allclose(mutated_dist.weights.sum(), 1.0)
        
        # And all positive
        assert np.all(mutated_dist.weights > 0)
    
    def test_mutate_support_basic(self):
        """Test support mutation using base mutation operator."""
        particles = [
            Solution(int_values=[np.array([0, 0, 0])], fitness=0)
            for _ in range(5)
        ]
        weights = np.ones(5) / 5
        
        dist = ParticleDistribution(particles, weights)
        
        # Simple mutation function
        def simple_mutate(population, candidates, settypes, mutprob, mutintensity):
            # Just flip some bits
            for sol in population:
                if np.random.rand() < mutprob:
                    idx = np.random.randint(len(sol.int_values[0]))
                    sol.int_values[0][idx] = 1 - sol.int_values[0][idx]
        
        mutated_dist = mutate_support(
            dist,
            base_mutate_fn=simple_mutate,
            candidates=[[0, 1]],
            settypes=["BOOL"],
            support_prob=1.0,
            mutintensity=0.5
        )
        
        # At least some particles should have changed
        changed = sum(
            1 for i in range(5)
            if not np.array_equal(
                mutated_dist.particles[i].int_values[0],
                particles[i].int_values[0]
            )
        )
        assert changed > 0
    
    def test_birth_death_mutation(self):
        """Test birth-death mutation (add/remove particles)."""
        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(10)
        ]
        weights = np.ones(10) / 10
        
        dist = ParticleDistribution(particles, weights)
        
        # Birth 2, death 3 -> net -1 particle
        mutated_dist = birth_death_mutation(
            dist,
            candidates=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            setsizes=[1],
            settypes=["BOOL"],
            birth_rate=0.2,  # Add 2
            death_rate=0.3   # Remove 3
        )
        
        # Should have 10 - 3 + 2 = 9 particles
        assert mutated_dist.K == 9
        assert np.allclose(mutated_dist.weights.sum(), 1.0)


class TestDistributionalSolution:
    """Test DistributionalSolution wrapper class."""
    
    def test_creation_basic(self):
        """Test creating a DistributionalSolution."""
        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(5)
        ]
        weights = np.ones(5) / 5
        
        dist = ParticleDistribution(particles, weights)
        dsol = DistributionalSolution(dist)
        
        assert isinstance(dsol, DistributionalSolution)
        assert dsol.distribution is dist
        assert dsol.fitness == float('-inf')  # Not yet evaluated
    
    def test_copy_creates_independent(self):
        """Test that copy creates independent DistributionalSolution."""
        particles = [Solution(int_values=[np.array([i])]) for i in range(3)]
        weights = np.ones(3) / 3
        dist = ParticleDistribution(particles, weights)
        
        dsol1 = DistributionalSolution(dist, fitness=10.0)
        dsol2 = dsol1.copy()
        
        # Modify dsol2
        dsol2.fitness = 20.0
        dsol2.distribution.weights[0] = 0.9
        dsol2.distribution.weights = dsol2.distribution.weights / dsol2.distribution.weights.sum()
        
        # dsol1 should be unchanged
        assert dsol1.fitness == 10.0
        assert not np.allclose(dsol1.distribution.weights, dsol2.distribution.weights)


class TestIntegrationWithExistingTypes:
    """Test that distributional head works with existing decision types."""
    
    def test_dist_wraps_bool(self):
        """Test particle distribution wrapping BOOL decision variables."""
        # Create particles from BOOL solutions
        candidates = [list(range(5))]
        setsizes = [5]
        settypes = ["BOOL"]
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size=10)
        
        # Create distribution from population
        weights = np.ones(10) / 10
        dist = ParticleDistribution(pop, weights)
        
        assert dist.K == 10
        structure = dist.get_base_structure()
        assert structure['has_int'] == True
    
    def test_dist_wraps_continuous(self):
        """Test particle distribution wrapping DBL decision variables."""
        candidates = [[]]
        setsizes = [3]
        settypes = ["DBL"]
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size=5)
        
        weights = np.ones(5) / 5
        dist = ParticleDistribution(pop, weights)
        
        assert dist.K == 5
        structure = dist.get_base_structure()
        assert structure['has_dbl'] == True
    
    def test_dist_wraps_mixed(self):
        """Test particle distribution wrapping mixed INT+DBL+BOOL."""
        candidates = [[0, 100], [], [0, 1, 2, 3, 4]]
        setsizes = [5, 3, 5]
        settypes = ["INT", "DBL", "BOOL"]
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size=8)
        
        weights = np.ones(8) / 8
        dist = ParticleDistribution(pop, weights)
        
        assert dist.K == 8
        structure = dist.get_base_structure()
        assert structure['has_int'] == True
        assert structure['has_dbl'] == True


class TestDistributionalMultiobjective:
    """Test distributional optimization with multi-objective scenarios."""
    
    def simple_fitness(self, int_vals, dbl_vals, data=None):
        """Simple fitness: sum of values."""
        if int_vals:
            return float(np.sum(int_vals[0]))
        return 0.0
    
    def two_objective_fitness(self, int_vals, dbl_vals, data=None):
        """Return a two-objective vector (sum, negative sum) for trade-offs."""
        if isinstance(int_vals, (list, tuple)) and len(int_vals) > 0:
            core = int_vals[0]
        else:
            core = int_vals
        score = float(np.sum(core)) if core is not None else 0.0
        return [score, -score]
    
    def test_multi_fitness_on_dist_solution(self):
        """Test that DistributionalSolution supports multi-fitness."""
        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(5)
        ]
        weights = np.ones(5) / 5
        dist = ParticleDistribution(particles, weights)
        
        # Create distributional solution with multi-fitness
        dsol = DistributionalSolution(dist, fitness=10.0, multi_fitness=[5.0, 5.0])
        
        assert dsol.multi_fitness == [5.0, 5.0]
        assert dsol.fitness == 10.0
        
        # Test copy preserves multi_fitness
        dsol2 = dsol.copy()
        assert dsol2.multi_fitness == [5.0, 5.0]
        dsol2.multi_fitness[0] = 3.0
        assert dsol.multi_fitness[0] == 5.0  # Original unchanged
    
    def test_mean_variance_as_multiobjective(self):
        """Test using mean and variance as separate objectives."""
        # Create distributions with different mean-variance tradeoffs
        
        # Low mean, low variance
        p1_particles = [Solution(int_values=[np.array([1, 1])]) for _ in range(5)]
        p1_dist = ParticleDistribution(p1_particles, np.ones(5) / 5)
        
        # High mean, low variance  
        p2_particles = [Solution(int_values=[np.array([5, 5])]) for _ in range(5)]
        p2_dist = ParticleDistribution(p2_particles, np.ones(5) / 5)
        
        # Medium mean, high variance
        p3_particles = [
            Solution(int_values=[np.array([0, 0])]),
            Solution(int_values=[np.array([6, 6])]),
            Solution(int_values=[np.array([0, 0])]),
            Solution(int_values=[np.array([6, 6])]),
            Solution(int_values=[np.array([3, 3])])
        ]
        p3_dist = ParticleDistribution(p3_particles, np.ones(5) / 5)
        
        # Evaluate mean and variance separately
        def get_mean_var(dist):
            samples = dist.sample(100)
            fitness_values = [self.simple_fitness(s.int_values, s.dbl_values) for s in samples]
            return float(np.mean(fitness_values)), float(np.var(fitness_values))
        
        mean1, var1 = get_mean_var(p1_dist)
        mean2, var2 = get_mean_var(p2_dist)
        mean3, var3 = get_mean_var(p3_dist)
        
        # Verify expected relationships
        assert mean1 < mean2  # p2 has higher mean
        assert var1 < var3  # p3 has higher variance
        assert mean3 > mean1 and mean3 < mean2  # p3 is in between
    
    def test_distributional_multiobjective_weighted_mean(self):
        """Distributional solutions should compute multi-fitness per objective."""
        if not HAS_DIST_HEAD:
            pytest.skip("Distributional head not available")
        
        # Two particles with different sums
        p1 = Solution(int_values=[np.array([3, 1])])
        p2 = Solution(int_values=[np.array([2, 0])])
        weights = np.array([0.25, 0.75])
        dist = ParticleDistribution([p1, p2], weights)
        dsol = DistributionalSolution(dist)
        
        population = [dsol]
        control = {"dist_objective": "mean", "dist_eval_mode": "weighted"}
        
        evaluate_fitness(
            population=population,
            stat_func=self.two_objective_fitness,
            data={},
            n_stat=2,
            control=control
        )
        
        expected = [2.5, -2.5]  # Weighted mean of (sum, -sum)
        assert np.allclose(population[0].multi_fitness, expected)
        assert population[0].fitness == pytest.approx(sum(expected))
    
    def test_distributional_non_dominated_sorting(self):
        """Distributional multi-objective solutions should be non-dominated when appropriate."""
        if not HAS_DIST_HEAD:
            pytest.skip("Distributional head not available")
        
        # Solution A: higher first objective, lower second
        a_particles = [
            Solution(int_values=[np.array([4, 2])]),
            Solution(int_values=[np.array([2, 2])])
        ]
        a_dist = ParticleDistribution(a_particles, np.array([0.5, 0.5]))
        a_sol = DistributionalSolution(a_dist)
        
        # Solution B: lower first objective, higher second
        b_particles = [
            Solution(int_values=[np.array([1, 1])]),
            Solution(int_values=[np.array([3, 0])])
        ]
        b_dist = ParticleDistribution(b_particles, np.array([0.5, 0.5]))
        b_sol = DistributionalSolution(b_dist)
        
        population = [a_sol, b_sol]
        control = {"dist_objective": "mean", "dist_eval_mode": "weighted"}
        
        evaluate_fitness(
            population=population,
            stat_func=self.two_objective_fitness,
            data={},
            n_stat=2,
            control=control
        )
        
        # Verify the objectives are in conflict so neither dominates the other
        fronts = fast_non_dominated_sort(population)
        assert len(fronts) > 0
        assert len(fronts[0]) == 2  # both should appear on the first Pareto front

    def test_control_supports_nsga_means_flag(self):
        """evolve_control should accept dist_use_nsga_means."""
        control = evolve_control(dist_use_nsga_means=True)
        assert control["dist_use_nsga_means"] is True

    def test_distributional_nsga_on_mean_objectives_scalar_setup(self):
        """
        When dist_use_nsga_means is enabled, distributional solutions should
        expose their mean objectives for NSGA selection even if n_stat=1.
        """
        if not HAS_DIST_HEAD:
            pytest.skip("Distributional head not available")

        # Solution A: higher first objective, lower second
        a_particles = [
            Solution(int_values=[np.array([4, 2])]),
            Solution(int_values=[np.array([2, 2])])
        ]
        a_dist = ParticleDistribution(a_particles, np.array([0.5, 0.5]))
        a_sol = DistributionalSolution(a_dist)

        # Solution B: lower first objective, higher second
        b_particles = [
            Solution(int_values=[np.array([1, 1])]),
            Solution(int_values=[np.array([3, 0])])
        ]
        b_dist = ParticleDistribution(b_particles, np.array([0.5, 0.5]))
        b_sol = DistributionalSolution(b_dist)

        population = [a_sol, b_sol]
        control = {
            "dist_objective": "mean",
            "dist_eval_mode": "weighted",
            "dist_use_nsga_means": True,
        }

        evaluate_fitness(
            population=population,
            stat_func=self.two_objective_fitness,
            data={},
            n_stat=1,  # simulate scalarized setup
            control=control
        )

        assert population[0].multi_fitness == pytest.approx([5.0, -5.0])
        assert population[1].multi_fitness == pytest.approx([2.5, -2.5])

        # Selection should treat this as multi-objective even though n_stat=1
        selected = _select_next_generation(
            combined_pop=population,
            pop_size=2,
            n_stat=1,
            use_nsga3=False,
            reference_points=None,
            candidates=None,
            setsizes=None,
            settypes=None,
            stat_func=None,
            data={},
            is_parallel=False,
            control=control,
            fitness_cache={},
            surrogate_model=None,
        )

        fronts = fast_non_dominated_sort(selected)
        assert len(selected) == 2
        assert len(fronts) > 0 and len(fronts[0]) == 2
    
    def test_distributional_crossover_preserves_structure(self):
        """Test that crossover works with distributions that have multi_fitness."""
        particles1 = [Solution(int_values=[np.array([i])]) for i in range(3)]
        particles2 = [Solution(int_values=[np.array([i+5])]) for i in range(3)]
        
        dist1 = ParticleDistribution(particles1, np.ones(3) / 3)
        dist2 = ParticleDistribution(particles2, np.ones(3) / 3)
        
        # Create DistributionalSolutions with multi_fitness
        dsol1 = DistributionalSolution(dist1, fitness=10.0, multi_fitness=[5.0, 5.0])
        dsol2 = DistributionalSolution(dist2, fitness=12.0, multi_fitness=[6.0, 6.0])
        
        # Crossover at distribution level
        child_dist = crossover_particle_mixture(dist1, dist2, alpha=0.5, K_target=3)
        
        # Child should have expected structure
        assert child_dist.K == 3
        assert len(child_dist.particles) == 3
        assert np.allclose(child_dist.weights.sum(), 1.0)

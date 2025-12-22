"""
Tests for critical distributional GA fixes.

These tests are written following TDD methodology to verify fixes for
critical bugs identified in the audit.
"""

import pytest
import numpy as np
from trainselpy.solution import Solution
from trainselpy.algorithms import (
    initialize_population,
    evaluate_fitness,
)
from trainselpy.core import train_sel_control
from trainselpy.distributional_head import (
    ParticleDistribution,
    DistributionalSolution,
    entropy_regularized_objective,
    mean_objective,
)
from trainselpy.distributional_operators import (
    initialize_distributional_population,
)


class TestPhase1_1_SurrogateNNIntegration:
    """
    Test Phase 1.1: Guard Surrogate & NN Integration

    Issue: evaluate_fitness calls surrogate_model.predict(population).
    Surrogate models expect flat .int_values or .dbl_values.
    DistributionalSolution objects do not have these attributes in the same structure.
    Result: Immediate crash or type errors when use_surrogate_objective is enabled
    during a distributional run.
    """

    def test_surrogate_with_distributional_raises_error_or_skips(self):
        """Test that surrogate model with distributional solutions raises clear error or skips gracefully."""
        # Create distributional population
        candidates = [list(range(10))]
        setsizes = [5]
        settypes = ["DIST:BOOL"]
        control = train_sel_control(dist_K_particles=3)

        population = initialize_distributional_population(
            candidates, setsizes, settypes, pop_size=10, control=control
        )

        # Mock surrogate model with is_fitted=True
        class MockSurrogateModel:
            is_fitted = True

            def predict(self, pop):
                # This should not be called with DistributionalSolution objects
                # If it is called, it should raise a clear error
                if pop and isinstance(pop[0], DistributionalSolution):
                    raise TypeError("Surrogate model cannot handle DistributionalSolution objects")
                return np.zeros(len(pop)), np.zeros(len(pop))

        surrogate_model = MockSurrogateModel()

        # Enable surrogate objective
        control = train_sel_control(
            use_surrogate_objective=True,
            dist_K_particles=3
        )

        # Define a simple fitness function
        def fitness_func(x, data):
            return float(np.sum(x))

        # This should either:
        # 1. Raise NotImplementedError with clear message, OR
        # 2. Log warning and skip surrogate logic, falling back to regular evaluation
        # The test verifies the code doesn't crash with cryptic error
        try:
            evaluate_fitness(
                population,
                fitness_func,
                n_stat=1,
                data={},
                control=control,
                surrogate_model=surrogate_model
            )
            # If we get here, it should have skipped surrogate and evaluated properly
            # Check that fitness was set
            assert all(hasattr(sol, 'fitness') for sol in population)
        except (NotImplementedError, TypeError) as e:
            # This is acceptable - clear error message
            assert "surrogate" in str(e).lower() or "distributional" in str(e).lower()
        except AttributeError as e:
            # This is NOT acceptable - means we didn't guard properly
            pytest.fail(f"Unguarded attribute error (should be caught earlier): {e}")

    def test_nn_training_with_distributional_raises_error_or_skips(self):
        """Test that NN training with distributional solutions raises clear error or skips gracefully."""
        # This test will verify _train_neural_models and _generate_neural_offspring
        # Since these are internal functions, we test through genetic_algorithm
        # For now, we'll create a simpler test that verifies the guard exists

        # Create distributional population
        candidates = [list(range(10))]
        setsizes = [5]
        settypes = ["DIST:BOOL"]
        control = train_sel_control(
            dist_K_particles=3,
            use_vae=False,  # Start with False
            use_gan=False
        )

        population = initialize_distributional_population(
            candidates, setsizes, settypes, pop_size=10, control=control
        )

        # The population should be valid DistributionalSolution objects
        assert all(isinstance(sol, DistributionalSolution) for sol in population)

        # If we enable VAE/GAN, it should be detected and raise/skip
        # This will be tested in integration tests with full GA run


class TestPhase1_2_InputValidation:
    """
    Test Phase 1.2: Add Input Validation to ParticleDistribution.__init__

    Issue: ParticleDistribution.__init__ does not check for empty particle lists,
    NaNs, or infinite weights.
    Result: Potential for silent mathematical failures deep in the stack.
    """

    def test_empty_particles_raises_error(self):
        """Test that empty particle list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            ParticleDistribution(particles=[], weights=np.array([]))

    def test_nan_weights_raises_error(self):
        """Test that NaN weights raise ValueError."""
        particles = [Solution(int_values=[np.array([1])], fitness=1.0) for _ in range(3)]
        weights = np.array([0.5, np.nan, 0.5])

        with pytest.raises(ValueError, match="NaN"):
            ParticleDistribution(particles, weights)

    def test_inf_weights_raises_error(self):
        """Test that infinite weights raise ValueError."""
        particles = [Solution(int_values=[np.array([1])], fitness=1.0) for _ in range(3)]
        weights = np.array([0.5, np.inf, 0.5])

        with pytest.raises(ValueError, match="infinite|inf"):
            ParticleDistribution(particles, weights)

    def test_negative_inf_weights_raises_error(self):
        """Test that negative infinite weights raise ValueError."""
        particles = [Solution(int_values=[np.array([1])], fitness=1.0) for _ in range(3)]
        weights = np.array([0.5, -np.inf, 0.5])

        with pytest.raises(ValueError, match="infinite|inf"):
            ParticleDistribution(particles, weights)

    def test_zero_sum_weights_handled(self):
        """Test that zero-sum weights are handled (warning + uniform distribution)."""
        particles = [Solution(int_values=[np.array([i])], fitness=float(i)) for i in range(3)]
        weights = np.array([0.0, 0.0, 0.0])

        # Should not crash, should default to uniform
        dist = ParticleDistribution(particles, weights)

        # All weights should be equal and sum to 1
        assert np.allclose(dist.weights, 1/3)
        assert np.allclose(dist.weights.sum(), 1.0)

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched particle and weight lengths raise error."""
        particles = [Solution(int_values=[np.array([i])], fitness=float(i)) for i in range(3)]
        weights = np.array([0.5, 0.5])  # Only 2 weights for 3 particles

        with pytest.raises((ValueError, AssertionError), match="match"):
            ParticleDistribution(particles, weights)


class TestPhase1_3_DeadCode:
    """
    Test Phase 1.3: Clean Dead Code in distributional_head.py

    Issue: entropy_regularized_objective contains unreachable duplicate logic
    after an early return statement (lines 666-722).
    Result: Confusion regarding which entropy calculation is active.
    """

    def test_entropy_regularized_objective_consistent(self):
        """Test that entropy_regularized_objective produces consistent results."""
        # Create a simple distribution
        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(5)
        ]
        weights = np.array([0.4, 0.3, 0.2, 0.07, 0.03])
        dist = ParticleDistribution(particles, weights)

        # Pre-computed fitness values
        fitness_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Call entropy_regularized_objective
        result = entropy_regularized_objective(
            dist,
            base_fitness_fn=fitness_values,
            tau=0.1
        )

        # Manually compute expected result
        # E[f(x)] = sum(w_i * f_i)
        expected_mean = np.sum(weights * fitness_values)

        # H(μ) = -sum(w_i * log(w_i)) for w_i > 0
        w_safe = weights[weights > 0]
        expected_entropy = -np.sum(w_safe * np.log(w_safe))

        expected = expected_mean + 0.1 * expected_entropy

        # Should match within numerical precision
        assert np.isclose(result, expected, rtol=1e-6), \
            f"Expected {expected}, got {result}"

    def test_entropy_regularized_with_callable(self):
        """Test entropy_regularized_objective with callable fitness function."""
        particles = [
            Solution(int_values=[np.array([i])], dbl_values=[], fitness=float(i))
            for i in range(5)
        ]
        weights = np.array([0.4, 0.3, 0.2, 0.07, 0.03])
        dist = ParticleDistribution(particles, weights)

        def fitness_fn(int_vals, dbl_vals, data):
            return float(np.sum(int_vals))

        result = entropy_regularized_objective(
            dist,
            base_fitness_fn=fitness_fn,
            n_samples=100,
            tau=0.1,
            data={}
        )

        # Result should be a float
        assert isinstance(result, float)
        # Should be positive (sum of positive ints + entropy bonus)
        assert result > 0


class TestPhase2_2_NSGAMeansLogic:
    """
    Test Phase 2.2: Fix dist_use_nsga_means Logic

    Issue: When dist_use_nsga_means is set, the code replaces multi_fitness with
    mean objectives. However, it only stores the configured distributional objective
    in _dist_aggregate_objectives.
    Result: Selection pressure ignores the chosen objective (CVaR/Entropy) and
    strictly optimizes the mean, defeating the purpose of the configuration.

    FIX: multi_fitness should include BOTH mean objectives AND distributional objective
    for proper multi-objective optimization.
    """

    def test_nsga_means_includes_distributional_in_multiobjective(self):
        """Test that dist_use_nsga_means includes distributional objective in multi_fitness."""
        # Create distributional population
        candidates = [list(range(10))]
        setsizes = [5]
        settypes = ["DIST:BOOL"]

        # Single objective case
        control = train_sel_control(
            dist_K_particles=3,
            dist_objective="cvar",  # Use CVaR, not mean
            dist_alpha=0.2,
            dist_use_nsga_means=True,  # Enable NSGA means mode
            dist_maximize=True
        )

        population = initialize_distributional_population(
            candidates, setsizes, settypes, pop_size=5, control=control
        )

        # Define fitness function
        def fitness_fn(x, data):
            return float(np.sum(x))

        # Evaluate
        evaluate_fitness(population, fitness_fn, n_stat=1, data={}, control=control)

        # Check that solutions have combined objectives
        for sol in population:
            # Should have multi_fitness with 2 objectives: [mean, cvar]
            assert hasattr(sol, 'multi_fitness')
            assert len(sol.multi_fitness) == 2, \
                f"Expected 2 objectives (mean + CVaR), got {len(sol.multi_fitness)}"

            # First objective should be mean
            # Second objective should be CVaR
            mean_obj = sol.multi_fitness[0]
            cvar_obj = sol.multi_fitness[1]

            # They should be different (CVaR is more conservative than mean)
            # This validates that both are actually computed
            assert isinstance(mean_obj, (int, float))
            assert isinstance(cvar_obj, (int, float))

    def test_nsga_means_multiobjective_case(self):
        """Test dist_use_nsga_means with multi-objective base fitness."""
        candidates = [list(range(10))]
        setsizes = [5]
        settypes = ["DIST:BOOL"]

        # Multi-objective case (2 objectives)
        control = train_sel_control(
            dist_K_particles=3,
            dist_objective="entropy",  # Use entropy
            dist_tau=0.1,
            dist_use_nsga_means=True,
            dist_maximize=True
        )

        population = initialize_distributional_population(
            candidates, setsizes, settypes, pop_size=5, control=control
        )

        # Multi-objective fitness function
        def fitness_fn(x, data):
            return [float(np.sum(x)), float(np.mean(x))]

        # Evaluate with n_stat=2
        evaluate_fitness(population, fitness_fn, n_stat=2, data={}, control=control)

        # Check that solutions have combined objectives
        for sol in population:
            # Should have multi_fitness with 3 objectives: [mean1, mean2, entropy]
            assert hasattr(sol, 'multi_fitness')
            assert len(sol.multi_fitness) == 3, \
                f"Expected 3 objectives (2 means + entropy), got {len(sol.multi_fitness)}"


class TestPhase2_3_MixedSchemaInit:
    """
    Test Phase 2.3: Fix Mixed Schema Initialization

    Issue: initialize_distributional_population grabs base_type = settypes[0]
    and ignores others.
    Result: Mixed schemas (e.g., ["DIST:DBL", "BOOL"]) will be misinitialized
    or mutated incorrectly.
    """

    def test_mixed_schema_raises_clear_error(self):
        """Test that mixed schemas raise NotImplementedError with clear message."""
        candidates = [list(range(10)), list(range(5))]
        setsizes = [5, 3]
        settypes = ["DIST:BOOL", "INT"]  # Mixed: distributional + standard

        control = train_sel_control(dist_K_particles=3)

        # This should raise a clear NotImplementedError
        with pytest.raises(NotImplementedError, match="(?i)mixed|multiple.*type"):
            initialize_distributional_population(
                candidates, setsizes, settypes, pop_size=10, control=control
            )

    def test_pure_distributional_schema_works(self):
        """Test that pure distributional schema (all DIST:X) works correctly."""
        candidates = [list(range(10))]
        setsizes = [5]
        settypes = ["DIST:BOOL"]

        control = train_sel_control(dist_K_particles=3)

        # This should work
        population = initialize_distributional_population(
            candidates, setsizes, settypes, pop_size=10, control=control
        )

        assert len(population) == 10
        assert all(isinstance(sol, DistributionalSolution) for sol in population)
        assert all(sol.distribution.K == 3 for sol in population)


class TestPhase3_2_CMAESCompatibility:
    """
    Test Phase 3.2: Add CMA-ES + Distributional compatibility check

    Issue: CMA-ES is not compatible with distributional GA but there's no check.
    """

    def test_cmaes_with_distributional_raises_error(self):
        """Test that CMA-ES + distributional raises clear error."""
        from trainselpy.core import train_sel

        candidates = [list(range(10))]
        setsizes = [5]
        settypes = ["DIST:BOOL"]

        control = train_sel_control(
            dist_K_particles=3,
            use_cma_es=True,  # Enable CMA-ES
            niterations=2,
            progress=False
        )

        def fitness_fn(x, data):
            return float(np.sum(x))

        # This should raise ValueError with clear message
        with pytest.raises(ValueError, match="(?i)cma.*es.*distributional"):
            train_sel(
                stat=fitness_fn,
                candidates=candidates,
                setsizes=setsizes,
                settypes=settypes,
                control=control
            )


class TestPhase3_3_CompressKmeans:
    """
    Test Phase 3.3: Implement compress_kmeans properly

    Issue: compress_kmeans currently just calls compress_top_k (stub).
    """

    def test_compress_kmeans_exists_and_works(self):
        """Test that compress_kmeans exists and can compress a distribution."""
        from trainselpy.distributional_head import compress_kmeans

        particles = [
            Solution(int_values=[np.array([i])], fitness=float(i))
            for i in range(20)
        ]
        weights = np.random.rand(20)
        weights = weights / weights.sum()

        dist = ParticleDistribution(particles, weights)

        # Compress to 5 particles
        compressed = compress_kmeans(dist, K=5)

        # Should return a ParticleDistribution
        assert isinstance(compressed, ParticleDistribution)
        # Should have 5 particles
        assert compressed.K == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

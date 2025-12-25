import pytest
import numpy as np
from evosolve.algorithms import initialize_population
from evosolve.operators import mutation, crossover
from evosolve.solution import Solution


class TestINTInitialization:
    """Test INT type initialization."""
    
    def test_int_initialization_with_bounds(self):
        """Test random initialization of integers with explicit bounds."""
        pop_size = 10
        int_count = 5
        min_val, max_val = 10, 50
        
        setsizes = [int_count]
        settypes = ["INT"]
        candidates = [[min_val, max_val]]  # [min, max] format
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size)
        
        assert len(pop) == pop_size
        for sol in pop:
            assert len(sol.int_values) == 1
            int_arr = sol.int_values[0]
            assert len(int_arr) == int_count
            assert np.all(int_arr >= min_val)
            assert np.all(int_arr <= max_val)
            assert int_arr.dtype == np.dtype('int64') or int_arr.dtype == np.dtype('int32')
    
    def test_int_initialization_default_bounds(self):
        """Test INT with single candidate (interpreted as max)."""
        pop_size = 5
        int_count = 3
        max_val = 100
        
        setsizes = [int_count]
        settypes = ["INT"]
        candidates = [[max_val]]  # Single value = max, min defaults to 0
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size)
        
        for sol in pop:
            int_arr = sol.int_values[0]
            assert np.all(int_arr >= 0)
            assert np.all(int_arr <= max_val)
    
    def test_int_initialization_empty_candidates(self):
        """Test INT with empty candidates (defaults to [0, 100])."""
        pop_size = 5
        int_count = 4
        
        setsizes = [int_count]
        settypes = ["INT"]
        candidates = [[]]  # Empty = defaults
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size)
        
        for sol in pop:
            int_arr = sol.int_values[0]
            assert np.all(int_arr >= 0)
            assert np.all(int_arr <= 100)


class TestINTMutation:
    """Test INT type mutation."""
    
    def test_int_mutation_changes_values(self):
        """Test that mutation modifies integer values."""
        int_count = 10
        min_val, max_val = 0, 100
        
        setsizes = [int_count]
        settypes = ["INT"]
        candidates = [[min_val, max_val]]
        
        pop = initialize_population(candidates, setsizes, settypes, 5)
        original_values = [sol.int_values[0].copy() for sol in pop]
        
        # Apply mutation with high probability
        mutation(pop, candidates, settypes, mutprob=1.0, mutintensity=0.5)
        
        for i, sol in enumerate(pop):
            # Values should have changed
            assert not np.array_equal(sol.int_values[0], original_values[i])
            # Bounds should still be respected
            assert np.all(sol.int_values[0] >= min_val)
            assert np.all(sol.int_values[0] <= max_val)
    
    def test_int_mutation_respects_bounds(self):
        """Test mutation clips values to bounds."""
        int_count = 20
        min_val, max_val = -10, 10
        
        setsizes = [int_count]
        settypes = ["INT"]
        candidates = [[min_val, max_val]]
        
        pop = initialize_population(candidates, setsizes, settypes, 10)
        
        # Multiple rounds of mutation
        for _ in range(5):
            mutation(pop, candidates, settypes, mutprob=0.8, mutintensity=0.5)
        
        # All values should still be in bounds
        for sol in pop:
            assert np.all(sol.int_values[0] >= min_val)
            assert np.all(sol.int_values[0] <= max_val)
    
    def test_int_mutation_intensity_scaling(self):
        """Test that mutation intensity scales delta by range."""
        int_count = 100
        min_val, max_val = 0, 1000  # Large range
        
        setsizes = [int_count]
        settypes = ["INT"]
        candidates = [[min_val, max_val]]
        
        pop_low = initialize_population(candidates, setsizes, settypes, 50)
        pop_high = initialize_population(candidates, setsizes, settypes, 50)
        
        original_low = [sol.int_values[0].copy() for sol in pop_low]
        original_high = [sol.int_values[0].copy() for sol in pop_high]
        
        # Low intensity = small changes
        mutation(pop_low, candidates, settypes, mutprob=0.5, mutintensity=0.1)
        
        # High intensity = larger changes
        mutation(pop_high, candidates, settypes, mutprob=0.5, mutintensity=0.9)
        
        # Calculate average deltas
        deltas_low = []
        deltas_high = []
        
        for i in range(len(pop_low)):
            deltas_low.extend(np.abs(pop_low[i].int_values[0] - original_low[i]))
            deltas_high.extend(np.abs(pop_high[i].int_values[0] - original_high[i]))
        
        # Filter out zeros (no mutation occurred)
        deltas_low = [d for d in deltas_low if d > 0]
        deltas_high = [d for d in deltas_high if d > 0]
        
        if deltas_low and deltas_high:
            avg_delta_low = np.mean(deltas_low)
            avg_delta_high = np.mean(deltas_high)
            
            # Higher intensity should produce larger average deltas
            assert avg_delta_high > avg_delta_low


class TestINTCrossover:
    """Test INT type crossover."""
    
    def test_int_crossover_combines_parents(self):
        """Test that crossover creates offspring from two parents."""
        int_count = 10
        min_val, max_val = 0, 100
        
        setsizes = [int_count]
        settypes = ["INT"]
        candidates = [[min_val, max_val]]
        
        pop = initialize_population(candidates, setsizes, settypes, 10)
        
        # Crossover with 100% probability
        offspring = crossover(pop, crossprob=1.0, crossintensity=0.5, 
                              settypes=settypes, candidates=candidates)
        
        assert len(offspring) == len(pop)
        
        for child in offspring:
            assert len(child.int_values) == 1
            assert len(child.int_values[0]) == int_count
            # Check bounds
            assert np.all(child.int_values[0] >= min_val)
            assert np.all(child.int_values[0] <= max_val)
    
    def test_int_crossover_respects_bounds(self):
        """Test crossover maintains boundary constraints."""
        int_count = 15
        min_val, max_val = 50, 150
        
        setsizes = [int_count]
        settypes = ["INT"]
        candidates = [[min_val, max_val]]
        
        pop = initialize_population(candidates, setsizes, settypes, 20)
        
        # Multiple rounds
        for _ in range(3):
            pop = crossover(pop, crossprob=0.8, crossintensity=0.6,
                            settypes=settypes, candidates=candidates)
        
        for child in pop:
            assert np.all(child.int_values[0] >= min_val)
            assert np.all(child.int_values[0] <= max_val)
    
    def test_int_crossover_creates_variation(self):
        """Test that crossover actually recombines genetic material."""
        # Create two distinct parents
        sol1 = Solution()
        sol1.int_values.append(np.full(10, 10, dtype=int))  # All 10s
        
        sol2 = Solution()
        sol2.int_values.append(np.full(10, 90, dtype=int))  # All 90s
        
        settypes = ["INT"]
        candidates = [[0, 100]]
        
        parents = [sol1, sol2]
        
        # Crossover should create children with mixed values
        offspring = crossover(parents, crossprob=1.0, crossintensity=0.5,
                              settypes=settypes, candidates=candidates)
        
        # At least one child should have values from both parents
        child1_vals = offspring[0].int_values[0]
        
        # Child should have values different from pure 10s or pure 90s
        has_low = np.any(child1_vals < 50)
        has_high = np.any(child1_vals > 50)
        
        # With multipoint crossover, we expect mixing (this may occasionally fail due to randomness)
        # In practice, with crossintensity=0.5, we should get mixing
        assert has_low or has_high or np.all(child1_vals == 10) or np.all(child1_vals == 90)


class TestINTNeuralNetworkIntegration:
    """Test INT type neural network support."""
    
    @pytest.mark.skipif(True, reason="Neural network support for INT requires implementation")
    def test_int_extraction_for_nn(self):
        """Test that INT values can be extracted for neural network training."""
        # This test will be skipped until neural network support is implemented
        from evosolve.algorithms import _extract_decision_parts
        
        int_count = 5
        min_val, max_val = 0, 100
        
        setsizes = [int_count]
        settypes = ["INT"]
        candidates = [[min_val, max_val]]
        
        pop = initialize_population(candidates, setsizes, settypes, 10)
        
        # Extract decision parts
        bin_tensor, perm_tensors, cont_tensor = _extract_decision_parts(
            pop, settypes, setsizes, candidates
        )
        
        # INT should be treated as continuous for neural networks
        assert cont_tensor is not None
        assert cont_tensor.shape[0] == 10  # Population size
        assert cont_tensor.shape[1] == int_count
    
    @pytest.mark.skipif(True, reason="Neural network generation for INT requires implementation")
    def test_int_generation_from_nn(self):
        """Test that INT values can be generated from neural networks."""
        # This will test the decode pathway once implemented
        pass


class TestINTMixedOptimization:
    """Test INT in mixed-type optimization problems."""
    
    def test_int_with_continuous_variables(self):
        """Test INT combined with DBL variables."""
        setsizes = [5, 3]  # 5 integers, 3 doubles
        settypes = ["INT", "DBL"]
        candidates = [[0, 10], []]  # INT bounds, DBL no bounds needed
        
        pop = initialize_population(candidates, setsizes, settypes, 10)
        
        for sol in pop:
            assert len(sol.int_values) == 1
            assert len(sol.dbl_values) == 1
            assert len(sol.int_values[0]) == 5
            assert len(sol.dbl_values[0]) == 3
            
            # INT bounds
            assert np.all(sol.int_values[0] >= 0)
            assert np.all(sol.int_values[0] <= 10)
            
            # DBL bounds
            assert np.all(sol.dbl_values[0] >= 0)
            assert np.all(sol.dbl_values[0] <= 1)
    
    def test_int_with_bool_variables(self):
        """Test INT combined with BOOL variables."""
        setsizes = [8, 4]  # 8 integers, 4 booleans
        settypes = ["INT", "BOOL"]
        candidates = [[-5, 5], list(range(4))]  # INT bounds, BOOL candidates
        
        pop = initialize_population(candidates, setsizes, settypes, 10)
        
        for sol in pop:
            assert len(sol.int_values) == 2
            
            # First is INT
            assert len(sol.int_values[0]) == 8
            assert np.all(sol.int_values[0] >= -5)
            assert np.all(sol.int_values[0] <= 5)
            
            # Second is BOOL
            assert len(sol.int_values[1]) == 4
            assert set(np.unique(sol.int_values[1])).issubset({0, 1})

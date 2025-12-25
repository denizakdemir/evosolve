"""
Comprehensive integration tests for advanced optimization heads.

Tests full end-to-end workflows with evolve(), including:
- Single and multi-objective optimization
- Neural enhancement (VAE/GAN)
- Mixed-type optimization
- Realistic application scenarios
"""

import pytest
import numpy as np
from evosolve.core import evolve


class TestAdvancedHeadsIntegration:
    """End-to-end integration tests for advanced heads."""
    
    def test_graph_w_full_workflow(self):
        """Test complete GRAPH_W optimization workflow."""
        n_nodes = 3
        setsizes = [n_nodes * n_nodes]
        settypes = ["GRAPH_W"]
        candidates = [list(range(n_nodes))]
        
        def graph_fitness(dbl_vals, data):
            """Simple fitness: sum of edges."""
            return float(np.sum(dbl_vals))
        
        # Single-objective
        result = evolve(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=graph_fitness,
            n_stat=1,
            control={"generations": 10, "popsize": 15, "progress": False}
        )
        
        assert result is not None
        assert result.fitness >= 0
        
        # Verify constraint satisfaction
        graph = result.selected_values[0]
        assert np.all(graph >= 0) and np.all(graph <= 1)
    
    def test_spd_full_workflow(self):
        """Test complete SPD optimization workflow."""
        n = 2
        setsizes = [n * n]
        settypes = ["SPD"]
        candidates = [[]]
        
        def spd_fitness(dbl_vals, data):
            """Maximize determinant."""
            mat = dbl_vals.reshape(n, n)
            return float(np.linalg.det(mat))
        
        result = evolve(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=spd_fitness,
            n_stat=1,
            control={"generations": 10, "popsize": 15, "progress": False}
        )
        
        assert result is not None
        assert result.fitness > 0
        
        # Verify SPD constraints
        mat = result.selected_values[0].reshape(n, n)
        assert np.allclose(mat, mat.T, atol=1e-5)
        eigvals = np.linalg.eigvalsh(mat)
        assert np.all(eigvals > 0)
    
    def test_simplex_full_workflow(self):
        """Test complete SIMPLEX optimization workflow."""
        dim = 4
        setsizes = [dim]
        settypes = ["SIMPLEX"]
        candidates = [[]]
        
        def simplex_fitness(dbl_vals, data):
            """Maximize entropy."""
            vec = dbl_vals
            eps = 1e-10
            return -float(np.sum(vec * np.log(vec + eps)))
        
        result = evolve(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=simplex_fitness,
            n_stat=1,
            control={"generations": 10, "popsize": 15, "progress": False}
        )
        
        assert result is not None
        
        # Verify simplex constraints
        vec = result.selected_values[0]
        assert np.all(vec >= -1e-6)
        assert np.abs(np.sum(vec) - 1.0) < 1e-5
    
    def test_mixed_types_with_advanced_heads(self):
        """Test mixed optimization with advanced heads + legacy types."""
        n_nodes = 3
        setsizes = [n_nodes * n_nodes, 3, 5]  # GRAPH_W + DBL + BOOL
        settypes = ["GRAPH_W", "DBL", "BOOL"]
        candidates = [list(range(n_nodes)), [], list(range(5))]
        
        def mixed_fitness(int_vals, dbl_vals, data):
            """Combined fitness from all three types."""
            # When multiple arrays exist, they're in lists
            graph_sum = float(np.sum(dbl_vals[0]))  # GRAPH_W
            param_sum = float(np.sum(dbl_vals[1]))  # DBL
            bool_count = float(np.sum(int_vals[0]))  # BOOL
            return graph_sum + param_sum + bool_count
        
        result = evolve(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=mixed_fitness,
            n_stat=1,
            control={"generations": 10, "popsize": 20, "progress": False}
        )
        
        assert result is not None
        assert len(result.selected_indices) == 1  # BOOL
        assert len(result.selected_values) == 2  # GRAPH_W + DBL
        
        # Verify constraints
        graph = result.selected_values[0]
        assert np.all(graph >= 0) and np.all(graph <= 1)
        params = result.selected_values[1]
        assert len(params) == 3
        bool_mask = result.selected_indices[0]
        assert set(np.unique(bool_mask)).issubset({0, 1})
    
    def test_graph_multiobjective_integration(self):
        """Test GRAPH_W with multi-objective in full pipeline."""
        n_nodes = 3
        setsizes = [n_nodes * n_nodes]
        settypes = ["GRAPH_W"]
        candidates = [list(range(n_nodes))]
        
        def dual_objective(dbl_vals, data):
            graph = dbl_vals
            obj1 = float(np.sum(graph))
            obj2 = float(np.sum(graph ** 2))
            return [obj1, obj2]
        
        result = evolve(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=dual_objective,
            n_stat=2,
            control={"generations": 10, "popsize": 20, "progress": False}
        )
        
        assert hasattr(result, 'pareto_front')
        assert len(result.pareto_front) >= 1  # May be single point if objectives correlate
        assert hasattr(result, 'pareto_solutions')
        
        # Verify all Pareto solutions satisfy constraints
        for sol_dict in result.pareto_solutions:
            graph = sol_dict['selected_values'][0]
            assert np.all(graph >= 0) and np.all(graph <= 1)
    
    def test_partition_realistic_clustering(self):
        """Test PARTITION with realistic clustering scenario."""
        np.random.seed(42)
        n_items = 15
        n_groups = 3
        
        # Simulated item features
        features = np.random.randn(n_items, 2)
        
        setsizes = [n_items]
        settypes = ["PARTITION"]
        candidates = [list(range(n_groups))]
        
        def clustering_fitness(int_vals, data):
            """Maximize within-cluster similarity."""
            partition = int_vals
            features = data['features']
            
            total_variance = 0.0
            for g in range(n_groups):
                mask = partition == g
                if np.sum(mask) > 0:
                    cluster_features = features[mask]
                    centroid = np.mean(cluster_features, axis=0)
                    variance = np.sum((cluster_features - centroid) ** 2)
                    total_variance += variance
            
            return -float(total_variance)  # Minimize variance
        
        result = evolve(
            data={'features': features},
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=clustering_fitness,
            n_stat=1,
            control={"generations": 15, "popsize": 25, "progress": False}
        )
        
        assert result is not None
        partition = result.selected_indices[0]
        
        # Verify partition validity
        assert len(partition) == n_items
        assert np.max(partition) < n_groups
        assert np.min(partition) >= 0
        
        # Verify at least some diversity (not all in one cluster)
        assert len(set(partition)) > 1
    
    def test_simplex_portfolio_optimization(self):
        """Test SIMPLEX with realistic portfolio optimization."""
        np.random.seed(42)
        n_assets = 5
        
        # Simulated asset returns
        asset_returns = np.array([0.05, 0.08, 0.12, 0.06, 0.10])
        
        setsizes = [n_assets]
        settypes = ["SIMPLEX"]
        candidates = [[]]
        
        def portfolio_fitness(dbl_vals, data):
            """Maximize expected return."""
            weights = dbl_vals
            returns = data['returns']
            expected_return = float(np.dot(weights, returns))
            return expected_return
        
        result = evolve(
            data={'returns': asset_returns},
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=portfolio_fitness,
            n_stat=1,
            control={"generations": 15, "popsize": 20, "progress": False}
        )
        
        assert result is not None
        weights = result.selected_values[0]
        
        # Verify simplex constraints
        assert np.all(weights >= -1e-6)
        assert np.abs(np.sum(weights) - 1.0) < 1e-5
        
        # Verify portfolio makes sense (should favor higher return assets)
        expected_return = np.dot(weights, asset_returns)
        assert expected_return > 0.05  # At least better than worst asset
    
    @pytest.mark.slow
    def test_graph_with_neural_enhancement(self):
        """Test GRAPH_W with VAE enabled."""
        n_nodes = 3
        setsizes = [n_nodes * n_nodes]
        settypes = ["GRAPH_W"]
        candidates = [list(range(n_nodes))]
        
        def graph_fitness(dbl_vals, data):
            return float(np.sum(dbl_vals))
        
        result = evolve(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=graph_fitness,
            n_stat=1,
            control={
                "generations": 10,
                "popsize": 20,
                "progress": False,
                "use_vae": True,
                "nn_sample_fraction": 0.2
            }
        )
        
        assert result is not None
        graph = result.selected_values[0]
        assert np.all(graph >= 0) and np.all(graph <= 1)


class TestAdvancedHeadsConstraintPreservation:
    """Test that all advanced heads preserve constraints across all operations."""
    
    def test_spd_constraint_preservation_chain(self):
        """Test SPD constraints through initialization, mutation, crossover."""
        from evosolve.algorithms import initialize_population
        from evosolve.operators import mutation, crossover
        
        n = 3
        setsizes = [n * n]
        settypes = ["SPD"]
        candidates = [[]]
        
        # Initialize
        pop = initialize_population(candidates, setsizes, settypes, pop_size=10)
        
        for sol in pop:
            mat = sol.dbl_values[0].reshape(n, n)
            assert np.allclose(mat, mat.T, atol=1e-5), "Init: not symmetric"
            eigvals = np.linalg.eigvalsh(mat)
            assert np.all(eigvals > 0), "Init: not PD"
        
        # Mutate
        mutation(pop, candidates, settypes, mutprob=1.0, mutintensity=0.2)
        
        for sol in pop:
            mat = sol.dbl_values[0].reshape(n, n)
            assert np.allclose(mat, mat.T, atol=1e-5), "After mutation: not symmetric"
            eigvals = np.linalg.eigvalsh(mat)
            assert np.all(eigvals > 0), "After mutation: not PD"
        
        # Crossover
        from evosolve.operators import crossover
        offspring = crossover(pop, crossprob=0.8, crossintensity=0.5, 
                             settypes=settypes, candidates=candidates)[:5]
        
        for sol in offspring:
            mat = sol.dbl_values[0].reshape(n, n)
            assert np.allclose(mat, mat.T, atol=1e-5), "After crossover: not symmetric"
            eigvals = np.linalg.eigvalsh(mat)
            assert np.all(eigvals > 0), "After crossover: not PD"
    
    def test_simplex_constraint_preservation_chain(self):
        """Test SIMPLEX constraints through initialization, mutation, crossover."""
        from evosolve.algorithms import initialize_population
        from evosolve.operators import mutation, crossover
        
        dim = 4
        setsizes = [dim]
        settypes = ["SIMPLEX"]
        candidates = [[]]
        
        # Initialize
        pop = initialize_population(candidates, setsizes, settypes, pop_size=10)
        
        for sol in pop:
            vec = sol.dbl_values[0]
            assert np.all(vec >= -1e-6), "Init: negative values"
            assert np.abs(np.sum(vec) - 1.0) < 1e-5, "Init: doesn't sum to 1"
        
        # Mutate
        mutation(pop, candidates, settypes, mutprob=1.0, mutintensity=0.2)
        
        for sol in pop:
            vec = sol.dbl_values[0]
            assert np.all(vec >= -1e-6), "After mutation: negative values"
            assert np.abs(np.sum(vec) - 1.0) < 1e-5, "After mutation: doesn't sum to 1"
        
        # Crossover
        from evosolve.operators import crossover
        offspring = crossover(pop, crossprob=0.8, crossintensity=0.5,
                             settypes=settypes, candidates=candidates)[:5]
        
        for sol in offspring:
            vec = sol.dbl_values[0]
            assert np.all(vec >= -1e-6), "After crossover: negative values"
            assert np.abs(np.sum(vec) - 1.0) < 1e-5, "After crossover: doesn't sum to 1"

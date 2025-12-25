import pytest
import numpy as np
from evosolve.algorithms import initialize_population
from evosolve.operators import mutation, crossover
from evosolve.solution import Solution

class TestGraphHead:
    """Tests for Graph Optimization Head."""
    
    def test_initialization_weighted_graph(self):
        """Test random initialization of weighted graphs (continuous)."""
        pop_size = 5
        n_nodes = 4
        setsizes = [n_nodes * n_nodes] # Flattended adjacency matrix
        settypes = ["GRAPH_W"]
        candidates = [list(range(n_nodes))] # Not strictly used for continuous but needed for api
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size)
        
        assert len(pop) == pop_size
        for sol in pop:
            assert len(sol.dbl_values) == 1
            matrix = sol.dbl_values[0]
            assert matrix.shape == (n_nodes * n_nodes,)
            assert np.all(matrix >= 0) and np.all(matrix <= 1)
            
    def test_initialization_unweighted_graph(self):
        """Test random initialization of unweighted graphs (discrete)."""
        pop_size = 5
        n_nodes = 4
        setsizes = [n_nodes * n_nodes]
        settypes = ["GRAPH_U"]
        candidates = [list(range(n_nodes))] 
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size)
        
        assert len(pop) == pop_size
        for sol in pop:
            assert len(sol.int_values) == 1
            matrix = sol.int_values[0]
            assert matrix.shape == (n_nodes * n_nodes,)
            assert set(np.unique(matrix)).issubset({0, 1})

    def test_graph_mutation_weighted(self):
        """Test mutation modifies the graph."""
        n_nodes = 4
        setsizes = [n_nodes * n_nodes]
        settypes = ["GRAPH_W"]
        candidates = [list(range(n_nodes))]
        
        pop = initialize_population(candidates, setsizes, settypes, 5)
        original_values = [sol.dbl_values[0].copy() for sol in pop]
        
        # Apply mutation
        mutation(pop, candidates, settypes, mutprob=1.0, mutintensity=0.1)
        
        for i, sol in enumerate(pop):
            assert not np.array_equal(sol.dbl_values[0], original_values[i])
            # Ensure bounds are respected
            assert np.all(sol.dbl_values[0] >= 0)
            assert np.all(sol.dbl_values[0] <= 1)
            
    def test_graph_mutation_unweighted(self):
        """Test mutation modifies the discrete graph (bit flip)."""
        n_nodes = 4
        setsizes = [n_nodes * n_nodes]
        settypes = ["GRAPH_U"]
        candidates = [list(range(n_nodes))]
        
        pop = initialize_population(candidates, setsizes, settypes, 5)
        original_values = [sol.int_values[0].copy() for sol in pop]
        
        mutation(pop, candidates, settypes, mutprob=1.0, mutintensity=0.1)
        
        for i, sol in enumerate(pop):
            assert not np.array_equal(sol.int_values[0], original_values[i])
            assert set(np.unique(sol.int_values[0])).issubset({0, 1})


class TestManifoldHead:
    """Tests for Manifold-Constrained Optimization Head."""
    
    def test_spd_initialization_and_constraints(self):
        """Test SPD matrix initialization and mutation preservation."""
        pop_size = 5
        n = 3
        setsizes = [n * n]
        settypes = ["SPD"]
        candidates = [[]]
        
        # Initialize
        pop = initialize_population(candidates, setsizes, settypes, pop_size)
        
        for sol in pop:
            assert len(sol.dbl_values) == 1
            flat = sol.dbl_values[0]
            mat = flat.reshape(n, n)
            
            # Check Symmetry
            assert np.allclose(mat, mat.T, atol=1e-5)
            # Check Positive Definiteness (Eigenvalues > 0)
            vals = np.linalg.eigvalsh(mat)
            assert np.all(vals > 0)

        # Mutate and check preservation
        mutation(pop, candidates, settypes, mutprob=1.0, mutintensity=0.1)
        
        for sol in pop:
            flat = sol.dbl_values[0]
            mat = flat.reshape(n, n)
            assert np.allclose(mat, mat.T, atol=1e-5) # Symmetry must be preserved
            vals = np.linalg.eigvalsh(mat)
            assert np.all(vals > 0) # PD must be preserved

    def test_simplex_initialization_and_constraints(self):
        """Test Simplex initialization and mutation preservation."""
        pop_size = 5
        dim = 5
        setsizes = [dim]
        settypes = ["SIMPLEX"]
        candidates = [[]]
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size)
        
        for sol in pop:
            vec = sol.dbl_values[0]
            assert np.all(vec >= 0)
            assert np.isclose(np.sum(vec), 1.0)
            
        # Mutate
        mutation(pop, candidates, settypes, mutprob=1.0, mutintensity=0.1)
        
        for sol in pop:
            vec = sol.dbl_values[0]
            assert np.all(vec >= 0)
            assert np.isclose(np.sum(vec), 1.0)


class TestPartitionHead:
    """Tests for Partition/Clustering Head."""
    
    def test_partition_mechanics(self):
        """Test partition initialization and mutation."""
        pop_size = 5
        n_items = 10
        n_groups = 3
        setsizes = [n_items]
        settypes = ["PARTITION"]
        # Candidates define the valid group IDs
        candidates = [list(range(n_groups))]
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size)
        
        for sol in pop:
            assert len(sol.int_values) == 1
            part = sol.int_values[0]
            assert len(part) == n_items
            assert np.max(part) < n_groups
            assert np.min(part) >= 0
            
        original_parts = [sol.int_values[0].copy() for sol in pop]
        
        # Mutate (Move operator)
        mutation(pop, candidates, settypes, mutprob=1.0, mutintensity=1.0)
        
        for i, sol in enumerate(pop):
            part = sol.int_values[0]
            # Verify changes happened
            assert not np.array_equal(part, original_parts[i])
            # Verify constraints
            assert np.max(part) < n_groups
            assert np.min(part) >= 0


class TestMultiObjectiveHeads:
    """Tests for multi-objective optimization with advanced heads."""
    
    def test_graph_w_multiobjective(self):
        """Test GRAPH_W with multi-objective optimization (sparsity vs sum)."""
        from evosolve.core import train_sel
        
        n_nodes = 3
        setsizes = [n_nodes * n_nodes]
        settypes = ["GRAPH_W"]
        candidates = [list(range(n_nodes))]
        
        def dual_objective(dbl_vals, data):
            """Maximize sparsity (# zeros), maximize edge sum."""
            graph = dbl_vals[0]
            sparsity = float(np.sum(graph < 0.1))  # Count near-zero edges
            edge_sum = float(np.sum(graph))
            return [sparsity, edge_sum]
        
        result = train_sel(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=dual_objective,
            n_stat=2,
            control={"generations": 15, "popsize": 20, "progress": False}
        )
        
        # Verify we got a Pareto front
        assert hasattr(result, 'pareto_front')
        assert len(result.pareto_front) > 1
        
        # Verify all solutions satisfy constraints
        for pf in result.pareto_front:
            assert len(pf) == 2
            assert pf[0] >= 0  # Sparsity >= 0
            assert 0 <= pf[1] <= n_nodes * n_nodes  # Sum bounded
    
    def test_graph_u_multiobjective(self):
        """Test GRAPH_U with multi-objective (connectivity vs edges)."""
        from evosolve.core import train_sel
        
        n_nodes = 4
        setsizes = [n_nodes * n_nodes]
        settypes = ["GRAPH_U"]
        candidates = [list(range(n_nodes))]
        
        def dual_objective(int_vals, data):
            """Minimize edges, maximize connectivity."""
            # When there's only one int array, it's passed directly (not in a list)
            graph = int_vals.reshape(n_nodes, n_nodes)
            n_edges = float(np.sum(graph))
            # Simple connectivity: diagonal + off-diagonal sum
            connectivity = float(np.sum(np.diag(graph)) + np.sum(graph))
            return [-n_edges, connectivity]  # Minimize edges, maximize connectivity
        
        result = train_sel(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=dual_objective,
            n_stat=2,
            control={"generations": 15, "popsize": 20, "progress": False}
        )
        
        assert hasattr(result, 'pareto_front')
        assert len(result.pareto_front) > 0
        
        # Verify all are binary
        for sol_dict in result.pareto_solutions:
            graph = sol_dict['selected_indices'][0]
            assert set(np.unique(graph)).issubset({0, 1})
    
    def test_spd_multiobjective(self):
        """Test SPD with multi-objective (determinant vs condition number)."""
        from evosolve.core import train_sel
        
        n = 2
        setsizes = [n * n]
        settypes = ["SPD"]
        candidates = [[]]
        
        def dual_objective(dbl_vals, data):
            """Maximize determinant, minimize condition number."""
            # When there's only one dbl array, it's passed directly
            flat = dbl_vals
            mat = flat.reshape(n, n)
            
            # Verify SPD constraints
            assert np.allclose(mat, mat.T, atol=1e-5), "Matrix not symmetric"
            eigvals = np.linalg.eigvalsh(mat)
            assert np.all(eigvals > 0), "Matrix not positive definite"
            
            det = float(np.linalg.det(mat))
            cond = float(np.linalg.cond(mat))
            
            return [det, -cond]  # Maximize det, minimize cond
        
        result = train_sel(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=dual_objective,
            n_stat=2,
            control={"generations": 15, "popsize": 20, "progress": False}
        )
        
        assert hasattr(result, 'pareto_front')
        assert len(result.pareto_front) > 0
        
        # Verify all Pareto solutions maintain SPD property
        for sol_dict in result.pareto_solutions:
            flat = sol_dict['selected_values'][0]
            mat = flat.reshape(n, n)
            assert np.allclose(mat, mat.T, atol=1e-5)
            eigvals = np.linalg.eigvalsh(mat)
            assert np.all(eigvals > 0)
    
    def test_simplex_multiobjective(self):
        """Test SIMPLEX with multi-objective (entropy vs concentration)."""
        from evosolve.core import train_sel
        
        dim = 5
        setsizes = [dim]
        settypes = ["SIMPLEX"]
        candidates = [[]]
        
        def dual_objective(dbl_vals, data):
            """Maximize entropy, minimize max component (concentration)."""
            # When there's only one dbl array, it's passed directly
            vec = dbl_vals
            
            # Verify simplex constraints (relaxed tolerance)
            assert np.all(vec >= -1e-6), "Negative values in simplex"
            assert np.abs(np.sum(vec) - 1.0) < 0.1, f"Simplex sum is {np.sum(vec)}, expected 1.0"
            
            # Entropy
            eps = 1e-10
            entropy = -float(np.sum(vec * np.log(vec + eps)))
            
            # Concentration (negative max for minimization)
            max_component = float(np.max(vec))
            
            return [entropy, -max_component]
        
        result = train_sel(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=dual_objective,
            n_stat=2,
            control={"generations": 15, "popsize": 20, "progress": False}
        )
        
        assert hasattr(result, 'pareto_front')
        assert len(result.pareto_front) > 0
        
        # Verify all Pareto solutions maintain simplex constraints
        for sol_dict in result.pareto_solutions:
            vec = sol_dict['selected_values'][0]
            assert np.all(vec >= -1e-6)
            assert np.abs(np.sum(vec) - 1.0) < 1e-5
    
    def test_partition_multiobjective(self):
        """Test PARTITION with multi-objective (balance vs diversity)."""
        from evosolve.core import train_sel
        
        n_items = 12
        n_groups = 3
        setsizes = [n_items]
        settypes = ["PARTITION"]
        candidates = [list(range(n_groups))]
        
        def dual_objective(int_vals, data):
            """Maximize balance, maximize number of used groups."""
            # When there's only one int array, it's passed directly
            part = int_vals
            
            # Verify partition constraints
            assert np.max(part) < n_groups
            assert np.min(part) >= 0
            
            # Balance: negative std of cluster sizes
            sizes = [np.sum(part == g) for g in range(n_groups)]
            balance = -float(np.std(sizes))
            
            # Diversity: number of non-empty groups
            diversity = float(len(set(part.tolist())))  # Convert to list for set()
            
            return [balance, diversity]
        
        result = train_sel(
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat=dual_objective,
            n_stat=2,
            control={"generations": 15, "popsize": 20, "progress": False}
        )
        
        assert hasattr(result, 'pareto_front')
        assert len(result.pareto_front) > 0
        
        # Verify all partitions are valid
        for sol_dict in result.pareto_solutions:
            part = sol_dict['selected_indices'][0]
            assert np.max(part) < n_groups
            assert np.min(part) >= 0

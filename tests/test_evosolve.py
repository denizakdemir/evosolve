import unittest
import numpy as np
import random
from evosolve.solution import Solution
from evosolve.operators import crossover, mutation
from evosolve.selection import selection, fast_non_dominated_sort
from evosolve.algorithms import initialize_population, genetic_algorithm

class TestTrainSelPy(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_initialization(self):
        candidates = [[1, 2, 3, 4, 5]]
        setsizes = [3]
        settypes = ["UOS"]
        pop_size = 10
        
        pop = initialize_population(candidates, setsizes, settypes, pop_size)
        
        self.assertEqual(len(pop), pop_size)
        for sol in pop:
            self.assertEqual(len(sol.int_values), 1)
            self.assertEqual(len(sol.int_values[0]), 3)
            self.assertEqual(len(sol.dbl_values), 0)

    def test_crossover(self):
        parent1 = Solution(int_values=[[1, 2, 3]])
        parent2 = Solution(int_values=[[4, 5, 6]])
        
        parents = [parent1, parent2]
        offspring = crossover(parents, crossprob=1.0, crossintensity=0.5, settypes=["UOS"], candidates=[[1, 2, 3, 4, 5, 6]])
        
        self.assertEqual(len(offspring), 2)
        # Check if crossover happened (some values swapped)
        # Compare arrays properly
        self.assertFalse(np.array_equal(offspring[0].int_values[0], parent1.int_values[0]))

    def test_mutation(self):
        pop = [Solution(int_values=[[1, 2, 3]])]
        
        candidates = [[1, 2, 3, 4, 5]]
        settypes = ["UOS"]
        
        # Force mutation
        mutation(pop, candidates, settypes, mutprob=1.0, mutintensity=1.0)
        
        # Check if mutation happened
        self.assertFalse(np.array_equal(pop[0].int_values[0], np.array([1, 2, 3])))

    def test_selection(self):
        pop = []
        for i in range(10):
            sol = Solution()
            sol.fitness = i
            pop.append(sol)
            
        selected = selection(pop, n_elite=2, tournament_size=3)
        
        self.assertEqual(len(selected), 10)
        # Elitism should preserve best (fitness 9 and 8)
        fitnesses = [s.fitness for s in selected]
        self.assertIn(9, fitnesses)
        self.assertIn(8, fitnesses)

    def test_integration(self):
        # Simple problem: maximize sum of selected integers
        candidates = [list(range(10))]
        setsizes = [3]
        settypes = ["UOS"]
        
        def fitness(sol, data):
            # Sol is array now? NO, prepare_function_args returns what the storage is.
            # If sol.int_values is list[ndarray], arg is ndarray (if len 1)
            # sum(sol) works on ndarray
            return sum(sol)
            
        data = {}
        
        result = genetic_algorithm(
            data=data,
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat_func=fitness,
            control={"niterations": 10, "npop": 20, "progress": False}
        )
        
        # Max possible sum is 7+8+9 = 24
        self.assertTrue(result["fitness"] >= 20)  # Should be close to optimal

    def test_surrogate_objective(self):
        # Mock surrogate model
        class MockSurrogate:
            def __init__(self):
                self.is_fitted = True
                
            def predict(self, solutions):
                # Return fake fitness based on first value
                means = [float(sum(sol.int_values[0])) for sol in solutions]
                stds = [0.0] * len(solutions)
                return means, stds
                
            def fit(self, solutions, fitnesses):
                pass

        candidates = [list(range(10))]
        setsizes = [3]
        settypes = ["UOS"]
        
        def fitness(sol, data):
            return 0 # Should not be called if surrogate is used
            
        data = {}
        surrogate = MockSurrogate()
        
        from evosolve.algorithms import evaluate_fitness
        
        # Correct initialization using constructor to ensure array storage
        pop = [Solution(int_values=[[1, 2, 3]])]
        
        control = {"use_surrogate_objective": True}
        
        evaluate_fitness(pop, fitness, data, control=control, surrogate_model=surrogate)
        
        self.assertEqual(pop[0].fitness, 6.0)

    def test_generate_from_surrogate(self):
        # Mock surrogate model
        class MockSurrogate:
            def __init__(self):
                self.is_fitted = True
                
            def predict(self, solutions):
                # Return fake fitness based on first value
                means = [float(sum(sol.int_values[0])) for sol in solutions]
                stds = [0.0] * len(solutions)
                return means, stds

            def predict_from_values(self, int_values, dbl_values):
                score = 0.0
                if int_values is not None and len(int_values) > 0:
                    score = float(sum(int_values[0]))
                return score, 0.0
                
        from evosolve.algorithms import generate_from_surrogate
        
        candidates = [list(range(10))]
        setsizes = [3]
        settypes = ["UOS"]
        surrogate = MockSurrogate()
        
        generated = generate_from_surrogate(surrogate, candidates, setsizes, settypes, n_solutions=2, n_iter=10)
        
        self.assertEqual(len(generated), 2)
        # Check length of generated array
        self.assertEqual(len(generated[0].int_values[0]), 3)

if __name__ == '__main__':
    unittest.main()

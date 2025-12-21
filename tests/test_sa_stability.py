
import unittest
import numpy as np
import random
from trainselpy.algorithms import simulated_annealing
from trainselpy.solution import Solution

class TestSimulatedAnnealingRepro(unittest.TestCase):
    def test_sa_instability(self):
        """
        Test that simulates conditions where Simulated Annealing might fail due to
        numerical instability (overflow or zero division) when temperature drops too low.
        """
        # Mock fitness function
        def mock_fitness(int_vals, data):
            # Return random fitness to drive SA
            return random.random()

        # Setup minimal problem
        candidates = [[1, 2, 3, 4, 5]]
        settypes = ["UOS"]
        
        # Initial solution
        sol = Solution(int_values=[[1]])
        sol.fitness = 0.5
        
        # Force very fast cooling to reach near-zero temperature
        # n_iter large + temp_final very small implies decay rate is handled,
        # but locally 'temp' might become 0.0 or effectively 0.0 leading to overflow in exp
        
        # We manually check the behavior when temp is extremely low
        # Note: The current implementation initializes temp = temp_init
        # and updates temp = temp * exp(...)
        
        # We want to use parameters that usually cause issues.
        # If temp_final is 1e-300, it might underflow to 0.
        
        try:
            simulated_annealing(
                solution=sol,
                candidates=candidates,
                settypes=settypes,
                stat_func=mock_fitness,
                data={},
                n_stat=1,
                n_iter=1000,
                temp_init=100.0,
                temp_final=1e-320 # Simulating extreme cooling target
            )
        except (ZeroDivisionError, RuntimeWarning, OverflowError):
            self.fail("Simulated Annealing raised numerical error with extreme temperature parameters")
            
    def test_sa_acceptance_logic(self):
        """
        Directly test the acceptance logic isolation. This is hard to unit test 
        against the function which encapsulates the loop, but we can verify 
        it doesn't crash on standard extreme cases.
        """
        candidates = [[1, 2, 3]]
        settypes = ["UOS"]
        sol = Solution(int_values=[[1]])
        sol.fitness = 0.5
        
        # Mock fitness that always improves vs always worsens
        # Case 1: Worsening move with 0 temperature -> Should not accept (unless Delta > 0, which it isn't)
        # But wait, Delta = New - Old. Maximization.
        # If New < Old, Delta < 0. exp(Delta/T).
        # If T -> 0, Delta/T -> -Inf. exp(-Inf) -> 0. Safe.
        
        # Case 2: Worsening move with T very small positive.
        
        # The crash risk is usually:
        # 1. T becomes exactly 0.0 -> Divide by Zero.
        # 2. Delta is positive (New > Old), T is near 0. Delta/T -> +Inf. exp(+Inf) -> Overflow.
        # But wait, if Delta > 0, we accept immediately!
        # code: if delta > 0 or random.random() < np.exp(delta / temp):
        # The exp is only evaluated if delta <= 0.
        # So only exp(negative / small_positive) is evaluated.
        # negative / small_positive -> -Inf.
        # exp(-Infinite) -> 0.0.
        # This seems safe from OverflowError for the standard case!
        
        # Re-evaluating the "Instability" claim.
        # "delta > 0 or ..." -> Short circuit evaluation.
        # If delta <= 0, then delta/temp is negative.
        # np.exp(large_negative_number) is 0.0 (underflow to zero is safe in python floats usually).
        
        # Where is the risk?
        # If temp becomes 0.0 literally. 
        # delta/0 -> DivByZero.
        
        def mock_worse_fitness(int_vals, data):
             return 0.1 # Worse than 0.5
             
        try:
             # Force temp to reach 0
             # temp_final=0.0 is not allowed by math.log(temp_init/temp_final) usually
             # but user might pass very small number.
             # If n_iter is huge, decay rate is small.
             # If temp_final is effectively 0 (e.g. 0.0 passed typically raises error in log).
             
             simulated_annealing(
                solution=sol,
                candidates=candidates,
                settypes=settypes,
                stat_func=mock_worse_fitness,
                data={},
                n_iter=100,
                temp_init=1.0,
                temp_final=0.0 # Should be handled safely now
            )
        except ZeroDivisionError:
             self.fail("ZeroDivisionError occurred inside loop")
        except ValueError:
             self.fail("ValueError occurred (should have been handled by safety check)")

if __name__ == '__main__':
    unittest.main()

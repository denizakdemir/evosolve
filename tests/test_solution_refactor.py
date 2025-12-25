
import unittest
import numpy as np
from evosolve.solution import Solution, flatten_dbl_values, unflatten_dbl_values

class TestSolutionStorage(unittest.TestCase):
    def test_init_and_storage(self):
        """Test that Solution stores values as NumPy arrays."""
        int_vals = [[1, 2, 3], [4, 5]]
        dbl_vals = [[0.1, 0.2], [0.3]]
        
        sol = Solution(int_values=int_vals, dbl_values=dbl_vals)
        
        # Check storage type (TARGET BEHAVIOR)
        self.assertIsInstance(sol.int_values, list)
        self.assertIsInstance(sol.int_values[0], np.ndarray)
        self.assertTrue(np.array_equal(sol.int_values[0], np.array([1, 2, 3])))
        
        self.assertIsInstance(sol.dbl_values, list)
        self.assertIsInstance(sol.dbl_values[0], np.ndarray)
        self.assertTrue(np.allclose(sol.dbl_values[0], np.array([0.1, 0.2])))
        
    def test_copy(self):
        """Test deep copy with NumPy arrays."""
        int_vals = [[1, 2, 3]]
        sol = Solution(int_vals)
        sol_copy = sol.copy()
        
        # Verify independence
        sol_copy.int_values[0][0] = 999
        self.assertEqual(sol.int_values[0][0], 1)
        self.assertEqual(sol_copy.int_values[0][0], 999)
        
    def test_hashing(self):
        """Test hashing works with NumPy arrays."""
        sol = Solution([[1, 2, 3]], [[0.1]])
        h1 = sol.get_hash()
        
        sol2 = Solution([[1, 2, 3]], [[0.1]])
        h2 = sol2.get_hash()
        
        self.assertEqual(h1, h2)
        
        # Change value
        sol.int_values[0][0] = 9
        sol.invalidate_hash()
        h3 = sol.get_hash()
        
        self.assertNotEqual(h1, h3)

    def test_flatten_helpers(self):
        """Test flatten helpers accept list of arrays."""
        template = [np.array([1.0, 2.0]), np.array([3.0])]
        flat = flatten_dbl_values(template)
        self.assertEqual(len(flat), 3)
        
        restored = unflatten_dbl_values(flat, template)
        self.assertTrue(np.allclose(restored[0], template[0]))

if __name__ == '__main__':
    unittest.main()

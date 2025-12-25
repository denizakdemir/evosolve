
import unittest
import numpy as np
from evosolve.relaxations import DOptimality, AOptimality, ConvexRelaxationSolver, discretize, CDMeanOptimality, PEVOptimality

class TestRelaxations(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create a random design matrix
        self.N = 50
        self.P = 5
        self.K = 10
        self.X = np.random.randn(self.N, self.P)
        
    def test_d_opt_slsqp(self):
        crit = DOptimality()
        solver = ConvexRelaxationSolver(crit, method="SLSQP")
        w_opt = solver.solve(self.X, self.K)
        
        self.assertAlmostEqual(np.sum(w_opt), self.K, places=4)
        self.assertTrue(np.all(w_opt >= -1e-5))
        self.assertTrue(np.all(w_opt <= 1.00001))
        
        # Check that we actually optimized something (better than random uniform)
        w_unif = np.ones(self.N) * (self.K / self.N)
        # Minimize -logdet implies maximize logdet
        val_opt = crit.evaluate(w_opt, self.X)
        val_unif = crit.evaluate(w_unif, self.X)
        
        # Recall evaluate returns -logdet (minimization)
        # So val_opt should be < val_unif
        self.assertLess(val_opt, val_unif)

    def test_d_opt_fw(self):
        crit = DOptimality()
        solver = ConvexRelaxationSolver(crit, method="FW")
        w_opt = solver.solve(self.X, self.K)
        
        self.assertAlmostEqual(np.sum(w_opt), self.K, places=4)
        # FW keeps feasibility exactly usually
        self.assertTrue(np.all(w_opt >= 0))
        self.assertTrue(np.all(w_opt <= 1))
        
        w_unif = np.ones(self.N) * (self.K / self.N)
        val_opt = crit.evaluate(w_opt, self.X)
        val_unif = crit.evaluate(w_unif, self.X)
        self.assertLess(val_opt, val_unif)

    def test_a_opt_slsqp(self):
        crit = AOptimality()
        solver = ConvexRelaxationSolver(crit, method="SLSQP")
        w_opt = solver.solve(self.X, self.K)
        
        self.assertAlmostEqual(np.sum(w_opt), self.K, places=4)
        
        w_unif = np.ones(self.N) * (self.K / self.N)
        val_opt = crit.evaluate(w_opt, self.X)
        val_unif = crit.evaluate(w_unif, self.X)
        self.assertLess(val_opt, val_unif)

    def test_cdmean_opt(self):
        # Create random G matrix
        G = self.X @ self.X.T + np.eye(self.N) # Make it PD
        crit = CDMeanOptimality(G, lambda_val=1.0)
        solver = ConvexRelaxationSolver(crit, method="FW")
        
        # Optimize with K=10
        w_opt = solver.solve(self.X, self.K)
        
        self.assertAlmostEqual(np.sum(w_opt), self.K, places=4)
        self.assertTrue(np.all(w_opt >= 0))
        
        # Compare with uniform
        w_unif = np.ones(self.N) * (self.K / self.N)
        val_opt = crit.evaluate(w_opt)
        val_unif = crit.evaluate(w_unif)
        
        # CDMean crit minimizes weighted PEV.
        # So val_opt should be < val_unif.
        self.assertLess(val_opt, val_unif)

    def test_pev_opt(self):
        # Create random G matrix
        G = self.X @ self.X.T + np.eye(self.N)
        crit = PEVOptimality(G, lambda_val=0.5)
        # Use SLSQP for variety
        solver = ConvexRelaxationSolver(crit, method="SLSQP")
        
        w_opt = solver.solve(self.X, self.K)
        
        self.assertAlmostEqual(np.sum(w_opt), self.K, places=4)
        
        w_unif = np.ones(self.N) * (self.K / self.N)
        val_opt = crit.evaluate(w_opt)
        val_unif = crit.evaluate(w_unif)
        self.assertLess(val_opt, val_unif)

    def test_discretize_top_k(self):
        w = np.array([0.1, 0.9, 0.8, 0.2, 0.5])
        k = 2
        indices = discretize(w, k, method='top_k')
        self.assertEqual(indices, [1, 2]) # 0-indexed: 1 (0.9), 2 (0.8)

if __name__ == '__main__':
    unittest.main()

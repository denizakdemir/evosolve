"""
Surrogate model implementation for EvoSolve.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from evosolve.solution import Solution, flatten_dbl_values

class SurrogateModel:
    """
    Base class for surrogate models.
    """
    def __init__(
        self,
        candidates: List[List[int]],
        settypes: List[str],
        model_type: str = "gp"
    ):
        """
        Initialize surrogate model.
        
        Parameters
        ----------
        candidates : List[List[int]]
            List of candidate sets
        settypes : List[str]
            List of set types
        model_type : str
            Type of model: "gp" (Gaussian Process) or "rf" (Random Forest)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for surrogate optimization")
            
        self.candidates = candidates
        self.settypes = settypes
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
        # Calculate dimensions
        self.int_dims = [len(c) for c in candidates]
        self.total_int_dim = sum(self.int_dims)
        
        # Initialize model
        if model_type == "gp":
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
        elif model_type == "rf":
            self.model = RandomForestRegressor(n_estimators=100, n_jobs=1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _encode_values(self, int_values: List[List[int]], dbl_values: List[List[float]]) -> np.ndarray:
        """
        Encode values directly into a feature vector using internal data structures.
        Avoids dependency on Solution object structure.
        """
        features = []
        
        # Encode integer values (One-Hot-ish)
        for i, values in enumerate(int_values):
            vec = np.zeros(self.int_dims[i])
            
            # Simple mapping assuming candidates[i] contains the values
            # Optimization: Map creation could be cached in __init__ for very large candidate sets
            cand_map = {val: idx for idx, val in enumerate(self.candidates[i])}
            
            for val in values:
                if val in cand_map:
                    vec[cand_map[val]] = 1.0
            
            features.append(vec)
            
        # Encode double values
        if dbl_values:
            features.append(flatten_dbl_values(dbl_values))
            
        return np.concatenate(features)
            
    def _encode_distributional(self, dist_solution) -> np.ndarray:
        """
        Encode a DistributionalSolution into a feature vector.

        Extracts statistical features from the distribution:
        - Mean values (expected value of each decision variable)
        - Variance values (variance of each decision variable)
        - Distribution entropy
        - Number of particles (K)

        Parameters
        ----------
        dist_solution : DistributionalSolution
            Distributional solution to encode

        Returns
        -------
        np.ndarray
            Feature vector representing the distribution
        """
        dist = dist_solution.distribution
        particles = dist.particles
        weights = dist.weights

        # Extract int_values and dbl_values from all particles
        # Compute weighted mean and variance for each dimension

        # Initialize accumulators
        mean_features = []
        var_features = []

        # Process integer values
        if particles[0].int_values:
            for set_idx in range(len(particles[0].int_values)):
                # Collect all values for this set across all particles
                set_shape = particles[0].int_values[set_idx].shape
                all_vals = np.array([p.int_values[set_idx].flatten() for p in particles])  # (K, dims)

                # Weighted mean
                mean_vals = np.sum(weights[:, None] * all_vals, axis=0)
                # Weighted variance
                var_vals = np.sum(weights[:, None] * (all_vals - mean_vals) ** 2, axis=0)

                # Encode mean using the standard encoding
                mean_vec = np.zeros(self.int_dims[set_idx])
                cand_map = {val: idx for idx, val in enumerate(self.candidates[set_idx])}
                # For continuous approximation, use fractional encoding
                for idx, val in enumerate(mean_vals):
                    rounded_val = int(np.round(val))
                    if rounded_val in cand_map:
                        mean_vec[cand_map[rounded_val]] = 1.0

                mean_features.append(mean_vec)
                # Append normalized variance (simple approach)
                var_features.append(var_vals)

        # Process double values
        if particles[0].dbl_values:
            all_dbl = []
            for p in particles:
                if p.dbl_values:
                    all_dbl.append(flatten_dbl_values(p.dbl_values))
                else:
                    all_dbl.append(np.array([]))

            if all_dbl and len(all_dbl[0]) > 0:
                all_dbl = np.array(all_dbl)  # (K, dbl_dim)
                mean_dbl = np.sum(weights[:, None] * all_dbl, axis=0)
                var_dbl = np.sum(weights[:, None] * (all_dbl - mean_dbl) ** 2, axis=0)
                mean_features.append(mean_dbl)
                var_features.append(var_dbl)

        # Compute entropy
        w_safe = weights[weights > 0]
        entropy = -np.sum(w_safe * np.log(w_safe)) if len(w_safe) > 0 else 0.0

        # Concatenate all features
        features = []
        if mean_features:
            features.extend(mean_features)
        if var_features:
            # Flatten and normalize variance features
            var_flat = np.concatenate([v.flatten() for v in var_features])
            # Scale variance to similar range as other features (0-1)
            var_flat = np.clip(var_flat, 0, 10) / 10.0
            features.append(var_flat)

        # Add entropy and K as scalar features
        features.append(np.array([entropy, float(dist.K) / 100.0]))  # Normalize K

        return np.concatenate(features)

    def encode(self, solution) -> np.ndarray:
        """
        Encode a solution into a feature vector.

        Handles both regular Solution and DistributionalSolution objects.

        Parameters
        ----------
        solution : Solution or DistributionalSolution
            Solution to encode

        Returns
        -------
        np.ndarray
            Feature vector
        """
        # Check if it's a DistributionalSolution
        if hasattr(solution, 'distribution') and hasattr(solution.distribution, 'particles'):
            return self._encode_distributional(solution)

        # Regular solution
        return self._encode_values(solution.int_values, solution.dbl_values)

    def fit(self, solutions: List[Solution], fitnesses: List[float]) -> None:
        """
        Fit the surrogate model.
        
        Parameters
        ----------
        solutions : List[Solution]
            List of solutions
        fitnesses : List[float]
            List of fitness values
        """
        X = np.array([self.encode(s) for s in solutions])
        y = np.array(fitnesses).reshape(-1, 1)
        
        # Scale data? GP handles normalize_y=True.
        # But scaling X is good for isotropic kernels.
        # However, X is binary (for int) + continuous.
        # Scaling binary data is debatable.
        # Let's not scale X for now, or only scale continuous part.
        # For simplicity, pass raw X.
        
        self.model.fit(X, y.ravel())
        self.is_fitted = True
        
    def predict(self, solutions: List[Solution]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict fitness for solutions.
        
        Parameters
        ----------
        solutions : List[Solution]
            List of solutions
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Mean prediction and standard deviation (uncertainty)
        """
        if not self.is_fitted:
            return np.zeros(len(solutions)), np.ones(len(solutions))
            
        X = np.array([self.encode(s) for s in solutions])
        
        if self.model_type == "gp":
            mean, std = self.model.predict(X, return_std=True)
            return mean, std
        elif self.model_type == "rf":
            mean = self.model.predict(X)
            # RF doesn't give std directly, can use variance of trees
            # But sklearn RF doesn't expose it easily without loop.
            # Simple approximation: 0 std
            return mean, np.zeros_like(mean)
            
        return np.zeros(len(solutions)), np.zeros(len(solutions))

    def predict_from_values(self, int_values: List[List[int]], dbl_values: List[List[float]]) -> Tuple[float, float]:
        """
        Predict fitness from raw values, avoiding Solution creation overhead.
        
        Parameters
        ----------
        int_values : List[List[int]]
            Integer solution values
        dbl_values : List[List[float]]
            Double solution values
            
        Returns
        -------
        Tuple[float, float]
            Mean and std for the single solution
        """
        if not self.is_fitted:
            return 0.0, 1.0

        # Note: predict expects 2D array (n_samples, n_features)
        X = self._encode_values(int_values, dbl_values).reshape(1, -1)
        
        if self.model_type == "gp":
            mean, std = self.model.predict(X, return_std=True)
            return mean[0], std[0]
        elif self.model_type == "rf":
            mean = self.model.predict(X)
            return mean[0], 0.0
            
        return 0.0, 0.0

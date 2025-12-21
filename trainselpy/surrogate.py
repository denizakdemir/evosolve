"""
Surrogate model implementation for TrainSelPy.
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

from trainselpy.solution import Solution, flatten_dbl_values

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
            
    def encode(self, solution: Solution) -> np.ndarray:
        """
        Encode a solution into a feature vector.
        
        Parameters
        ----------
        solution : Solution
            Solution to encode
            
        Returns
        -------
        np.ndarray
            Feature vector
        """
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

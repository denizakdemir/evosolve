
"""
Convex approximations and relaxations for experimental design / subset selection problems.
Includes generic solvers like Frank-Wolfe and Projected Gradient Descent, 
as well as Scipy-based solvers.
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from typing import List, Dict, Any, Union, Optional, Callable, Tuple
import abc

class ContinuousCriterion(abc.ABC):
    """
    Abstract base class for continuous relaxation criteria.
    """
    
    @abc.abstractmethod
    def evaluate(self, w: np.ndarray, X: np.ndarray) -> float:
        """
        Evaluate the objective function.
        """
        pass
    
    @abc.abstractmethod
    def gradient(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function with respect to w.
        """
        pass
        
    def hessian(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian (optional).
        """
        raise NotImplementedError("Hessian not implemented")


class DOptimality(ContinuousCriterion):
    """
    Continuous relaxation of D-optimality.
    Objective: -log det(X.T @ diag(w) @ X + epsilon * I)
    (We minimize the negative log determinant)
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        
    def evaluate(self, w: np.ndarray, X: np.ndarray) -> float:
        n, p = X.shape
        # M = X^T W X
        # efficient: (X.T * w) @ X
        M = (X.T * w) @ X
        M += np.eye(p) * self.epsilon
        
        # We want to maximize log det, so we minimize -log det
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            return np.inf # Penalize non-pd matrices
        return -logdet
    
    def gradient(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Gradient of -log det(M) is -diag(X M^-1 X^T)
        """
        n, p = X.shape
        M = (X.T * w) @ X + np.eye(p) * self.epsilon
        
        try:
            # Solve M * Y = X.T -> Y = M^-1 X.T
            # The gradient element i is -x_i^T M^-1 x_i
            # We can compute this as - sum(X * Y.T, axis=1)
            # Y.T is X @ M^-1
            
            # Using Cholesky for stability
            L = linalg.cho_factor(M)
            Minv = linalg.cho_solve(L, np.eye(p))
            
            # d_i = x_i^T M^-1 x_i
            # This is the diagonal of X M^-1 X^T
            # Can be computed efficiently
            # term = (X @ Minv) * X
            # grad = -np.sum(term, axis=1)
            
            # More efficient:
            XM = X @ Minv
            grad = -np.sum(XM * X, axis=1)
            
            return grad
            
        except linalg.LinAlgError:
            # If singular, return approximate gradient or large values?
            # For minimization, we want to move away from singularity.
            # Singularity usually means determinant is 0, -logdet is inf.
            return np.zeros_like(w)


class AOptimality(ContinuousCriterion):
    """
    Continuous relaxation of A-optimality.
    Objective: Trace( (X.T @ diag(w) @ X + epsilon * I)^-1 )
    (Minimize the trace of the inverse information matrix)
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        
    def evaluate(self, w: np.ndarray, X: np.ndarray) -> float:
        n, p = X.shape
        M = (X.T * w) @ X + np.eye(p) * self.epsilon
        
        try:
            # Trace(M^-1)
            # Use Cholesky
            L = linalg.cho_factor(M)
            Minv = linalg.cho_solve(L, np.eye(p))
            return np.trace(Minv)
        except linalg.LinAlgError:
            return np.inf

    def gradient(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Gradient of Tr(M^-1) is -diag(X M^-2 X^T)
        """
        n, p = X.shape
        M = (X.T * w) @ X + np.eye(p) * self.epsilon
        
        try:
            L = linalg.cho_factor(M)
            Minv = linalg.cho_solve(L, np.eye(p))
            
            # M^-2 = Minv @ Minv
            Minv2 = Minv @ Minv
            
            # grad_i = -x_i^T M^-2 x_i
            XM2 = X @ Minv2
            grad = -np.sum(XM2 * X, axis=1)
            
            return grad
        except linalg.LinAlgError:
            return np.zeros_like(w)

class CDMeanOptimality(ContinuousCriterion):
    """
    Continuous relaxation of CDMean Optimality.
    Maximizes the mean Coefficient of Determination (Reliability) of random effects.
    Equivalent to minimizing the weighted mean Prediction Error Variance (PEV).
    
    Objective: Sum(PEV_ii / G_ii) for all i.
    PEV_ii is the diagonal of the inverse of the MME matrix LHS.
    
    We approximate the MME system for G-BLUP.
    Standard MME LHS with weights w:
    [ sum(w)     w^T        ]
    [ w          diag(w)+K  ]
    where K = lambda * G^-1.
    
    This solver requires access to G inverse. For large N, this is initially expensive (O(N^3)).
    Once K is computed, each evaluation is O(N^3).
    """
    def __init__(self, G_matrix: np.ndarray, lambda_val: float = 1.0, target_indices: Optional[List[int]] = None):
        self.G = G_matrix
        self.lambda_val = lambda_val
        self.n = G_matrix.shape[0]
        
        # Precompute K = lambda * G^-1
        # Add epsilon for stability
        try:
            self.K = lambda_val * linalg.inv(G_matrix + np.eye(self.n)*1e-6)
        except linalg.LinAlgError:
             # Fallback: pseudo-inverse
            self.K = lambda_val * linalg.pinv(G_matrix)
            
        # Target weights D
        # CDMean maximizes 1 - PEV/G. Equivalent to minimizing PEV/G.
        # So we weight PEV diagonal by 1/G_ii
        G_diag = np.diag(G_matrix)
        weights = 1.0 / (G_diag + 1e-8)
        
        if target_indices is not None:
             # Only target indices matter.
             mask = np.zeros(self.n)
             mask[target_indices] = 1.0
             weights *= mask
        
        # Create full D_aug matrix for the trace gradient
        # D_aug corresponds to the (N+1) x (N+1) inverse matrix
        # Top-left is fixed effect (beta), we ignore it (weight 0).
        # Bottom-right is random effects (u), weights 1/G_ii.
        self.D_u = np.diag(weights)
        self.D_aug = np.zeros((self.n + 1, self.n + 1))
        self.D_aug[1:, 1:] = self.D_u
        
    def evaluate(self, w: np.ndarray, X: np.ndarray = None) -> float:
        # X is ignored here as G is provided in init.
        # Build MME matrix M
        # [ sum(w)   w.T ]
        # [ w        W+K ]
        
        M = np.zeros((self.n + 1, self.n + 1))
        
        w_sum = np.sum(w)
        M[0, 0] = w_sum
        M[0, 1:] = w
        M[1:, 0] = w
        
        # Bottom right: diag(w) + K
        # Efficiently:
        M[1:, 1:] = self.K.copy()
        M[1:, 1:].flat[::self.n + 1] += w
        
        try:
            # We need the inverse to get PEV
            # Z = inv(M)
            # PEV of u is Z[1:, 1:]
            
            # Use Cholesky if M is positive definite. MME typically is if lambda > 0.
            # But with w approx 0, might be issues.
            # Add small epsilon to diagonal for stability
            M.flat[::self.n + 2] += 1e-8
            
            # Using standard solve for inverse column by column? Or full inverse.
            # Since we need trace(D @ Z), we only need diagonal of D @ Z?
            # Actually we need diagonal of Z since D is diagonal.
            # Tr(D Z) = sum_i d_i z_ii
            # So we only need diagonal of Z.
            # Can we compute diagonal of inverse faster? Not really for general matrices.
            # So compute full inverse.
            
            Z = linalg.inv(M)
            
            # Objective: Sum (Z_uu_ii * weights_i)
            # This is trace(D_aug @ Z) efficiently
            # val = np.sum(np.diag(Z)[1:] * np.diag(self.D_u))
            # More robustly:
            val = np.trace(self.D_aug @ Z)
            
            return val
            
        except linalg.LinAlgError:
            return np.inf

    def gradient(self, w: np.ndarray, X: np.ndarray = None) -> np.ndarray:
        # Gradient of f(w) = Tr(D_aug M(w)^-1)
        # d/dw_k = - Tr( M^-1 (dM/dw_k) M^-1 D_aug )
        # = - Tr( (dM/dw_k) Q ) where Q = Z D_aug Z
        
        M = np.zeros((self.n + 1, self.n + 1))
        w_sum = np.sum(w)
        M[0, 0] = w_sum
        M[0, 1:] = w
        M[1:, 0] = w
        M[1:, 1:] = self.K.copy()
        M[1:, 1:].flat[::self.n + 1] += w
        M.flat[::self.n + 2] += 1e-8
        
        try:
            Z = linalg.inv(M)
            
            # Q = Z @ D_aug @ Z
            Q = Z @ self.D_aug @ Z
            
            # Compute gradient vector
            # dM/dw_k has 1s at (0,0), (0, k+1), (k+1, 0), (k+1, k+1)
            # Grad_k = - (Q_00 + Q_0,k+1 + Q_k+1,0 + Q_k+1,k+1)
            # Since Q is symmetric (Z sym, D_aug sym -> ZDZ sym)
            # Grad_k = - (Q_00 + 2*Q_0,k+1 + Q_k+1,k+1)
            
            # Vectorized computation
            Q_00 = Q[0, 0]
            Q_row0 = Q[0, 1:] # vector of Q_0,k+1
            Q_diag = np.diag(Q)[1:] # vector of Q_k+1,k+1
            
            grad = -(Q_00 + 2 * Q_row0 + Q_diag)
            
            return grad
            
        except linalg.LinAlgError:
            return np.zeros_like(w)


class PEVOptimality(CDMeanOptimality):
    """
    Continuous relaxation of Prediction Error Variance (PEV) optimality.
    This minimizes the mean PEV (A-optimality on the random effects).
    
    This is equivalent to CDMeanOptimality but without 1/G_ii weighting.
    """
    def __init__(self, G_matrix: np.ndarray, lambda_val: float = 1.0, target_indices: Optional[List[int]] = None):
        super().__init__(G_matrix, lambda_val, target_indices)
        
        # Override weights to be 1.0 (mean PEV)
        # D_aug was set in super().__init__
        weights = np.ones(self.n)
        
        if target_indices is not None:
             mask = np.zeros(self.n)
             mask[target_indices] = 1.0
             weights *= mask
             
        self.D_u = np.diag(weights)
        self.D_aug = np.zeros((self.n + 1, self.n + 1))
        self.D_aug[1:, 1:] = self.D_u
class ConvexRelaxationSolver:
    """
    Solves the convex relaxation problem using constraints:
    0 <= w_i <= 1
    sum(w) = k
    """
    def __init__(self, criterion: ContinuousCriterion, method: str = "SLSQP"):
        self.criterion = criterion
        self.method = method
    
    def solve(self, X: Optional[np.ndarray], k: int, w_init: Optional[np.ndarray] = None) -> np.ndarray:
        if X is not None:
            n = X.shape[0]
        elif hasattr(self.criterion, 'n'):
            n = self.criterion.n
        else:
            raise ValueError("X must be provided unless criterion defines 'n'.")
        
        if w_init is None:
            # Start with uniform weights summing to k
            w_init = np.ones(n) * (k / n)
            
        # Constraints
        # eq: sum(w) - k = 0
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - k}]
        
        # Bounds: 0 <= w <= 1
        bounds = [(0, 1) for _ in range(n)]
        
        # Scipy Interface
        fun = lambda w: self.criterion.evaluate(w, X)
        jac = lambda w: self.criterion.gradient(w, X)
        
        if self.method in ["SLSQP", "trust-constr"]:
            res = minimize(
                fun, 
                w_init, 
                method=self.method, 
                jac=jac, 
                bounds=bounds, 
                constraints=constraints,
                options={'disp': False}
            )
            return res.x
            
        elif self.method == "FW": # Frank-Wolfe / Conditional Gradient
            return self._solve_frank_wolfe(X, k, w_init, fun, jac)
            
        else:
            raise ValueError(f"Unknown method {self.method}")

    def _solve_frank_wolfe(self, X: np.ndarray, k: int, w_init: np.ndarray, 
                          fun: Callable, jac: Callable, max_iter: int = 500, tol: float = 1e-5) -> np.ndarray:
        """
        Frank-Wolfe (Conditional Gradient) algorithm.
        Constraint set C: {w | 0 <= w <= 1, sum(w) = k}
        """
        w = w_init.copy()
        n = w.shape[0]
        
        for i in range(max_iter):
            grad = jac(w)
            
            # Linear minimization oracle:
            # minimize <s, grad> s.t. s in C
            # This is solved by setting s_i = 1 for indices corresponding to k smallest gradients
            # and s_i = 0 otherwise.
            # (Note: we want to MOVE against gradient to minimize function, so we look for direction of descent.
            # But FW defines s as argmin <s, grad> to find best direction within feasible set)
            
            # Argmin s^T grad -> pick smallest components of grad
            idx = np.argsort(grad)
            s = np.zeros(n)
            s[idx[:k]] = 1.0 # Set top k smallest gradient components to 1 (since we minimize, generic gradient descent goes opposite to grad, logic holds)
            
            # Check convergence gap
            gap = np.dot(w - s, grad)
            if gap < tol:
                break
                
            # Line search or standard step size
            # Standard step size gamma = 2 / (i + 2)
            gamma = 2.0 / (i + 2.0)
            
            # Optional: Exact line search could be better but expensive
            
            w = (1 - gamma) * w + gamma * s
            
        return w

def discretize(w: np.ndarray, k: int, method: str = 'top_k') -> List[int]:
    """
    Convert continuous weights to discrete selection.
    """
    if method == 'top_k':
        # Select indices with largest weights
        idx = np.argsort(w)[::-1] # descending
        return sorted(list(idx[:k]))
    elif method == 'sample':
        # Sample proportional to weights (normalized)
        # This might not respect constraint exactly if not careful, 
        # but for fixed size sampling generic approach:
        p = w / np.sum(w)
        return sorted(list(np.random.choice(len(w), size=k, replace=False, p=p)))
    else:
        raise ValueError("Unknown discretization method")


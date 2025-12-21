
import numpy as np
import pandas as pd
from trainselpy.core import train_sel, train_sel_control
from scipy.spatial.distance import pdist, squareform

def demo_causal_discovery():
    """
    Scenario 1: Structure Learning (Graph) + parameter estimation.
    We want to find a DAG (Graph) and edge weights (DBL) that explain data.
    """
    print("\n" + "="*60)
    print("SCENARIO 1: Causal Discovery (Graph + Continuous Parameters)")
    print("Problem: Find DAG structure and edge weights matching synthetic data.")
    print("Heads: GRAPH_W (Adjacency) + DBL (Noise Variances)")
    print("="*60)

    # 1. Synthetic Ground Truth
    n_nodes = 5
    # Truth: 0->1, 1->2, 2->3 (Chain)
    true_adj = np.zeros((n_nodes, n_nodes))
    true_adj[0, 1] = 0.8
    true_adj[1, 2] = -0.5
    true_adj[2, 3] = 0.9
    
    # Generate data
    n_samples = 100
    data_X = np.zeros((n_samples, n_nodes))
    noise = np.random.randn(n_samples, n_nodes) * 0.1
    
    # Simple linear SEM generation
    data_X[:, 0] = noise[:, 0]
    data_X[:, 1] = 0.8 * data_X[:, 0] + noise[:, 1]
    data_X[:, 2] = -0.5 * data_X[:, 1] + noise[:, 2]
    data_X[:, 3] = 0.9 * data_X[:, 2] + noise[:, 3]
    data_X[:, 4] = noise[:, 4] # Independent

    emp_cov = np.cov(data_X, rowvar=False)

    def causal_fitness(dbl_vals, data_dict):
        # Scenario 1 has 2 continuous vars: GRAPH_W and DBL.
        # dbl_vals is a list [graph, noise_scales] because len > 1.
        
        flat_graph = dbl_vals[0]
        noise_scales = dbl_vals[1]
        
        n = int(np.sqrt(len(flat_graph)))
        W = flat_graph.reshape(n, n)
        
        try:
            I = np.eye(n)
            # Threshold small weights
            W_thresh = W.copy()
            W_thresh[np.abs(W_thresh) < 0.1] = 0
            
            inv = np.linalg.inv(I - W_thresh)
            Sigma_noise = np.diag(noise_scales * 0.2)
            
            Sigma_est = inv @ Sigma_noise @ inv.T
            
            diff = emp_cov - Sigma_est
            fit_loss = np.sum(diff**2)
            
            l1_loss = np.sum(np.abs(W_thresh))
            
            return -(fit_loss + 0.1 * l1_loss)
            
        except np.linalg.LinAlgError:
            return -1e9 

    # Setup
    candidates = [ list(range(n_nodes)) ] * 2 
    setsizes = [n_nodes*n_nodes, n_nodes]     
    settypes = ["GRAPH_W", "DBL"]
    
    control = train_sel_control(
        npop=50, niterations=20, 
        use_vae=True, vae_lr=0.005, nn_epochs=20, 
        progress=True
    )
    
    result = train_sel(
        candidates=candidates, 
        setsizes=setsizes, 
        settypes=settypes,
        stat=causal_fitness,
        control=control,
        verbose=True
    )
    
    print("Best Fitness:", result.fitness)
    # Result values are flattened list of arrays
    best_W_flat = result.selected_values[0] 
    best_W = np.array(best_W_flat).reshape(n_nodes, n_nodes)
    best_W[np.abs(best_W) < 0.1] = 0
    print("Learned Adjacency Matrix:\n", np.round(best_W, 2))
    print("Ground Truth:\n", true_adj)


def demo_portfolio_clustering():
    """
    Scenario 2: Stratified Portfolio Optimization.
    Group assets into K clusters (PARTITION) and assign weights (SIMPLEX) 
    to maximize return/risk ratio, penalizing intra-cluster correlation.
    """
    print("\n" + "="*60)
    print("SCENARIO 2: Hierarchical Risk Parity / Portfolio (Simple + Partition)")
    print("Problem: Cluster assets and allocate capital to minimize risk.")
    print("Heads: PARTITION (Clustering) + SIMPLEX (Weights)")
    print("="*60)
    
    n_assets = 10
    n_clusters = 3
    
    np.random.seed(42)
    means = np.random.uniform(0.05, 0.15, n_assets)
    cov = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            if abs(i-j) < 3: cov[i,j] = 0.5
            if i==j: cov[i,j] = 1.0
            
    def portfolio_fitness(partition, weights, data_dict):
        # Scenario 2 has 1 int var (PARTITION) and 1 dbl var (SIMPLEX).
        # Arguments are unwrapped arrays directly: partition, weights.
        
        # partition is array of ints
        # weights is array of floats
        
        port_return = np.dot(weights, means)
        port_var = weights.T @ cov @ weights
        
        sharpe = port_return / np.sqrt(port_var)
        
        cluster_weights = np.zeros(n_clusters)
        for i in range(len(partition)):
            c_id = partition[i] % n_clusters 
            cluster_weights[c_id] += weights[i]
            
        balance_penalty = np.std(cluster_weights)
        
        return sharpe - 2.0 * balance_penalty

    candidates_all = [ list(range(n_clusters)), [] ] 
    setsizes_all = [ n_assets, n_assets ]
    settypes_all = ["PARTITION", "SIMPLEX"]
    
    control = train_sel_control(
        npop=50, niterations=15, 
        mutprob=0.1, 
        progress=True
    )
    
    result = train_sel(
        candidates=candidates_all,
        setsizes=setsizes_all,
        settypes=settypes_all,
        stat=portfolio_fitness,
        control=control,
        verbose=True
    )
    
    print("Best Fitness:", result.fitness)
    # selected_indices holds integers (Partition)
    # selected_values holds doubles (Simplex)
    best_part = result.selected_indices[0]
    best_weights = result.selected_values[0]
    
    print("Best Partition:\n", best_part)
    print("Best Weights:\n", np.round(best_weights, 3))


def demo_metric_learning():
    """
    Scenario 3: Metric Learning.
    Learn a Mahalanobis distance matrix M (SPD) and select features (BOOL)
    to maximize class separation.
    """
    print("\n" + "="*60)
    print("SCENARIO 3: Metric Learning (SPD + Feature Selection)")
    print("Problem: Learn metric M and subset of features to separate classes.")
    print("Heads: SPD (Metric) + BOOL (Feature Select)")
    print("="*60)
    
    # Data: 2 Classes, 5 Features. Only first 2 are useful.
    n_samples = 40
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples)
    
    # Make class 1 distinct on feat 0,1
    X[:20, 0] += 2.0
    X[:20, 1] -= 2.0
    y[:20] = 1
    
    def metric_fitness(mask, flat_spd, data_dict):
        # Scenario 3 has 1 int (BOOL) and 1 dbl (SPD).
        # Arguments unwrapped: mask, flat_spd
        
        mask = np.array(mask, dtype=bool)
        if np.sum(mask) == 0: return -1e9 
        
        n = int(np.sqrt(len(flat_spd)))
        M = flat_spd.reshape(n, n)
        
        try:
            X_masked = X * mask.astype(float) 
            
            mu0 = np.mean(X_masked[y==0], axis=0)
            mu1 = np.mean(X_masked[y==1], axis=0)
            
            S_B = np.outer(mu1-mu0, mu1-mu0)
            
            X0 = X_masked[y==0] - mu0
            X1 = X_masked[y==1] - mu1
            S_W = X0.T @ X0 + X1.T @ X1
            
            num = np.trace(M @ S_B)
            den = np.trace(M @ S_W) + 1e-6
            
            sparsity_pen = 0.01 * np.sum(mask)
            
            return (num / den) - sparsity_pen
            
        except:
            return -1e9

    candidates = [ list(range(n_features)), [] ]
    setsizes = [ n_features, n_features * n_features ] 
    settypes = ["BOOL", "SPD"]
    
    control = train_sel_control(
        npop=30, niterations=10, 
        mutprob=0.05,
        progress=True
    )
    
    result = train_sel(
        candidates=candidates,
        setsizes=setsizes, 
        settypes=settypes,
        stat=metric_fitness,
        control=control,
        verbose=True
    )
    
    print("Best Fitness:", result.fitness)
    # selected_indices holds integers/bools (Feature Mask)
    # selected_values holds doubles (SPD)
    print("Selected Features:", result.selected_indices[0])
    M = np.array(result.selected_values[0]).reshape(n_features, n_features)
    print("Learned Metric Diagonal:\n",  np.diag(M))


if __name__ == "__main__":
    demo_causal_discovery()
    demo_portfolio_clustering()
    demo_metric_learning()

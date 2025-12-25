# EvoSolve: General Purpose Evolutionary Optimization Framework

A general-purpose hybrid optimizer combining Genetic Algorithms (GA), Simulated Annealing (SANN), and modern Neural enhancements. Originally derived from the EvoSolve project, EvoSolve is designed to solve a wide range of complex optimization problems involving subset selection, integer decisions, continuous variables, graph structures, and manifolds.

## Overview

EvoSolve provides a flexible, powerful framework for solving difficult optimization problems. Its hybrid engine combines the global search capabilities of Genetic Algorithms with the local refinement of Simulated Annealing, enhanced by modern techniques like surrogate modeling and distributional optimization.

It handles diverse decision variables simultaneously:
- **Discrete**: Subset selection (ordered/unordered), Integer variables, Boolean vectors.
- **Continuous**: Real-valued vectors, Probability Simplices, SPD Matrices.
- **Structural**: Graph and network structure learning, Clustering/Partitioning.

Applications include:
- **Feature Selection & AutoML**: Selecting optimal subsets of features or hyperparameters.
- **Genomic Prediction**: Optimizing training populations (CDMean, D-optimality).
- **Portfolio Optimization**: Allocating assets with complex constraints.
- **Experimental Design**: Optimal selection of samples or treatments.
- **Causal Discovery**: Learning graph structures from data.
- **Multi-Objective Problems**: Finding Pareto-optimal trade-offs (NSGA-II/NSGA-III).

## Installation

### From PyPI (Coming Soon)

```bash
pip install evosolve
```

### From Source

```bash
git clone https://github.com/denizakdemir/evosolve.git
cd evosolve
pip install -e .
```

### Optional Dependencies

For R data conversion:
```bash
pip install rpy2
```

For enhanced plotting:
```bash
pip install seaborn
```

## Key Features

- **Comprehensive Decision Variable Types**:
  - **Core Types**: Continuous (`DBL`), Binary (`BOOL`), Subsets (`UOS`, `OS`), Multisets (`UOMS`, `OMS`)
  - **Graph/Structure Optimization**: Weighted graphs (`GRAPH_W`), Unweighted graphs (`GRAPH_U`)
  - **Manifold-Constrained**: SPD matrices (`SPD`), Probability simplex (`SIMPLEX`)
  - **Clustering/Partitioning**: Group assignment optimization (`PARTITION`)
  - **Mixed Forms**: Arbitrary combinations of variable types in single optimization
  - **Multi-Objective**: Pareto front optimization with NSGA-II/NSGA-III
  
  See [CAPABILITIES.md](CAPABILITIES.md) for complete documentation.

- **Built-in Optimization Criteria**:
  - CDMean: For mixed models, optimizing prediction accuracy
  - CDMean Target: CDMean focused on a specific target set of individuals
  - D-optimality: Maximizing the determinant of the information matrix
  - A-optimality: Minimizing the average variance (trace of the inverse information matrix)
  - E-optimality: Minimizing the worst-case variance (maximum eigenvalue of the inverse information matrix)
  - PEV: Minimizing prediction error variance
  - Maximin: Maximizing the minimum distance between selected samples
  - Coverage: Minimizing the maximum distance from any candidate to the nearest selected sample

- **Advanced Optimization Algorithms**:
  - Genetic Algorithm (GA): Population-based optimization with specialized operators for each variable type
  - Simulated Annealing (SANN): Fine-tuning solutions
  - Island Model: Multiple populations evolving in parallel
  - Multi-objective optimization (NSGA-II/NSGA-III) with diverse Pareto front solutions
  - Distributional head with NSGA selection on distribution means (`dist_use_nsga_means`) to avoid scalarization collapse
  - **Neural Network-Enhanced Optimization**: VAE and GAN for learning solution distributions
  - **CMA-ES Integration**: Covariance Matrix Adaptation for continuous optimization
  - **Surrogate Models**: Gaussian Process models for expensive fitness functions

- **Constraint Handling**:
  - Automatic repair mechanisms for manifold-constrained variables (SPD, SIMPLEX)
  - Graph structure preservation during genetic operations
  - Custom repair functions for domain-specific constraints

- **Parallelization**:
  - Multi-core parallel fitness evaluation
  - Island model parallelization
  - GPU acceleration for neural network components (when available)

## Basic Usage

```python
import numpy as np
from evosolve import make_data, train_sel, set_control_default

# Load example data
from evosolve.data import wheat_data

# Create the EvoSolve data object
ts_data = make_data(M=wheat_data["M"])

# Set control parameters
control = set_control_default()
control["niterations"] = 10

# Run the selection algorithm
result = train_sel(
    data=ts_data,
    candidates=[list(range(200))],  # Select from first 200 lines
    setsizes=[50],                  # Select 50 lines
    settypes=["UOS"],              # Unordered set
    stat=None,                     # Use default CDMean
    control=control
)

print("Selected indices:", result.selected_indices)
print("Fitness value:", result.fitness)
```

## Examples

### D-optimality Criterion

```python
from evosolve import dopt

# Add feature matrix to data
ts_data["FeatureMat"] = wheat_data["M"]

# Run with D-optimality criterion
result = train_sel(
    data=ts_data,
    candidates=[list(range(200))],
    setsizes=[50],
    settypes=["UOS"],
    stat=dopt,  # Use D-optimality
    control=control
)
```

### Parallel Processing

```python
# Set control parameters for parallel processing
control = train_sel_control(
    size="demo",
    niterations=50,
    npop=200,
    nislands=4,      # Use 4 islands
    parallelizable=True,
    mc_cores=4       # Use 4 cores
)

# Run with parallel processing
result = train_sel(
    data=ts_data,
    candidates=[list(range(200))],
    setsizes=[50],
    settypes=["UOS"],
    stat=None,     # Use default CDMean
    control=control,
    n_jobs=4       # Use 4 parallel jobs
)
```

### Multi-objective Optimization with Diverse Solutions

```python
from evosolve import cdmean_opt, dopt

# Define a multi-objective function
def multi_objective(solution, data):
    cdmean_value = cdmean_opt(solution, data)
    dopt_value = dopt(solution, data)
    return [cdmean_value, dopt_value]

# Run with multi-objective optimization and ensure diverse solutions
control = train_sel_control(
    niterations=100,
    npop=500,
    solution_diversity=True  # Ensure unique solutions on Pareto front
)

result = train_sel(
    data=ts_data,
    candidates=[list(range(200))],
    setsizes=[50],
    settypes=["UOS"],
    stat=multi_objective,
    n_stat=2,       # 2 objectives
    control=control
)

# Plot the Pareto front
from evosolve.utils import plot_pareto_front
plot_pareto_front(
    result.pareto_front,
    obj_names=["CDMean", "D-optimality"],
    title="Pareto Front: CDMean vs D-optimality",
    output_file="pareto_front.png"
)
```

### Multiple Sets with Different Types

```python
# Define a custom fitness function
def custom_fitness(int_solutions, data):
    # Calculate fitness based on both sets
    set1 = int_solutions[0]  # First set (UOS)
    set2 = int_solutions[1]  # Second set (OS)
    
    # Implement your own fitness calculation
    return some_fitness_measure(set1, set2, data)

# Run with multiple sets
result = train_sel(
    data=ts_data,
    candidates=[list(range(100)), list(range(100, 200))],  # Two candidate sets
    setsizes=[30, 20],                     # Different sizes
    settypes=["UOS", "OS"],               # Different types
    stat=custom_fitness,
    control=control
)
```

### Graph Structure Learning (New!)

```python
# Optimize a causal graph structure and edge weights
def causal_fitness(dbl_vals, data):
    """Learn DAG structure from observational data"""
    flat_graph = dbl_vals[0]  # GRAPH_W adjacency
    noise_vars = dbl_vals[1]   # DBL noise variances
    
    n = int(np.sqrt(len(flat_graph)))
    W = flat_graph.reshape(n, n)
    
    # Score based on fit to empirical covariance
    # (see examples/universal_optimization_demo.py for full implementation)
    return -dag_loss(W, noise_vars, data)

result = train_sel(
    candidates=[list(range(5))] * 2,  # 5 nodes
    setsizes=[25, 5],                  # 5×5 graph + 5 noise vars
    settypes=["GRAPH_W", "DBL"],
    stat=causal_fitness,
    control=train_sel_control(npop=50, niterations=20, use_vae=True)
)
```

### Metric Learning with Feature Selection (New!)

```python
# Learn Mahalanobis metric and select features jointly
def metric_fitness(mask, flat_spd, data):
    """Optimize metric for class separation"""
    n = int(np.sqrt(len(flat_spd)))
    M = flat_spd.reshape(n, n)  # SPD metric matrix
    
    # Compute Fisher discriminant ratio with selected features
    X_masked = data['X'] * mask.astype(float)
    # ... compute between/within scatter matrices ...
    return fisher_ratio(M, X_masked, data['y'])

result = train_sel(
    candidates=[list(range(n_features)), []],
    setsizes=[n_features, n_features * n_features],
    settypes=["BOOL", "SPD"],
    stat=metric_fitness,
    control=train_sel_control(npop=30, niterations=10)
)
```

### Portfolio Optimization with Clustering (New!)

```python
# Cluster assets and allocate weights simultaneously
def portfolio_fitness(partition, weights, data):
    """Maximize Sharpe ratio with balanced clusters"""
    returns = np.dot(weights, data['means'])
    risk = np.sqrt(weights.T @ data['cov'] @ weights)
    sharpe = returns / risk
    
    # Penalize imbalanced clusters
    cluster_balance = compute_balance_penalty(partition, weights)
    
    return sharpe - cluster_balance

result = train_sel(
    candidates=[list(range(n_clusters)), []],
    setsizes=[n_assets, n_assets],
    settypes=["PARTITION", "SIMPLEX"],  # Clustering + Weights
    stat=portfolio_fitness,
    control=train_sel_control(npop=50, niterations=15)
)
```

## Converting R Data

If you have the original WheatData from the R package, you can convert it:

```python
from evosolve.utils import r_data_to_python

# Convert R data to Python format
python_data = r_data_to_python("path/to/WheatData.rda", "wheat_data.pkl")

# Load the converted data
import pickle
with open("wheat_data.pkl", "rb") as f:
    wheat_data = pickle.load(f)
```

## Full Documentation

For more details and advanced usage, see the examples directory and the API documentation.

## Requirements

- numpy>=1.19.0
- scipy>=1.5.0
- pandas>=1.0.0
- scikit-learn>=0.23.0
- matplotlib>=3.2.0
- joblib>=0.16.0
- torch>=1.7.0

Optional:
- rpy2 (for converting R data)
- seaborn (for advanced plotting)

## License

MIT License

## Acknowledgments

EvoSolve is a Python implementation derived from the concepts of the TrainSel R package. The original R package was written by Deniz Akdemir, Julio Isidro Sanchez, Simon Rio and Javier Fernandez-Gonzalez. This Python evolution was developed by Deniz Akdemir.

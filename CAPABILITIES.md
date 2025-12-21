# TrainSelPy Capabilities

## Decision Variable Types

TrainSelPy supports a comprehensive range of decision variable types for diverse optimization problems.

### Core Variable Types (Legacy)

#### 1. Continuous Vectors (`DBL`)
- **Description**: Real-valued continuous variables
- **Representation**: `numpy.ndarray` of floats
- **Use Cases**: Parameter optimization, feature weights, regression coefficients
- **Example**: `settypes=["DBL"]`, `setsizes=[10]` creates 10 continuous variables in [0, 1]
- **Status**: ✅ Fully Implemented

#### 2. Integers (`INT`)
- **Description**: Integer-valued variables with bounded ranges
- **Representation**: `numpy.ndarray` of integers
- **Candidates Format**: `[min_val, max_val]` for bounds, `[max_val]` for [0, max_val], or `[]` for [0, 100]
- **Use Cases**: Discrete parameters, counts, resource allocation, hyperparameter tuning
- **Example**: `settypes=["INT"]`, `candidates=[[0, 50]]`, `setsizes=[10]` creates 10 integers in [0, 50]
- **Status**: ✅ Fully Implemented
  - ✅ Initialization with configurable bounds
  - ✅ Mutation with range-scaled integer deltas and boundary clipping
  - ✅ Crossover with multi-point recombination and boundary clipping
  - ✅ Neural network (VAE/GAN) support via continuous normalization

#### 3. Binary Vectors (`BOOL`)
- **Description**: Binary selection variables (0/1)
- **Representation**: `numpy.ndarray` of {0, 1}
- **Use Cases**: Feature selection, on/off decisions, binary masks
- **Example**: `settypes=["BOOL"]`, `candidates=[list(range(100))]` creates binary mask for 100 features
- **Status**: ✅ Fully Implemented

#### 4. Subsets (`UOS`, `OS`)
- **UOS (Unordered Set)**: Subsets where order doesn't matter
  - **Example**: Selecting k items from n candidates
  - **Use Cases**: Feature selection, sample selection, team formation
- **OS (Ordered Set)**: Subsets where order matters
  - **Example**: Selecting and ranking top k items
  - **Use Cases**: Rankings, prioritization, sequential decisions
- **Status**: ✅ Fully Implemented

#### 5. Multisets (`UOMS`, `OMS`)
- **UOMS (Unordered Multiset)**: Subsets with repetitions, order doesn't matter
- **OMS (Ordered Multiset)**: Subsets with repetitions, order matters
- **Use Cases**: Resource allocation with repetition, portfolio optimization
- **Status**: ✅ Fully Implemented

#### 6. Permutations
- **Description**: Can be represented using `OS` (Ordered Set) where all n items are selected
- **Use Cases**: TSP, scheduling, routing problems
- **Example**: `settypes=["OS"]`, `setsizes=[n]` with `candidates=[list(range(n))]`
- **Status**: ✅ Fully Implemented (via `OS`)

### Advanced Variable Types (New Universal Optimization Heads)

#### 7. Graph / Structure Optimization (`GRAPH_W`, `GRAPH_U`)
- **GRAPH_W (Weighted Graphs)**: Adjacency matrices with continuous edge weights
  - **Representation**: Flattened adjacency matrix (size `n×n`) stored as continuous vector
  - **Constraints**: Values in [0, 1], can be thresholded for sparsity
  - **Use Cases**: Causal discovery, neural architecture search, network design, gene regulatory networks
- **GRAPH_U (Unweighted Graphs)**: Binary adjacency matrices
  - **Representation**: Flattened adjacency matrix stored as binary vector
  - **Constraints**: Binary {0, 1} values
  - **Use Cases**: Structure learning, interaction networks, connectivity optimization
- **Mutations**: 
  - `GRAPH_W`: Gaussian perturbations with boundary clipping
  - `GRAPH_U`: Bit-flip mutations
- **Neural Network Support**: 
  - `GRAPH_W`: Treated as continuous variables, clipped to [0, 1] after generation
  - `GRAPH_U`: Treated as binary variables
- **Example**:
  ```python
  # Learn a 5-node weighted graph
  settypes = ["GRAPH_W"]
  setsizes = [25]  # 5×5 = 25 edges
  candidates = [list(range(5))]  # Dummy candidates
  ```
- **Status**: ✅ Fully Implemented

#### 8. Manifold-Constrained Continuous Head (`SPD`, `SIMPLEX`)
- **SPD (Symmetric Positive Definite Matrices)**: 
  - **Description**: Matrices constrained to the SPD manifold
  - **Representation**: Flattened matrices stored as continuous vectors
  - **Constraints**: Symmetry (M = M^T) and positive definiteness (all eigenvalues > 0)
  - **Repair Mechanism**: After mutation/crossover/neural generation, matrices are symmetrized and projected to SPD cone via eigenvalue clipping
  - **Use Cases**: Covariance matrices, metric learning (Mahalanobis distance), kernel matrices, precision matrices
- **SIMPLEX (Probability Simplex)**:
  - **Description**: Vectors constrained to sum to 1 with non-negative entries
  - **Representation**: Flat vectors of size `n`
  - **Constraints**: All entries ≥ 0, sum(entries) = 1
  - **Repair Mechanism**: Clamp negative values to 0, then re-normalize
  - **Use Cases**: Probability distributions, portfolio weights, mixture coefficients
- **Mutations**:
  - `SPD`: Perturb, then project back to SPD manifold
  - `SIMPLEX`: Perturb with Dirichlet-style noise, re-normalize
- **Neural Network Support**: Explicit repair after VAE/GAN decoding
- **Example**:
  ```python
  # Optimize a 3×3 covariance matrix
  settypes = ["SPD"]
  setsizes = [9]  # 3×3 matrix
  
  # Optimize portfolio weights for 10 assets
  settypes = ["SIMPLEX"]
  setsizes = [10]
  ```
- **Status**: ✅ Fully Implemented

#### 9. Partition / Clustering Head (`PARTITION`)
- **Description**: Assign items to groups/clusters
- **Representation**: Integer array where each entry is a group ID
- **Constraints**: Group IDs must be within valid range (0 to K-1 for K groups)
- **Mutations**: Randomly reassign group memberships
- **Neural Network Support**: Treated as permutation-like discrete structure
- **Use Cases**: Clustering, community detection, stratified sampling, hierarchical optimization
- **Example**:
  ```python
  # Partition 100 items into 5 groups
  settypes = ["PARTITION"]
  setsizes = [100]  # 100 items to partition
  candidates = [list(range(5))]  # 5 possible groups (0-4)
  ```
- **Status**: ✅ Fully Implemented

### Mixed Forms

TrainSelPy supports **arbitrary combinations** of decision variable types in a single optimization problem.

**Example: Mixed-Type Optimization**
```python
# Combine feature selection (BOOL) + hyperparameters (DBL) + architecture (GRAPH_W)
settypes = ["BOOL", "DBL", "GRAPH_W"]
setsizes = [100, 5, 16]  # 100 features, 5 hyperparams, 4×4 graph
candidates = [list(range(100)), [], list(range(4))]

def mixed_fitness(mask, hyperparams, graph, data):
    # Fitness function using all three variable types
    ...
```

**Status**: ✅ Fully Implemented

### Multi-Task / Multi-Objective Optimization

TrainSelPy supports multi-objective optimization with Pareto front discovery.

**Features**:
- NSGA-II and NSGA-III algorithms
- Automatic Pareto front construction
- Crowding distance for diversity
- Solution uniqueness enforcement

**Example**:
```python
def multi_objective(solution, data):
    obj1 = compute_accuracy(solution, data)
    obj2 = compute_sparsity(solution, data)
    return [obj1, obj2]

result = train_sel(
    ...,
    stat=multi_objective,
    n_stat=2,  # 2 objectives
    control=control
)

# Access Pareto front
pareto_front = result.pareto_front
pareto_solutions = result.pareto_solutions
```

**Status**: ✅ Fully Implemented

## Advanced Optimization Features

### Neural Network-Enhanced Optimization
- **VAE (Variational Autoencoder)**: Learns latent representations of high-quality solutions
- **GAN (Generative Adversarial Network)**: Generates novel candidate solutions
- **Automatic Type Handling**: Decision structure automatically inferred from `settypes`
- **Post-Generation Repair**: Solutions are repaired to satisfy hard constraints (e.g., SPD, SIMPLEX)

### CMA-ES Integration
- Continuous variable optimization via Covariance Matrix Adaptation Evolution Strategy
- Automatically enabled for problems with continuous variables

### Surrogate Models
- Gaussian Process surrogate for expensive fitness functions
- Pre-screening with surrogate before expensive evaluation

### Island Model
- Multiple independent populations evolving in parallel
- Periodic migration between islands
- Enhanced exploration of solution space

## Summary Table

| Variable Type | Symbol | Implemented | Neural Network Support | Use Cases |
|--------------|--------|-------------|----------------------|-----------|
| Continuous | `DBL` | ✅ | ✅ | Parameter optimization |
| Integers | `INT` | ✅ | ✅ | Bounded integer optimization |
| Binary | `BOOL` | ✅ | ✅ | Feature selection |
| Unordered Set | `UOS` | ✅ | ✅ | Subset selection |
| Ordered Set | `OS` | ✅ | ✅ | Rankings, permutations |
| Unordered Multiset | `UOMS` | ✅ | ✅ | Resource allocation |
| Ordered Multiset | `OMS` | ✅ | ✅ | Sequential allocation |
| Weighted Graph | `GRAPH_W` | ✅ | ✅ | Causal discovery, networks |
| Unweighted Graph | `GRAPH_U` | ✅ | ✅ | Structure learning |
| SPD Matrix | `SPD` | ✅ | ✅ | Metric learning, covariance |
| Probability Simplex | `SIMPLEX` | ✅ | ✅ | Portfolio, distributions |
| Partition/Clustering | `PARTITION` | ✅ | ✅ | Clustering, grouping |
| **Mixed Forms** | Multiple | ✅ | ✅ | Complex real-world problems |
| **Multi-Objective** | `n_stat > 1` | ✅ | ✅ | Pareto optimization |

## Notes

- ✅ **Integer (`INT`) Type**: Fully implemented with:
  - Random initialization in [min, max] range via `candidates=[min, max]`
  - Mutation with integer deltas scaled by range and boundary clipping
  - Multi-point crossover with boundary clipping
  - Neural network support via continuous normalization to [0, 1]
  - Example usage:
    ```python
    # INT variables from 0 to 100
    settypes=["INT"], candidates=[[0, 100]], setsizes=[10]
    
    # INT variables from -50 to 50
    settypes=["INT"], candidates=[[-50, 50]], setsizes=[5]
    ```
  
- All new types (`GRAPH_W`, `GRAPH_U`, `SPD`, `SIMPLEX`, `PARTITION`, `INT`) are compatible with:
  - Standard genetic operators (crossover, mutation)
  - Neural network-enhanced generation (VAE/GAN)
  - Multi-objective optimization
  - Mixed-type problems

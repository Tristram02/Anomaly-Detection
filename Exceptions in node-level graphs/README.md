# Anomaly Detection - Node-Level Graph Anomalies

## Objective

The primary objective of this assignment was to implement and evaluate anomaly detection methods for node-level anomalies in graph-structured data using graph embeddings and classical machine learning detectors.

1. **Graph Data Processing:**

   - Loading and analyzing the Cora citation network dataset
   - Computing graph statistics (degree distribution, clustering coefficient, density)
   - Visualizing graph properties

2. **Anomaly Injection:**

   - Synthetic anomaly generation through edge rewiring
   - Controlled anomaly ratio and rewiring parameters
   - Ground truth labels for evaluation

3. **Graph Embedding Generation:**

   - Node2Vec algorithm for learning node representations
   - Hyperparameter exploration (dimensions, p, q parameters)
   - Random walk-based feature extraction

4. **Anomaly Detection Methods:**

   - Local Outlier Factor (LOF) on graph embeddings
   - Isolation Forest on graph embeddings
   - Comparison of detection performance

5. **Comprehensive Evaluation:**
   - Multiple metrics: PR-AUC, ROC-AUC, F1-Score, Recall
   - Analysis of anomaly characteristics (degree, centrality)
   - Visualization of embeddings and detection results

## Project Structure

```
Exceptions in node-level graphs/
├── main.py                                    # Main implementation script
├── cora/                                      # Cora dataset
│   ├── cora.content                          # Node features and labels
│   ├── cora.cites                            # Citation edges
│   └── README                                # Dataset description
├── results/                                   # Experiment results
│   ├── metrics.csv                           # Comparative metrics
│   ├── metrics.json                          # Detailed results in JSON
│   └── figures/                              # Generated visualizations
└── .gitignore                                # Git ignore file
```

> **Note:** This repository contains only the source code (`main.py`) and the required dataset (`cora/` directory). All other files (results and plots) are generated outputs from running the experiments.

## Key Features

### 1. Cora Citation Network Analysis

- **Dataset:** Academic paper citation network
- **Nodes:** 2,708 scientific publications
- **Edges:** 5,429 citation links
- **Features:** 1,433-dimensional binary word vectors
- **Categories:** 7 publication classes

### 2. Synthetic Anomaly Injection

Controlled anomaly generation through structural modification:

- **Edge Rewiring:** Randomly rewiring a percentage of edges for selected nodes
- **Anomaly Ratio:** Configurable percentage of anomalous nodes (default: 10%)
- **Rewiring Ratio:** Percentage of edges to rewire per anomalous node (default: 50%)
- **Ground Truth:** Binary labels for evaluation

### 3. Node2Vec Graph Embeddings

Random walk-based node representation learning:

- **Dimensions:** Configurable embedding size (32, 64, 128)
- **Walk Parameters:**
  - `p`: Return parameter (controls likelihood of returning to previous node)
  - `q`: In-out parameter (controls exploration vs. exploitation)
- **Walk Length:** 80 steps per walk
- **Number of Walks:** 10 walks per node
- **Exploration Strategies:**
  - BFS-like (p=0.5, q=2): Local neighborhood exploration
  - DFS-like (p=2, q=0.5): Broader network exploration
  - Balanced (p=1, q=1): Unbiased random walks

### 4. Anomaly Detection Algorithms

**Local Outlier Factor (LOF):**

- Density-based anomaly detection
- Compares local density of a node to its neighbors
- Identifies nodes in low-density regions
- Configurable neighborhood size (default: 20 neighbors)

**Isolation Forest:**

- Ensemble-based anomaly detection
- Isolates anomalies through random partitioning
- Anomalies require fewer splits to isolate
- Efficient for high-dimensional embeddings

### 5. Comprehensive Evaluation Pipeline

**Metrics:**

- **PR-AUC:** Precision-Recall Area Under Curve
- **ROC-AUC:** Receiver Operating Characteristic AUC
- **F1-Score:** Harmonic mean of precision and recall
- **Recall:** True positive rate

**Analysis:**

- Degree distribution comparison (normal vs. anomalous nodes)
- Betweenness centrality analysis
- 2D PCA visualization of embeddings
- Anomaly score distributions

## Usage

```bash
# Run complete experiment pipeline
python main.py
```

The script will:

1. Load the Cora citation network
2. Compute and visualize graph statistics
3. Inject synthetic anomalies
4. Generate Node2Vec embeddings with multiple configurations
5. Run LOF and Isolation Forest detectors
6. Evaluate performance and save results
7. Generate visualizations

## Experimental Configurations

The implementation tests multiple Node2Vec configurations:

| Config | Dimensions | p   | q   | Strategy |
| ------ | ---------- | --- | --- | -------- |
| 1      | 64         | 1.0 | 1.0 | Balanced |
| 2      | 64         | 0.5 | 2.0 | BFS-like |
| 3      | 64         | 2.0 | 0.5 | DFS-like |
| 4      | 32         | 1.0 | 1.0 | Low-dim  |
| 5      | 128        | 1.0 | 1.0 | High-dim |

Each configuration is evaluated with both LOF and Isolation Forest detectors.

## Implementation Details

### Anomaly Injection Process

```
For each selected anomalous node:
1. Identify all incoming and outgoing edges
2. Select a percentage of edges to rewire (rewiring_ratio)
3. For each selected edge:
   - Remove the original edge
   - Create a new edge to/from a random node
   - Ensure no duplicate edges are created
```

This process creates structural anomalies that deviate from normal citation patterns.

### Node2Vec Embedding Process

```
1. Generate random walks starting from each node
2. Apply Skip-Gram model to learn embeddings
3. Optimize embeddings to preserve walk co-occurrence
4. Result: Dense vector representation for each node
```

### Detection Process

```
1. Train detector on node embeddings
2. Compute anomaly scores for all nodes
3. Apply threshold (90th percentile) for binary classification
4. Evaluate against ground truth labels
```

## Visualizations

The pipeline generates several visualizations:

1. **Degree Distribution:** Histogram and log-log plot of node degrees
2. **Embedding Visualization:** 2D PCA projection showing:
   - True anomaly labels (red X markers)
   - Normal nodes (blue dots)
   - Anomaly scores (color-coded heatmap)
3. **Metrics Comparison:** Line plots comparing detector performance across configurations

## Dependencies

```
networkx
numpy
pandas
matplotlib
seaborn
node2vec
scikit-learn
```

## Results

Results are saved in multiple formats:

- **CSV (`metrics.csv`):** Tabular comparison of all configurations and detectors
- **JSON (`metrics.json`):** Detailed results including graph statistics and characteristics
- **PNG images:** Visualizations of degree distributions, embeddings, and metrics

All results can be found in the `results/` directory after running the experiment.

## Key Findings

The comparison reveals important insights:

1. **Embedding Quality:** Node2Vec successfully captures graph structure in low-dimensional space
2. **Detection Performance:** Both LOF and Isolation Forest effectively identify structural anomalies
3. **Parameter Sensitivity:** The p and q parameters significantly affect embedding quality and detection performance
4. **Dimension Trade-offs:** Higher dimensions capture more information but may include noise

## Author

Kamil Włodarczyk

# Anomaly Detection - Course Projects

This repository contains a comprehensive collection of anomaly detection projects exploring various data types, methodologies, and algorithms. The projects demonstrate the application of both classical machine learning and deep learning approaches to detect anomalies across different data modalities.

## Overview

Anomaly detection is the identification of rare items, events, or observations that deviate significantly from the majority of the data. This repository covers three fundamental domains of anomaly detection:

1. **Tabulated (Multidimensional) Data** - Classical methods on structured datasets
2. **Time Series Data** - Comparison of classical and deep learning approaches
3. **Graph-Structured Data** - Node-level anomaly detection in networks

## Projects

### 1. Tabulated (Multidimensional) Data: A Comparison of Classical Methods

**Focus:** Classical anomaly detection algorithms on structured, multidimensional datasets

**Key Techniques:**

- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM (OCSVM)
- Fuzzy Local Outlier Factor (custom implementation)

**Datasets:** Cardio, Musk, Vowels (MAT format)

**Highlights:**

- Custom Fuzzy LOF algorithm combining fuzzy logic with density-based detection
- Comprehensive hyperparameter tuning with grid search
- Cross-validation with stratified K-fold
- Multiple preprocessing strategies (standard scaling, robust scaling)
- Extensive visualization (PR curves, ROC curves, heatmaps)

**[View Project →](Tabulated%20%28multidimensional%29%20data%3A%20a%20comparison%20of%20classical%20methods/)**

---

### 2. Time Series Data: A Comparison of Classical and Deep Methods

**Focus:** Comparing classical machine learning and deep learning for time series anomaly detection

**Key Techniques:**

- **Classical:** Isolation Forest
- **Deep Learning:** Convolutional 1D Autoencoder

**Dataset:** UCR Time Series Archive (500 files)

**Highlights:**

- Custom Conv1D Autoencoder architecture with encoder-decoder structure
- Batch normalization, dropout, and early stopping
- Learning rate scheduling and GPU acceleration
- Reconstruction error-based anomaly detection
- Side-by-side comparison of classical vs. deep learning performance
- Modular architecture with reusable components

**[View Project →](Time%20Series%20Data%3A%20A%20Comparison%20of%20Classical%20and%20Deep%20Methods/)**

---

### 3. Exceptions in Node-Level Graphs

**Focus:** Detecting anomalous nodes in graph-structured data using embeddings

**Key Techniques:**

- Node2Vec graph embeddings
- Local Outlier Factor (LOF) on embeddings
- Isolation Forest on embeddings

**Dataset:** Cora citation network (2,708 nodes, 5,429 edges)

**Highlights:**

- Synthetic anomaly injection through edge rewiring
- Node2Vec with configurable exploration strategies (BFS-like, DFS-like, balanced)
- Hyperparameter exploration (dimensions, p, q parameters)
- Graph statistics and centrality analysis
- 2D PCA visualization of embeddings
- Comparison of detection performance across configurations

**[View Project →](Exceptions%20in%20node-level%20graphs/)**

---

## Common Themes

### Evaluation Metrics

All projects use comprehensive evaluation metrics:

- **PR-AUC** (Precision-Recall Area Under Curve)
- **ROC-AUC** (Receiver Operating Characteristic AUC)
- **F1-Score** (Harmonic mean of precision and recall)
- **Recall** (True positive rate)
- **Balanced Accuracy**
- **Matthews Correlation Coefficient (MCC)**

### Methodologies

**Classical Methods:**

- Isolation Forest: Ensemble-based isolation through random partitioning
- Local Outlier Factor: Density-based detection comparing local densities
- One-Class SVM: Support vector-based boundary learning

**Deep Learning:**

- Autoencoders: Reconstruction error-based anomaly detection
- Convolutional architectures: Temporal pattern learning

**Graph Methods:**

- Node2Vec: Random walk-based graph embeddings
- Structural analysis: Degree and centrality-based features

### Visualization

Each project includes extensive visualizations:

- Precision-Recall curves
- ROC curves
- Model comparison charts
- Heatmaps and distribution plots
- Embedding visualizations (for graph data)
- Parameter effect analysis

## Repository Structure

```
.
├── Tabulated (multidimensional) data: a comparison of classical methods/
│   ├── main.py
│   ├── README.md
│   └── [datasets and results]
├── Time Series Data: A Comparison of Classical and Deep Methods/
│   ├── src/
│   ├── scripts/
│   ├── README.md
│   └── [datasets and results]
├── Exceptions in node-level graphs/
│   ├── main.py
│   ├── README.md
│   └── [datasets and results]
└── README.md
```

> **Note:** Each project directory contains only source code and datasets. Results, plots, and reports are generated outputs and are excluded from the repository.

## Technologies Used

### Core Libraries

- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms and metrics
- **SciPy** - Scientific computing

### Visualization

- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical data visualization

### Deep Learning

- **PyTorch** - Deep learning framework
- **TensorFlow/Keras** - Neural network training

### Graph Processing

- **NetworkX** - Graph creation and analysis
- **Node2Vec** - Graph embedding generation

## Key Learnings

### 1. Data Modality Matters

Different data types require different approaches:

- **Tabulated data:** Classical methods excel with proper preprocessing
- **Time series:** Deep learning captures temporal dependencies better
- **Graphs:** Embeddings transform structural information into feature space

### 2. Trade-offs

- **Classical methods:** Faster, more interpretable, require less data
- **Deep learning:** Better for complex patterns, requires more data and computation
- **Ensemble methods:** Often provide robust baseline performance

### 3. Evaluation is Critical

- Multiple metrics provide comprehensive performance assessment
- PR-AUC is particularly important for imbalanced datasets
- Visualization helps understand model behavior and failure modes

### 4. Preprocessing Impact

- Scaling strategies significantly affect performance
- Missing value handling is crucial
- Feature engineering can improve classical methods

## Running the Projects

Each project can be run independently. Navigate to the project directory and follow the instructions in the respective README:

```bash
# Tabulated data project
cd "Tabulated (multidimensional) data: a comparison of classical methods"
python main.py

# Time series project
cd "Time Series Data: A Comparison of Classical and Deep Methods"
python scripts/main.py --all

# Graph anomaly detection
cd "Exceptions in node-level graphs"
python main.py
```

## Author

**Kamil Włodarczyk**

---

## License

This repository is for educational purposes as part of university coursework.

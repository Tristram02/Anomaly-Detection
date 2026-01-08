# Anomaly Detection - Time Series Anomaly Detection

The primary objective of this assignment was to implement and compare classical machine learning and deep learning approaches for time series anomaly detection.

1. **Implementation of Classical Method:**

   - Isolation Forest for anomaly detection on time series data
   - Hyperparameter tuning using grid search
   - Comprehensive evaluation on UCR time series datasets

2. **Implementation of Deep Learning Method:**

   - Convolutional 1D Autoencoder for time series reconstruction
   - Custom architecture with encoder-decoder structure
   - Latent space representation learning
   - Threshold-based anomaly detection using reconstruction error

3. **Comprehensive Comparison:**

   - Side-by-side evaluation of both methods
   - Multiple evaluation metrics: Precision, Recall, F1-Score, ROC-AUC, PR-AUC
   - Visualization of results including ROC curves and reconstruction plots

4. **Modular Architecture:**

   - Separate modules for data loading, model implementation, evaluation, and visualization
   - Reusable components for different time series datasets
   - Command-line interface for running experiments

5. **Extensive Evaluation:**
   - Training/validation/test split
   - Early stopping and learning rate scheduling for deep learning
   - Cross-validation for hyperparameter tuning
   - Detailed performance metrics and visualizations

## Project Structure

```
Time Series Data: A Comparison of Classical and Deep Methods/
├── src/
│   ├── __init__.py
│   ├── autoencoder.py
│   ├── isolation_forest.py
│   ├── data_loader.py
│   ├── evaluation.py
│   └── visualization.py
├── scripts/
│   ├── main.py
│   ├── run_autoencoder.py
│   └── run_isolation_forest.py
├── Dataset/
├── results/
│   ├── model_comparison.csv
│   ├── models/
│   ├── metrics/
│   └── figures/
```

> **Note:** This repository contains only the source code (`src/` and `scripts/` directories) and the required datasets (`Dataset/` directory). All other files (results, models, plots, and reports) are generated outputs from running the experiments.

## Key Features

### 1. Convolutional 1D Autoencoder

A custom deep learning architecture for time series anomaly detection:

- **Encoder:** 3 convolutional layers (32→64→128 filters) with batch normalization, ReLU activation, max pooling, and dropout
- **Latent Space:** Fully connected bottleneck layer (32 dimensions by default)
- **Decoder:** 3 upsampling layers with convolutional reconstruction
- **Training Features:**
  - Early stopping with patience
  - Learning rate scheduling (ReduceLROnPlateau)
  - Batch normalization for stable training
  - Dropout for regularization
  - GPU acceleration support

### 2. Isolation Forest Detector

Classical ensemble method for anomaly detection:

- Scikit-learn based implementation with custom wrapper
- Hyperparameter tuning via grid search:
  - Number of estimators
  - Maximum samples per tree
  - Contamination rate
  - Maximum features
- Parallel processing for faster training
- Anomaly score computation for ranking

### 3. Comprehensive Evaluation Pipeline

- **Metrics:** Precision, Recall, F1-Score, Balanced Accuracy, ROC-AUC, PR-AUC, MCC
- **Visualizations:**
  - ROC curves comparison
  - Precision-Recall curves
  - Reconstruction error distributions
  - Time series reconstruction plots
  - Model comparison bar charts
  - Confusion matrices

### 4. Modular Design

- **Data Loader:** Flexible UCR dataset loading with train/test splits
- **Evaluation Module:** Reusable metric computation and comparison functions
- **Visualization Module:** Consistent plotting utilities
- **Model Persistence:** Save and load trained models

## Usage

### Run Complete Pipeline

```bash
# Run both models and generate comparison
python scripts/main.py --all

# Run only Isolation Forest
python scripts/main.py --if

# Run only Autoencoder
python scripts/main.py --ae

# Run only comparison (requires previous results)
python scripts/main.py --compare
```

### Individual Experiments

```bash
# Isolation Forest experiment
python scripts/run_isolation_forest.py

# Autoencoder experiment
python scripts/run_autoencoder.py
```

## Implementation Details

### Autoencoder Architecture

```
Input (178 timesteps)
    ↓
Conv1D(1→32, k=7) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.2)
    ↓
Conv1D(32→64, k=5) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.2)
    ↓
Conv1D(64→128, k=3) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.2)
    ↓
Flatten → FC(2816→32) [Latent Space]
    ↓
FC(32→2816) → Reshape(128×22)
    ↓
Upsample(2×) + Conv1D(128→64, k=3) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Upsample(2×) + Conv1D(64→32, k=3) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Upsample(2×) + Conv1D(32→1, k=3)
    ↓
Output (178 timesteps reconstructed)
```

### Anomaly Detection Strategy

**Isolation Forest:**

- Anomaly score = negative of the average path length in isolation trees
- Lower path length → more isolated → higher anomaly score
- Predictions based on contamination threshold

**Autoencoder:**

- Anomaly score = Mean Squared Error between input and reconstruction
- Higher reconstruction error → anomaly
- Threshold tuned on validation set (95th percentile)

## Dependencies

```
numpy
pandas
torch
scikit-learn
matplotlib
seaborn
scipy
tqdm
```

## Results

Results are automatically saved in multiple formats:

- **Pickle files (`.pkl`):** Complete results including predictions, scores, and model states
- **CSV files:** Comparative metrics in tabular format
- **PNG images:** Visualizations of ROC curves, PR curves, and comparisons
- **Model checkpoints:** Trained models for later inference

All results can be found in the `results/` directory after running the experiments.

## Key Findings

The comparison between classical (Isolation Forest) and deep learning (Autoencoder) methods reveals:

1. **Autoencoder Advantages:**

   - Better at capturing temporal patterns and dependencies
   - More effective for complex, non-linear anomalies
   - Provides interpretable reconstructions

2. **Isolation Forest Advantages:**

   - Faster training and inference
   - No hyperparameter tuning required for basic usage
   - Works well with high-dimensional flattened time series

3. **Trade-offs:**
   - Autoencoder requires more computational resources (GPU recommended)
   - Isolation Forest is more robust to small datasets
   - Performance depends on the nature of anomalies in the data

## Author

Kamil Włodarczyk

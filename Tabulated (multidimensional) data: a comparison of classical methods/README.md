# Anomaly Detection - Tabulated (multidimensional) data: a comparison of classical methods

The primary objective of this assignment was to implement and compare multiple anomaly detection algorithms on various datasets.

1. **Implementation of Standard Anomaly Detection Methods:**

   - Isolation Forest
   - Local Outlier Factor (LOF)
   - One-Class SVM (OCSVM)

2. **Custom Algorithm Implementation:**

   - Fuzzy Local Outlier Factor (Fuzzy LOF) - a novel approach combining fuzzy logic with LOF

3. **Comprehensive Evaluation:**

   - Cross-validation using Stratified K-Fold (5 splits)
   - Multiple evaluation metrics: PR-AUC, ROC-AUC, F1-score, Recall, Balanced Accuracy, MCC
   - Hyperparameter tuning with grid search
   - Testing different preprocessing strategies (no scaling, standard scaling, robust scaling)

4. **Multi-Dataset Analysis:**

   - Evaluation on three different datasets: cardio, musk, and vowels
   - Comparison of model performance across datasets

5. **Visualization and Reporting:**
   - Precision-Recall curves
   - ROC curves
   - Model comparison plots
   - Heatmaps showing performance across datasets
   - Parameter effect analysis

## Project Structure

```
Zadanie 1/
├── main.py
├── cardio.mat
├── musk.mat
├── vowels.mat
├── results/
│   ├── *.csv
├── plots/
│   ├── model_comparison_*.png
│   ├── heatmap_*.png
│   └── param_effect_*.png
```

> **Note:** This repository contains only the source code (`main.py`) and the required datasets (`cardio.mat`, `musk.mat`, `vowels.mat`). All other files (results, plots, and report) are generated outputs from running the experiments.

## Key Features

### 1. Fuzzy LOF Implementation

A custom implementation of Fuzzy Local Outlier Factor that:

- Computes local reachability density (LRD) for each point
- Calculates LOF scores based on neighbor densities
- Applies fuzzy membership function using sigmoid transformation
- Provides soft anomaly scores instead of hard classifications

### 2. Comprehensive Preprocessing

- Missing value imputation using median strategy
- Multiple scaling options: None, StandardScaler, RobustScaler
- Automatic selection of best preprocessing strategy via cross-validation

### 3. Extensive Hyperparameter Search

Different parameter grids for each model:

- **Isolation Forest:** n_estimators, max_samples, contamination
- **LOF:** n_neighbors, contamination
- **OCSVM:** gamma, nu
- **Fuzzy LOF:** n_neighbors, alpha

### 4. Robust Evaluation Pipeline

- Stratified K-Fold cross-validation to handle imbalanced datasets
- Multiple metrics for comprehensive performance assessment
- Separate train/test split for final model evaluation
- Automated result aggregation and visualization

## Usage

```bash
python main.py
```

The script will:

1. Run all four models on all three datasets
2. Perform cross-validation with hyperparameter tuning
3. Train final models with best parameters
4. Generate evaluation metrics and visualizations
5. Save all results to CSV files and plots

## Dependencies

- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- seaborn

## Results

Results are saved in two formats:

- **CSV files:** Detailed metrics for cross-validation and test sets
- **Visualizations:** PR curves, ROC curves, comparison plots, and heatmaps

All results can be found in the `results/` and `plots/` directories after running the experiments.

## Author

Kamil Włodarczyk

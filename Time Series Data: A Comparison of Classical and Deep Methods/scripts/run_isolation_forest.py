import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from pathlib import Path
from src.data_loader import EEGBonnLoader
from src.isolation_forest import IsolationForestDetector
from src.evaluation import (
    calculate_metrics, calculate_latency_to_detection,
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    print_evaluation_report
)
from src.visualization import (
    plot_anomaly_heatmap
)
import matplotlib.pyplot as plt


def main():
    print("="*80)
    print(" "*20 + "ISOLATION FOREST - ANOMALY DETECTION")
    print("="*80)
    
    WINDOW_SIZE = 178  # 1s przy 173.61 Hz
    OVERLAP = 0.5
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    results_dir = Path('results')
    figures_dir = results_dir / 'figures' / 'isolation_forest'
    models_dir = results_dir / 'models'
    metrics_dir = results_dir / 'metrics'
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    loader = EEGBonnLoader(data_dir='Dataset')
    X_train, X_test, y_train, y_test = loader.get_train_test_split(
        test_size=TEST_SIZE,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        random_state=RANDOM_STATE
    )
    
    X_train_normal = X_train[y_train == 0]
    print(f"Dane treningowe po filtrowaniu: {X_train_normal.shape}")
    
    detector = IsolationForestDetector(contamination=0.1, random_state=RANDOM_STATE)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_samples': ['auto', 512],
        'contamination': [0.05, 0.1, 0.15],
        'max_features': [0.8, 1.0]
    }
    
    best_params = detector.tune_hyperparameters(
        X_train, y_train,
        param_grid=param_grid,
        cv=3,
        verbose=True
    )
    
    detector_final = IsolationForestDetector(
        contamination=best_params['contamination'],
        random_state=RANDOM_STATE
    )
    
    detector_final.train(
        X_train_normal,
        n_estimators=best_params['n_estimators'],
        max_samples=best_params['max_samples'],
        max_features=best_params['max_features'],
        verbose=True
    )
    
    y_pred, anomaly_scores = detector_final.predict(X_test)
    
    print_evaluation_report(y_test, y_pred, anomaly_scores, "Isolation Forest")
    
    latency_metrics = calculate_latency_to_detection(
        y_test, y_pred,
        window_size=WINDOW_SIZE,
        sampling_rate=loader.sampling_rate
    )
    
    for key, value in latency_metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key:<25}: {value:.4f}")
            else:
                print(f"  {key:<25}: {value}")
        else:
            print(f"  {key:<25}: N/A")
    
    metrics = calculate_metrics(y_test, y_pred, anomaly_scores)
    metrics.update(latency_metrics)
    metrics['best_params'] = best_params
    
    with open(metrics_dir / 'isolation_forest_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print(f"\nMetryki zapisane: {metrics_dir / 'isolation_forest_metrics.pkl'}")
    
    plot_confusion_matrix(
        y_test, y_pred,
        title="Isolation Forest - Confusion Matrix",
        save_path=figures_dir / 'confusion_matrix.png'
    )
    plt.close()
    
    plot_roc_curve(
        y_test, anomaly_scores,
        title="Isolation Forest - ROC Curve",
        save_path=figures_dir / 'roc_curve.png'
    )
    plt.close()
    
    plot_precision_recall_curve(
        y_test, anomaly_scores,
        title="Isolation Forest - Precision-Recall Curve",
        save_path=figures_dir / 'precision_recall_curve.png'
    )
    plt.close()
    
    plot_anomaly_heatmap(
        y_pred, y_test,
        window_size=WINDOW_SIZE,
        sampling_rate=loader.sampling_rate,
        title="Isolation Forest - Anomaly Detection Heatmap",
        save_path=figures_dir / 'anomaly_heatmap.png'
    )
    plt.close()
    
    print(f"\nWszystkie wizualizacje zapisane w: {figures_dir}")
    
    detector_final.save_model(models_dir / 'isolation_forest_model.pkl')
    
    results = {
        'y_test': y_test,
        'y_pred': y_pred,
        'anomaly_scores': anomaly_scores,
        'best_params': best_params
    }
    
    with open(results_dir / 'isolation_forest_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Wyniki zapisane: {results_dir / 'isolation_forest_results.pkl'}")

if __name__ == "__main__":
    main()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from pathlib import Path
from src.data_loader import EEGBonnLoader
from src.autoencoder import AnomalyDetector
from src.evaluation import (
    calculate_metrics, calculate_latency_to_detection,
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    print_evaluation_report
)
from src.visualization import (
    plot_anomaly_heatmap,
    plot_reconstruction_comparison
)
import matplotlib.pyplot as plt


def main():
    print("="*80)
    print(" "*20 + "AUTOENCODER - ANOMALY DETECTION")
    print("="*80)
    
    WINDOW_SIZE = 178  # 1s przy 173.61 Hz
    OVERLAP = 0.5
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    LATENT_DIM = 32
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    THRESHOLD_PERCENTILE = 95
    
    results_dir = Path('results')
    figures_dir = results_dir / 'figures' / 'autoencoder'
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
    
    # Podział train/val z normalnych danych
    val_split = 0.2
    n_val = int(len(X_train_normal) * val_split)
    X_val = X_train_normal[:n_val]
    X_train_final = X_train_normal[n_val:]
    
    print(f"Train: {X_train_final.shape}, Val: {X_val.shape}")
    
    detector = AnomalyDetector(
        input_length=WINDOW_SIZE,
        latent_dim=LATENT_DIM
    )
    
    detector.train(
        X_train_final,
        X_val=X_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        verbose=True
    )
    
    threshold = detector.tune_threshold(X_train_normal, percentile=THRESHOLD_PERCENTILE)
    
    y_pred, reconstruction_errors = detector.predict(X_test)
    
    print_evaluation_report(y_test, y_pred, reconstruction_errors, "Autoencoder")
    
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
    
    metrics = calculate_metrics(y_test, y_pred, reconstruction_errors)
    metrics.update(latency_metrics)
    metrics['latent_dim'] = LATENT_DIM
    metrics['threshold'] = threshold
    metrics['threshold_percentile'] = THRESHOLD_PERCENTILE
    
    with open(metrics_dir / 'autoencoder_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print(f"\nMetryki zapisane: {metrics_dir / 'autoencoder_metrics.pkl'}")
    
    plot_confusion_matrix(
        y_test, y_pred,
        title="Autoencoder - Confusion Matrix",
        save_path=figures_dir / 'confusion_matrix.png'
    )
    plt.close()
    
    plot_roc_curve(
        y_test, reconstruction_errors,
        title="Autoencoder - ROC Curve",
        save_path=figures_dir / 'roc_curve.png'
    )
    plt.close()
    
    plot_precision_recall_curve(
        y_test, reconstruction_errors,
        title="Autoencoder - Precision-Recall Curve",
        save_path=figures_dir / 'precision_recall_curve.png'
    )
    plt.close()
    
    plot_anomaly_heatmap(
        y_pred, y_test,
        window_size=WINDOW_SIZE,
        sampling_rate=loader.sampling_rate,
        title="Autoencoder - Anomaly Detection Heatmap",
        save_path=figures_dir / 'anomaly_heatmap.png'
    )
    plt.close()
    
    print("\nGenerowanie porównań rekonstrukcji")
    X_test_reconstructed = detector.reconstruct(X_test)
    
    normal_indices = np.where(y_test == 0)[0][:2]
    anomaly_indices = np.where(y_test == 1)[0][:2]
    example_indices = np.concatenate([normal_indices, anomaly_indices])
    
    plot_reconstruction_comparison(
        X_test, X_test_reconstructed,
        indices=example_indices,
        sampling_rate=loader.sampling_rate,
        title="Autoencoder - Signal Reconstruction Comparison",
        save_path=figures_dir / 'reconstruction_comparison.png'
    )
    plt.close()
    
    latent_vectors = detector.get_latent_representation(X_test)
    
    print(f"\nWszystkie wizualizacje zapisane w: {figures_dir}")
    
    detector.save_model(models_dir / 'autoencoder_model.pth')
    
    # Zapis predykcji
    results = {
        'y_test': y_test,
        'y_pred': y_pred,
        'reconstruction_errors': reconstruction_errors,
        'latent_vectors': latent_vectors,
        'X_test_reconstructed': X_test_reconstructed,
        'threshold': threshold
    }
    
    with open(results_dir / 'autoencoder_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Wyniki zapisane: {results_dir / 'autoencoder_results.pkl'}")

if __name__ == "__main__":
    main()

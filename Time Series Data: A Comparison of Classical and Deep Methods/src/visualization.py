import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_anomaly_heatmap(
    predictions: np.ndarray,
    y_true: np.ndarray = None,
    window_size: int = 178,
    sampling_rate: float = 173.61,
    title: str = "Anomaly Detection Heatmap",
    save_path: str = None
) -> plt.Figure:

    time_seconds = np.arange(len(predictions)) * (window_size / sampling_rate)
    
    if y_true is not None:
        data = np.vstack([predictions, y_true])
        labels = ['Predicted', 'True']
    else:
        data = predictions.reshape(1, -1)
        labels = ['Predicted']
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    im = ax.imshow(
        data, aspect='auto', cmap='RdYlGn_r',
        interpolation='nearest', vmin=0, vmax=1
    )
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Anomaly', fontsize=11, rotation=270, labelpad=20)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Normal', 'Anomaly'])
    
    n_ticks = 10
    tick_indices = np.linspace(0, len(predictions)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'{time_seconds[i]:.1f}' for i in tick_indices])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmapa zapisana: {save_path}")
    
    return fig


def plot_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    indices: List[int] = None,
    n_samples: int = 4,
    sampling_rate: float = 173.61,
    title: str = "Signal Reconstruction Comparison",
    save_path: str = None
) -> plt.Figure:

    if indices is None:
        indices = np.random.choice(len(original), n_samples, replace=False)
    
    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    
    if n_samples == 1:
        axes = [axes]
    
    time = np.arange(original.shape[1]) / sampling_rate
    
    for i, (ax, idx) in enumerate(zip(axes, indices)):

        ax.plot(time, original[idx], 'b-', linewidth=1.5, label='Original', alpha=0.7)
        
        ax.plot(time, reconstructed[idx], 'r--', linewidth=1.5, label='Reconstructed', alpha=0.7)
        
        mse = np.mean((original[idx] - reconstructed[idx])**2)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'Sample {idx} (MSE: {mse:.6f})', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Porównanie rekonstrukcji zapisane: {save_path}")
    
    return fig

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    title: str = "Model Comparison",
    save_path: str = None
) -> plt.Figure:
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'detection_rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        models = metrics_df['model'].values
        values = metrics_df[metric].values
        
        bars = ax.bar(models, values, color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(models, rotation=0, fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Porównanie modeli zapisane: {save_path}")
    
    return fig

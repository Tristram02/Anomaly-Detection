import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from typing import Dict
import pandas as pd

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray = None
) -> Dict:
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            
            from sklearn.metrics import auc
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            metrics['pr_auc'] = auc(recall, precision)
        except:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    
    return metrics

def calculate_latency_to_detection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_size: int = 178,
    sampling_rate: float = 173.61
) -> Dict:

    latencies = []
    
    i = 0
    while i < len(y_true):
        if y_true[i] == 1:
            j = i
            while j < len(y_true) and y_true[j] == 1:
                j += 1
            
            anomaly_segment = y_pred[i:j]
            if np.any(anomaly_segment == 1):
                first_detection = i + np.argmax(anomaly_segment == 1)
                latency_windows = first_detection - i
                latency_samples = latency_windows * window_size
                latency_seconds = latency_samples / sampling_rate
                latencies.append(latency_seconds)
            else:
                latencies.append(None)
            
            i = j
        else:
            i += 1
    
    detected_latencies = [l for l in latencies if l is not None]
    
    metrics = {
        'total_anomalies': len(latencies),
        'detected_anomalies': len(detected_latencies),
        'detection_rate': len(detected_latencies) / len(latencies) if latencies else 0,
        'mean_latency': np.mean(detected_latencies) if detected_latencies else None,
        'median_latency': np.median(detected_latencies) if detected_latencies else None,
        'min_latency': np.min(detected_latencies) if detected_latencies else None,
        'max_latency': np.max(detected_latencies) if detected_latencies else None,
        'std_latency': np.std(detected_latencies) if detected_latencies else None,
    }
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str = None
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly'],
        ax=ax, cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix zapisana: {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    save_path: str = None
) -> plt.Figure:

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve zapisana: {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: str = None
) -> plt.Figure:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.fill_between(recall, precision, alpha=0.2)
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve zapisana: {save_path}")
    
    return fig


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray = None,
    model_name: str = "Model"
):
    print("="*60)
    print(f"RAPORT EWALUACJI: {model_name}")
    print("="*60)
    
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    
    print(f"\n{'Metryka':<25} {'Wartość':>10}")
    print("-"*40)
    print(f"{'Accuracy':<25} {metrics['accuracy']:>10.4f}")
    print(f"{'Precision':<25} {metrics['precision']:>10.4f}")
    print(f"{'Recall':<25} {metrics['recall']:>10.4f}")
    print(f"{'F1-Score':<25} {metrics['f1_score']:>10.4f}")
    print(f"{'Detection Rate':<25} {metrics['detection_rate']:>10.4f}")
    print(f"{'False Alarm Rate':<25} {metrics['false_alarm_rate']:>10.4f}")
    
    if metrics.get('roc_auc') is not None:
        print(f"{'ROC-AUC':<25} {metrics['roc_auc']:>10.4f}")
    
    print("\n" + "-"*40)
    print("Confusion Matrix:")
    print(f"  True Negatives:  {metrics['true_negatives']:>6}")
    print(f"  False Positives: {metrics['false_positives']:>6}")
    print(f"  False Negatives: {metrics['false_negatives']:>6}")
    print(f"  True Positives:  {metrics['true_positives']:>6}")
    
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    print("="*60 + "\n")


def compare_models(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    scores_dict: Dict[str, np.ndarray] = None
) -> pd.DataFrame:
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        y_scores = scores_dict.get(model_name) if scores_dict else None
        metrics = calculate_metrics(y_true, y_pred, y_scores)
        metrics['model'] = model_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score',
            'detection_rate', 'false_alarm_rate']
    if 'roc_auc' in df.columns:
        cols.append('roc_auc')
    
    df = df[cols]
    
    return df


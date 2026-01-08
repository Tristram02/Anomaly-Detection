"""
    main.py --all # Uruchom wszystko
    main.py --if # Tylko Isolation Forest
    main.py --ae # Tylko Autoencoder
    main.py --compare # Tylko porównanie
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
from pathlib import Path
import pickle
import pandas as pd
from src.evaluation import compare_models
from src.visualization import plot_model_comparison
import matplotlib.pyplot as plt


def run_isolation_forest():
    print("\n" + "="*80)
    print("URUCHAMIANIE: Isolation Forest")
    print("="*80 + "\n")
    
    result = subprocess.run(
        ['python', 'scripts/run_isolation_forest.py'],
        cwd=Path.cwd()
    )
    
    if result.returncode != 0:
        print("\nIsolation Forest zakończony z błędem!")
        return False
    
    print("\nIsolation Forest zakończony pomyślnie!")
    return True


def run_autoencoder():
    print("\n" + "="*80)
    print("URUCHAMIANIE: Autoencoder")
    print("="*80 + "\n")
    
    result = subprocess.run(
        ['python', 'scripts/run_autoencoder.py'],
        cwd=Path.cwd()
    )
    
    if result.returncode != 0:
        print("\nAutoencoder zakończony z błędem!")
        return False
    
    print("\nAutoencoder zakończony pomyślnie!")
    return True


def compare_results():
    print("\n" + "="*80)
    print("PORÓWNANIE WYNIKÓW")
    print("="*80)
    
    results_dir = Path('results')
    figures_dir = results_dir / 'figures' / 'comparison'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(results_dir / 'isolation_forest_results.pkl', 'rb') as f:
            if_results = pickle.load(f)
        
        with open(results_dir / 'autoencoder_results.pkl', 'rb') as f:
            ae_results = pickle.load(f)
        
        with open(results_dir / 'metrics' / 'isolation_forest_metrics.pkl', 'rb') as f:
            if_metrics = pickle.load(f)
        
        with open(results_dir / 'metrics' / 'autoencoder_metrics.pkl', 'rb') as f:
            ae_metrics = pickle.load(f)
    
    except FileNotFoundError as e:
        print(f"\nNie znaleziono plików z wynikami: {e}")
        print("Uruchom najpierw oba eksperymenty")
        return False
    
    y_test = if_results['y_test']
    
    print("\n" + "-"*80)
    print("PORÓWNANIE METRYK")
    print("-"*80)
    
    predictions_dict = {
        'Isolation Forest': if_results['y_pred'],
        'Autoencoder': ae_results['y_pred']
    }
    
    scores_dict = {
        'Isolation Forest': if_results['anomaly_scores'],
        'Autoencoder': ae_results['reconstruction_errors']
    }
    
    comparison_df = compare_models(y_test, predictions_dict, scores_dict)
    
    print("\n" + str(comparison_df.to_string(index=False)))
    
    comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    print(f"\nPorównanie zapisane: {results_dir / 'model_comparison.csv'}")
    
    plot_model_comparison(
        comparison_df,
        title="Model Comparison: Isolation Forest vs Autoencoder",
        save_path=figures_dir / 'model_comparison.png'
    )
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fpr_if, tpr_if, _ = roc_curve(y_test, if_results['anomaly_scores'])
    auc_if = roc_auc_score(y_test, if_results['anomaly_scores'])
    ax.plot(fpr_if, tpr_if, 'b-', linewidth=2, label=f'Isolation Forest (AUC = {auc_if:.3f})')
    
    fpr_ae, tpr_ae, _ = roc_curve(y_test, ae_results['reconstruction_errors'])
    auc_ae = roc_auc_score(y_test, ae_results['reconstruction_errors'])
    ax.plot(fpr_ae, tpr_ae, 'r-', linewidth=2, label=f'Autoencoder (AUC = {auc_ae:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Wizualizacje porównawcze zapisane w: {figures_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection Pipeline')
    parser.add_argument('--all', action='store_true', help='Uruchom wszystko')
    parser.add_argument('--if', dest='isolation_forest', action='store_true', help='Tylko Isolation Forest')
    parser.add_argument('--ae', dest='autoencoder', action='store_true', help='Tylko Autoencoder')
    parser.add_argument('--compare', action='store_true', help='Tylko porównanie')
    
    args = parser.parse_args()
    
    if not any([args.all, args.isolation_forest, args.autoencoder, args.compare]):
        args.all = True
    
    success = True
    
    if args.all or args.isolation_forest:
        if not run_isolation_forest():
            success = False
    
    if args.all or args.autoencoder:
        if not run_autoencoder():
            success = False
    
    if args.all or args.compare:
        if not compare_results():
            success = False
    
    print("\n" + "="*80)
    if success:
        print(" "*25 + "ZAKOŃCZONO POMYŚLNIE")
    else:
        print(" "*25 + "ZAKOŃCZONO Z BŁĘDAMI")
    print("="*80)


if __name__ == "__main__":
    main()

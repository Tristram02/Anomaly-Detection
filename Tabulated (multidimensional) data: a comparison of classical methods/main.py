import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, precision_recall_curve, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

class FuzzyLOF(BaseEstimator):
    def __init__(self, n_neighbors=20, alpha=5.0):
        self.n_neighbors = n_neighbors
        self.alpha = alpha

    def fit(self, X, y=None):
        self.X_ = np.asarray(X)
        self.n_samples_ = self.X_.shape[0]
        self._neighbors = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X_)
        distances, indices = self._neighbors.kneighbors(self.X_)

        self.distances_ = distances[:, 1:]
        self.indices_ = indices[:, 1:]

        lrd = np.zeros(self.n_samples_)
        for i in range(self.n_samples_):
            reach_dists = np.maximum(
                self.distances_[self.indices_[i, :], -1],
                self.distances_[i]
            )
            lrd[i] = 1.0 / (np.mean(reach_dists) + 1e-10)
        self.lrd_ = lrd

        lof = np.zeros(self.n_samples_)
        for i in range(self.n_samples_):
            lof[i] = np.mean(self.lrd_[self.indices_[i, :]] / (self.lrd_[i] + 1e-10))
        self.lof_ = lof

        self.lof_min_ = np.min(self.lof_)
        self.lof_max_ = np.max(self.lof_)
        
        norm_lof = (self.lof_ - self.lof_min_) / (self.lof_max_ - self.lof_min_ + 1e-10)
        self.membership_ = 1.0 / (1.0 + np.exp(-self.alpha * (norm_lof - 0.5)))

        self.decision_scores_ = self.membership_
        return self

    def decision_function(self, X):
        if not hasattr(self, "X_"):
            raise RuntimeError("Model not trained!")
        X = np.asarray(X)
        
        distances, indices = self._neighbors.kneighbors(X, n_neighbors=self.n_neighbors)
        
        lrd_query = np.zeros(X.shape[0])
        lof_query = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):

            neighbor_indices = indices[i]
            neighbor_distances = distances[i]

            k_distances = self.distances_[neighbor_indices, -1]

            reach_dists = np.maximum(k_distances, neighbor_distances)
            lrd_query[i] = 1.0 / (np.mean(reach_dists) + 1e-10)

            lof_query[i] = np.mean(self.lrd_[neighbor_indices]) / (lrd_query[i] + 1e-10)

        norm_lof = (lof_query - self.lof_min_) / (self.lof_max_ - self.lof_min_ + 1e-10)
        norm_lof = np.clip(norm_lof, 0, 1)

        membership = 1.0 / (1.0 + np.exp(-self.alpha * (norm_lof - 0.5)))
        return membership

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > 0.5).astype(int)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return (self.membership_ > 0.5).astype(int)



def load_dataset(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Plik '{filename}' nie znaleziony.")
    
    data = loadmat(filename)
    X = data['X']
    y = data['y'].ravel()
    
    return X.astype(float), y.astype(int)


def preprocess_data(X_train, X_test, scaler_name):
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    if scaler_name == "standard":
        scaler = StandardScaler().fit(X_train)
    elif scaler_name == "robust":
        scaler = RobustScaler().fit(X_train)
    else:
        return X_train, X_test

    return scaler.transform(X_train), scaler.transform(X_test)


def get_model(model_name, params):
    if model_name == "isolation_forest":
        return IsolationForest(**params)
    elif model_name == "lof":
        return LocalOutlierFactor(**params)
    elif model_name == "ocsvm":
        return OneClassSVM(**params)
    elif model_name == "fuzzy_lof":
        return FuzzyLOF(**params)
    else:
        raise ValueError(f"Nieznany model: {model_name}")


def evaluate_model(model_name, X_train, y_train, scalers, param_grid, contamination_list, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for scaler_name, scaler in scalers.items():
        for g in list(ParameterGrid(param_grid)):
            for contamination in contamination_list:
                fold_metrics = {m: [] for m in ["pr", "roc", "f1", "recall", "bal", "mcc"]}

                for train_idx, val_idx in skf.split(X_train, y_train):
                    Xt_tr, Xt_val = X_train[train_idx], X_train[val_idx]
                    yt_tr, yt_val = y_train[train_idx], y_train[val_idx]
                    Xt_tr, Xt_val = preprocess_data(Xt_tr, Xt_val, scaler_name)

                    params = dict(g)
                    if model_name == "isolation_forest":
                        params.update({
                            "random_state": 0,
                            "n_jobs": -1,
                            "contamination": "auto" if contamination is None else contamination
                        })
                        model = get_model(model_name, params)
                        model.fit(Xt_tr)
                        y_pred = (model.predict(Xt_val) == -1).astype(int)
                        scores = -model.decision_function(Xt_val)

                    elif model_name == "lof":
                        params.update({
                            "n_neighbors": g["n_neighbors"],
                            "contamination": "auto" if contamination is None else contamination,
                            "novelty": True
                        })
                        model = get_model(model_name, params)
                        model.fit(Xt_tr)
                        y_pred = (model.predict(Xt_val) == -1).astype(int)
                        scores = -model.decision_function(Xt_val)
                    
                    elif model_name == "ocsvm":
                        params.update({
                            "gamma": g["gamma"],
                            "nu": g["nu"]
                        })
                        model = get_model(model_name, params)
                        model.fit(Xt_tr)
                        y_pred = (model.predict(Xt_val) == -1).astype(int)
                        scores = -model.decision_function(Xt_val)

                    elif model_name == "fuzzy_lof":
                        params.update({
                            "n_neighbors": g["n_neighbors"],
                            "alpha": g["alpha"]
                        })
                        model = get_model(model_name, params)
                        model.fit(Xt_tr)
                        y_pred = model.predict(Xt_val)
                        scores = model.decision_function(Xt_val)
                    else:
                        raise ValueError("Nieobsługiwany model.")

                    fold_metrics["pr"].append(average_precision_score(yt_val, scores))
                    fold_metrics["roc"].append(roc_auc_score(yt_val, scores))
                    fold_metrics["f1"].append(f1_score(yt_val, y_pred, zero_division=0))
                    fold_metrics["recall"].append(recall_score(yt_val, y_pred, zero_division=0))
                    fold_metrics["bal"].append(balanced_accuracy_score(yt_val, y_pred))
                    fold_metrics["mcc"].append(matthews_corrcoef(yt_val, y_pred))

                results.append({
                    "model": model_name,
                    "scaler": scaler_name,
                    **g,
                    "contamination": "auto" if contamination is None else contamination,
                    "pr_auc_mean": np.mean(fold_metrics["pr"]),
                    "roc_auc_mean": np.mean(fold_metrics["roc"]),
                    "f1_mean": np.mean(fold_metrics["f1"]),
                    "recall_mean": np.mean(fold_metrics["recall"]),
                    "bal_acc_mean": np.mean(fold_metrics["bal"]),
                    "mcc_mean": np.mean(fold_metrics["mcc"]),
                })

    return pd.DataFrame(results)


def plot_curves(y_true, scores, model_name, dataset_name):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall ({model_name} on {dataset_name})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig(f'{model_name}_pr_curves_{dataset_name}.png', dpi=300, bbox_inches='tight')

    plt.subplot(1,2,2)
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve ({model_name} on {dataset_name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_model_comparison(df_results, dataset_name):
    df = df_results[df_results["dataset"] == dataset_name]
    metrics = ["roc_auc_mean", "pr_auc_mean", "f1_mean", "recall_mean", "bal_acc_mean"]
    melted = df.melt(id_vars=["model"], value_vars=metrics,
                     var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="Metric", y="Score", hue="model", palette="Set2")
    plt.title(f"Porównanie modeli dla zbioru danych: {dataset_name}")
    plt.ylabel("Wartość metryki")
    plt.xlabel("Metryka")
    plt.ylim(0, 1)
    plt.legend(title="Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(f"plots/model_comparison_{dataset_name}.png", dpi=300)
    plt.close()

def plot_heatmap(df_results, metric="pr_auc_mean"):
    pivot = df_results.pivot_table(values=metric, index="model", columns="dataset", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)
    plt.title(f"Heatmapa wyników modeli ({metric})")
    plt.ylabel("Model")
    plt.xlabel("Zbiór danych")
    plt.tight_layout()
    plt.savefig(f"plots/heatmap_{metric}.png", dpi=300)
    plt.close()

def plot_parameter_effect(df_results, dataset_name, metric="pr_auc_mean", param="contamination"):
    df = df_results[df_results["dataset"] == dataset_name]
    if param not in df.columns:
        print(f"Brak {param} w rezultatach!")
        return

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x=param, y=metric, style="model", hue="model", marker="o", palette="Set1", errorbar=None)
    plt.title(f"Wpływ parametru {param} na {metric} ({dataset_name})")
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"plots/param_effect_{param}_{dataset_name}.png", dpi=300)
    plt.close()

# 
# MAIN EXPERIMENT
# 

def run_experiment(model_name, dataset_file):
    print(f"\n=== Model: {model_name} | Dataset: {dataset_file} ===")
    X, y = load_dataset(dataset_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        stratify=y, random_state=42)

    scalers = {"none": None, "standard": StandardScaler(), "robust": RobustScaler()}

    if model_name == "isolation_forest":
        param_grid = {"n_estimators": [100, 200], "max_samples": [0.5, "auto"]}
        contamination_list = [None, 0.01, 0.05]
    elif model_name == "lof":
        param_grid = {"n_neighbors": [10, 20, 30]}
        contamination_list = [None, 0.01, 0.05]
    elif model_name == "ocsvm":
        param_grid = {"gamma": ["scale", 0.01, 0.1], "nu": [0.01, 0.05, 0.1]}
        contamination_list = [None]
    elif model_name == "fuzzy_lof":
        param_grid = {"n_neighbors": [10, 20, 30], "alpha": [2.0, 5.0, 10.0]}
        contamination_list = [None]
    else:
        raise ValueError("Nieznany model.")


    df_results = evaluate_model(model_name, X_train, y_train, scalers, param_grid, contamination_list)

    best = df_results.sort_values("pr_auc_mean", ascending=False).iloc[0]
    print("Najlepsze parametry:\n", best)

    X_train_proc, X_test_proc = preprocess_data(X_train, X_test, best["scaler"])
    if model_name == "isolation_forest":
        best_params = {
            "n_estimators": int(best["n_estimators"]),
            "max_samples": best["max_samples"],
            "random_state": 0,
            "n_jobs": -1,
            "contamination": "auto" if best["contamination"] == "auto" else float(best["contamination"])
        }
    elif model_name == "lof":
        best_params = {
            "n_neighbors": int(best["n_neighbors"]),
            "contamination": "auto" if best["contamination"] == "auto" else float(best["contamination"]),
            "novelty": True
        }
    elif model_name == "ocsvm":
        best_params = {
            "gamma": best["gamma"],
            "nu": best["nu"]
        }
    elif model_name == "fuzzy_lof":
        best_params = {
            "n_neighbors": int(best["n_neighbors"]),
            "alpha": best["alpha"]
        }

    model = get_model(model_name, best_params)
    model.fit(X_train_proc)

    if model_name != "fuzzy_lof":
        y_scores = -model.decision_function(X_test_proc)
        y_pred = (model.predict(X_test_proc) == -1).astype(int)
    else:
        y_scores = model.decision_function(X_test_proc)
        y_pred = model.predict(X_test_proc)

    test_metrics = {
        "pr_auc": average_precision_score(y_test, y_scores),
        "roc_auc": roc_auc_score(y_test, y_scores),
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "bal_acc": balanced_accuracy_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }
    print("Test metrics:", test_metrics)

    name = f"{model_name}_{os.path.basename(dataset_file).split('.')[0]}"
    os.makedirs("results", exist_ok=True)
    df_results["dataset"] = os.path.basename(dataset_file).split(".")[0]
    df_results.to_csv(f"results/{name}_cv_results.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(f"results/{name}_test_metrics.csv", index=False)

    plot_curves(y_test, y_scores, model_name, dataset_file)


if __name__ == "__main__":
    run_experiment("isolation_forest", "musk.mat")
    run_experiment("lof", "musk.mat")
    run_experiment("ocsvm", "musk.mat")
    run_experiment("fuzzy_lof", "musk.mat")

    run_experiment("isolation_forest", "cardio.mat")
    run_experiment("lof", "cardio.mat")
    run_experiment("ocsvm", "cardio.mat")
    run_experiment("fuzzy_lof", "cardio.mat")

    run_experiment("isolation_forest", "vowels.mat")
    run_experiment("lof", "vowels.mat")
    run_experiment("ocsvm", "vowels.mat")
    run_experiment("fuzzy_lof", "vowels.mat")

    all_csvs = glob.glob("results/*_cv_results.csv")
    df_all = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    df_all.to_csv(f"results/all_cv_results.csv", index=False)
    metric_cols = [c for c in df_all.columns if c.endswith("_mean")]
    for metric in metric_cols:
        plot_heatmap(df_all, metric)
    
    for dataset in df_all["dataset"].unique():
        plot_model_comparison(df_all, dataset)
    
    possible_params = ["contamination", "scaler", "alpha", "n_neighbors", "nu", "gamma", "max_samples"]
    for param in possible_params:
        if param in df_all.columns:
            for dataset in df_all["dataset"].unique():
                plot_parameter_effect(df_all, dataset, metric="pr_auc_mean", param=param)

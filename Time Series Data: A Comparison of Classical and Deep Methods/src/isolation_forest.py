import numpy as np
import pickle
from typing import Tuple, Dict
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer, f1_score
from tqdm import tqdm


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.anomaly_scores = None
        
    def train(
        self, 
        X_train: np.ndarray,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        max_features: float = 1.0,
        verbose: bool = True
    ):
        if verbose:
            print("="*60)
            print("TRENING ISOLATION FOREST")
            print("="*60)
            print(f"Dane treningowe: {X_train.shape}")
            print(f"Parametry:")
            print(f"- n_estimators: {n_estimators}")
            print(f"- max_samples: {max_samples}")
            print(f"- max_features: {max_features}")
            print(f"- contamination: {self.contamination}")
        
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(len(X_train), -1)
        else:
            X_train_flat = X_train
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        if verbose:
            print("\nTrenowanie modelu")
        
        self.model.fit(X_train_flat)
        
        if verbose:
            print("Model wytrenowany!")
            print("="*60 + "\n")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model nie został wytrenowany! Użyj train() najpierw.")
        
        if len(X.shape) > 2:
            X_flat = X.reshape(len(X), -1)
        else:
            X_flat = X
        
        predictions_raw = self.model.predict(X_flat)
        
        predictions = (predictions_raw == -1).astype(int)
        
        anomaly_scores_raw = self.model.score_samples(X_flat)
        anomaly_scores = -anomaly_scores_raw
        self.anomaly_scores = anomaly_scores
        
        return predictions, anomaly_scores
    
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict = None,
        cv: int = 3,
        verbose: bool = True
    ) -> Dict:
        if verbose:
            print("="*60)
            print("GRID SEARCH - TUNING HIPERPARAMETRÓW")
            print("="*60)
        
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(len(X_train), -1)
        else:
            X_train_flat = X_train
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_samples': ['auto', 256, 512],
                'contamination': [0.05, 0.1, 0.15],
                'max_features': [0.5, 0.8, 1.0]
            }
        
        if verbose:
            print(f"Siatka parametrów:")
            for key, values in param_grid.items():
                print(f"  - {key}: {values}")
            print(f"\nLiczba kombinacji: {np.prod([len(v) for v in param_grid.values()])}")
            print(f"Cross-validation folds: {cv}\n")
        
        from itertools import product
        
        best_score = -np.inf
        best_params = None
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        if verbose:
            print("Rozpoczynam grid search...")
        
        from tqdm import tqdm
        total_combinations = np.prod([len(v) for v in values])
        
        for combination in tqdm(list(product(*values)), desc="Grid Search", disable=not verbose):
            params = dict(zip(keys, combination))
            
            model = IsolationForest(
                n_estimators=params['n_estimators'],
                max_samples=params['max_samples'],
                contamination=params['contamination'],
                max_features=params['max_features'],
                random_state=self.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train_flat)
            
            predictions_raw = model.predict(X_train_flat)
            predictions = (predictions_raw == -1).astype(int)
            
            from sklearn.metrics import f1_score
            score = f1_score(y_train, predictions, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        self.best_params = best_params
        
        self.model = IsolationForest(
            n_estimators=best_params['n_estimators'],
            max_samples=best_params['max_samples'],
            contamination=best_params['contamination'],
            max_features=best_params['max_features'],
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_flat)
        
        if verbose:
            print(f"\nGrid search zakończony")
            print(f"\nNajlepsze parametry:")
            for key, value in self.best_params.items():
                print(f"  - {key}: {value}")
            print(f"\nNajlepszy F1 score: {best_score:.4f}")
            print("="*60 + "\n")
        
        return self.best_params
    
    def get_anomaly_scores(self) -> np.ndarray:
        if self.anomaly_scores is None:
            raise ValueError("Najpierw wykonaj predict()!")
        return self.anomaly_scores
    
    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("Model nie został wytrenowany!")
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model zapisany: {filepath}")
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.best_params = model_data.get('best_params')
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        
        print(f"Model wczytany: {filepath}")


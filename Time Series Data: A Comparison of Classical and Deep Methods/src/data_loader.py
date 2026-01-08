import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class EEGBonnLoader:

    def __init__(self, data_dir: str = 'Dataset'):
        self.data_dir = Path(data_dir)
        self.sampling_rate = 173.61  # Hz
        self.duration = 23.6  # sekundy
        self.samples_per_file = 4097

        self.sets = {
            'Z': {'label': 0, 'description': 'Zdrowi, oczy otwarte'},
            'O': {'label': 0, 'description': 'Zdrowi, oczy zamknięte'},
            'N': {'label': 0, 'description': 'Epilepsja, bez napadu'},
            'F': {'label': 0, 'description': 'Epilepsja, bez napadu'},
            'S': {'label': 1, 'description': 'Napady epileptyczne (ANOMALIE)'}
        }
        
        self.data = None
        self.labels = None
        self.scaler = StandardScaler()
        
    def load_file(self, filepath: Path) -> np.ndarray:
        with open(filepath, 'r') as f:
            data = [float(line.strip()) for line in f.readlines()]
        return np.array(data)
    
    def load_set(self, set_name: str) -> Tuple[np.ndarray, np.ndarray]:
        set_dir = self.data_dir / set_name
        if not set_dir.exists():
            raise FileNotFoundError(f"Folder {set_dir} nie istnieje!")
        
        files = list(set_dir.glob('*.txt')) + list(set_dir.glob('*.TXT'))
        files = sorted(set(files))
        
        if len(files) == 0:
            raise FileNotFoundError(f"Brak plików .txt/.TXT w folderze {set_dir}!")
        
        label = self.sets[set_name]['label']
        
        print(f"Wczytywanie zbioru {set_name} ({self.sets[set_name]['description']})...")
        
        data_list = []
        for filepath in tqdm(files, desc=f"SET {set_name}"):
            signal = self.load_file(filepath)
            data_list.append(signal)
        
        data = np.array(data_list)
        labels = np.full(len(data), label)
        
        return data, labels
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        all_data = []
        all_labels = []
        
        for set_name in ['Z', 'O', 'N', 'F', 'S']:
            data, labels = self.load_set(set_name)
            all_data.append(data)
            all_labels.append(labels)
        
        self.data = np.vstack(all_data)
        self.labels = np.concatenate(all_labels)
        
        print(f"\nWczytano {len(self.data)} plików")
        print(f"- Normalne: {np.sum(self.labels == 0)}")
        print(f"- Anomalie: {np.sum(self.labels == 1)}")
        
        return self.data, self.labels
    
    def create_windows(self, window_size: int = 178, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        step = int(window_size * (1 - overlap))
        windows = []
        window_labels = []
        
        print(f"\nTworzenie okien czasowych (rozmiar={window_size}, overlap={overlap})...")
        
        for i, signal in enumerate(tqdm(self.data, desc="Segmentacja")):
            label = self.labels[i]

            for start in range(0, len(signal) - window_size + 1, step):
                window = signal[start:start + window_size]
                windows.append(window)
                window_labels.append(label)
        
        windows = np.array(windows)
        window_labels = np.array(window_labels)
        
        print(f"Utworzono {len(windows)} okien")
        print(f"- Normalne: {np.sum(window_labels == 0)}")
        print(f"- Anomalie: {np.sum(window_labels == 1)}")
        
        return windows, window_labels
    
    def normalize(self, X_train: np.ndarray, X_test: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        original_shape_train = X_train.shape
        X_train_flat = X_train.reshape(len(X_train), -1)

        X_train_norm = self.scaler.fit_transform(X_train_flat)
        X_train_norm = X_train_norm.reshape(original_shape_train)
        
        if X_test is not None:
            original_shape_test = X_test.shape
            X_test_flat = X_test.reshape(len(X_test), -1)
            X_test_norm = self.scaler.transform(X_test_flat)
            X_test_norm = X_test_norm.reshape(original_shape_test)
            return X_train_norm, X_test_norm
        
        return X_train_norm, None
    
    def get_train_test_split(
        self, 
        test_size: float = 0.3,
        window_size: int = 178,
        overlap: float = 0.5,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.data is None:
            self.load_all_data()

        X, y = self.create_windows(window_size=window_size, overlap=overlap)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print("\nNormalizacja danych (z-score)...")
        X_train, X_test = self.normalize(X_train, X_test)
        
        print(f"\n{'='*60}")
        print("PODSUMOWANIE DANYCH:")
        print(f"{'='*60}")
        print(f"Train set: {X_train.shape}")
        print(f"- Normalne: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
        print(f"- Anomalie: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
        print(f"\nTest set: {X_test.shape}")
        print(f"- Normalne: {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
        print(f"- Anomalie: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
        print(f"{'='*60}\n")
        
        return X_train, X_test, y_train, y_test
    
    def get_statistics(self) -> Dict:
        stats = {
            'total_files': len(self.data),
            'normal_files': np.sum(self.labels == 0),
            'anomaly_files': np.sum(self.labels == 1),
            'samples_per_file': self.samples_per_file,
            'sampling_rate': self.sampling_rate,
            'duration': self.duration,
            'mean': np.mean(self.data),
            'std': np.std(self.data),
            'min': np.min(self.data),
            'max': np.max(self.data)
        }
        
        return stats

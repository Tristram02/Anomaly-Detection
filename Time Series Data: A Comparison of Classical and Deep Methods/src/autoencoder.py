import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from tqdm import tqdm


class Conv1DAutoencoder(nn.Module):
    
    def __init__(self, input_length: int = 178, latent_dim: int = 32):
        super(Conv1DAutoencoder, self).__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        
        self.encoded_size = 128 * 22
        
        self.fc_encode = nn.Linear(self.encoded_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.encoded_size)
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )
        
        self.final_adjust = nn.Conv1d(1, 1, kernel_size=3, padding=1)
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encode(x)
        return x
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 22)
        x = self.decoder(x)
        if x.size(2) > 178:
            x = x[:, :, :178]
        elif x.size(2) < 178:
            padding = 178 - x.size(2)
            x = torch.nn.functional.pad(x, (0, padding))
        return x
    
    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class AnomalyDetector:
    def __init__(
        self, 
        input_length: int = 178,
        latent_dim: int = 32,
        device: str = None
    ):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = Conv1DAutoencoder(input_length, latent_dim).to(self.device)
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.threshold = None
        self.reconstruction_errors = None
        self.train_history = {'train_loss': [], 'val_loss': []}
        
        print(f"Autoencoder zainicjalizowany na: {self.device}")
        print(f"  - Input length: {input_length}")
        print(f"  - Latent dim: {latent_dim}")
        print(f"  - Parametry: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        verbose: bool = True
    ):
        if verbose:
            print("="*60)
            print("TRENING AUTOENCODER")
            print("="*60)
            print(f"Dane treningowe: {X_train.shape}")
            if X_val is not None:
                print(f"Dane walidacyjne: {X_val.shape}")
            print(f"Parametry:")
            print(f"- Epochs: {epochs}")
            print(f"- Batch size: {batch_size}")
            print(f"- Learning rate: {learning_rate}")
        
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
            val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10
        
        if verbose:
            print("\nRozpoczynam trening...\n")
        
        for epoch in range(epochs):
            train_loss = 0.0
            train_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader
            
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                reconstructed, _ = self.model(batch_x)
                loss = criterion(reconstructed, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                if verbose:
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            avg_train_loss = train_loss / train_batches
            self.train_history['train_loss'].append(avg_train_loss)
            
            if X_val is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        reconstructed, _ = self.model(batch_x)
                        loss = criterion(reconstructed, batch_y)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                self.train_history['val_loss'].append(avg_val_loss)
                
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping po {epoch+1} epokach")
                    break
                
                self.model.train()
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
        
        if verbose:
            print("\nTrening zakończony")
            print("="*60 + "\n")
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        
        errors = []
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                reconstructed, _ = self.model(batch)

                mse = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
                errors.extend(mse.cpu().numpy())
        
        self.reconstruction_errors = np.array(errors)
        return self.reconstruction_errors
    
    def tune_threshold(self, X_normal: np.ndarray, percentile: float = 95) -> float:

        errors = self.get_reconstruction_error(X_normal)
        self.threshold = np.percentile(errors, percentile)
        
        print(f"Próg ustawiony na {percentile}. percentyl: {self.threshold:.6f}")
        print(f"- Min error: {np.min(errors):.6f}")
        print(f"- Mean error: {np.mean(errors):.6f}")
        print(f"- Max error: {np.max(errors):.6f}")
        
        return self.threshold
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        errors = self.get_reconstruction_error(X)
        predictions = (errors > self.threshold).astype(int)
        
        return predictions, errors
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            reconstructed, _ = self.model(X_tensor)
        
        return reconstructed.squeeze(1).cpu().numpy()
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            _, latent = self.model(X_tensor)
        
        return latent.cpu().numpy()
    
    def save_model(self, filepath: str):
        model_data = {
            'state_dict': self.model.state_dict(),
            'input_length': self.input_length,
            'latent_dim': self.latent_dim,
            'threshold': self.threshold,
            'train_history': self.train_history
        }
        torch.save(model_data, filepath)
        print(f"Model zapisany: {filepath}")
    
    def load_model(self, filepath: str):
        model_data = torch.load(filepath, map_location=self.device)
        
        self.model = Conv1DAutoencoder(
            model_data['input_length'],
            model_data['latent_dim']
        ).to(self.device)
        
        self.model.load_state_dict(model_data['state_dict'])
        self.input_length = model_data['input_length']
        self.latent_dim = model_data['latent_dim']
        self.threshold = model_data.get('threshold')
        self.train_history = model_data.get('train_history', {'train_loss': [], 'val_loss': []})
        
        print(f"Model wczytany: {filepath}")

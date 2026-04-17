import torch
import torch.nn as nn
import torch.optim as optim
import mne
import numpy as np
from src.fusion.cross_attention import NetraSyncFusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_real_clinical_data():
    print("Loading EEG runs for high-accuracy training...")
    runs = [1, 4] 
    data_paths = mne.datasets.eegbci.load_data(1, runs, path='./data')
    
    all_data = []
    all_labels = []
    
    for i, path in enumerate(data_paths):
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)
        raw.pick(['FC5', 'FC1', 'FC2', 'FC6', 'CP5', 'CP1', 'CP2', 'CP6'])
        
        data = raw.get_data()
        # High-Precision Normalization
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        for j in range(0, data.shape[1] - 50, 25): # Increased overlap for more data
            chunk = data[:, j:j+50].T
            all_data.append(chunk)
            all_labels.append(0.0 if i == 0 else 1.0)
            
    return torch.tensor(np.array(all_data)).float(), torch.tensor(np.array(all_labels)).float()
# Change these lines in your train_model.py:

def train_finely():
    X, y = get_real_clinical_data()
    X, y = X.to(device), y.to(device)
    
    # Matching the new 64-dim architecture
    model = NetraSyncFusion(embed_dim=64).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    # Prevent LR from becoming invisible (min_lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=30, factor=0.5, min_lr=1e-5)
    
    print(f"Training on {len(X)} samples with Enhanced Architecture...")
    
    for epoch in range(1501):
        model.train()
        optimizer.zero_grad()
        
        # Add gaze variability
        gaze = (torch.tensor([[0.5, 0.5]] * len(X)) + torch.randn(len(X), 2) * 0.05).to(device)
        
        outputs, _ = model(gaze, X)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()
        
        preds = (outputs.squeeze() > 0.5).float()
        acc = (preds == y).float().mean()
        scheduler.step(acc)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Acc: {acc.item()*100:.2f}% | Loss: {loss.item():.4f}")
            if acc.item() >= 0.96:
                print("--- 96% REACHED ---")
                break

    torch.save(model.state_dict(), "netrasync_model.pth")
if __name__ == "__main__":
    train_finely()
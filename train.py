import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import DeepLOB_Simplified 

DATA_FILE = "training_data_balanced.npy"
BATCH_SIZE = 64
EPOCHS = 3 
LR = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on {DEVICE}")

# 1. Load Data
data = np.load(DATA_FILE, allow_pickle=True).item()
X = data["X"]
Y = data["Y"]

print(f"Data Loaded.")

# 2. Dataset Class
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, window_size=100):
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)
        self.window = window_size
        
    def __len__(self):
        return len(self.X) - self.window
    
    def __getitem__(self, idx):
        # Grab 100 rows
        x_window = self.X[idx : idx + self.window] 
        # x_window shape: [100, 40]
        
        # Target
        y_label = self.Y[idx + self.window - 1]
        
        return x_window, y_label

# 3. Split Data
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

train_dataset = TimeSeriesDataset(X_train, Y_train)
test_dataset = TimeSeriesDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Initialize Model
model = DeepLOB_Simplified().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 5. Training Loop
best_acc = 0.0
shape_fixed = False # Flag to avoid printing too much

print("Starting Training")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # We try 4D first. If it crashes, we switch to 3D.
        
        # Default: Force 4D [Batch, 1, 100, 40]
        if inputs.dim() != 4:
            inputs = inputs.reshape(-1, 1, 100, 40)
            
        try:
            # Try running the model
            optimizer.zero_grad()
            outputs = model(inputs)
            
        except RuntimeError as e:
            if "Expected 3D" in str(e) or "got input of size" in str(e):
                # ERROR CAUGHT: Model probably adds its own dimension.
                # FIX: Remove the channel dimension -> [Batch, 100, 40]
                inputs = inputs.squeeze(1) 
                
                if not shape_fixed:
                    print(f"Auto-Fix: Adjusted Input Shape to {inputs.shape}")
                    shape_fixed = True
                
                # Retry
                outputs = model(inputs)
            else:
                raise e # Real error, crash.

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 1000 == 0 and i > 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")
            
    train_acc = 100 * correct / total
    print(f"Train Accuracy: {train_acc:.2f}%")
    
    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Apply the same logic to validation
            inputs = inputs.reshape(-1, 1, 100, 40)
            try:
                outputs = model(inputs)
            except:
                inputs = inputs.squeeze(1)
                outputs = model(inputs)
                
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    val_acc = 100 * correct_val / total_val
    print(f"VAL ACCURACY: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "model_v2_war_brain.pth")
        print("Model Saved!")
        
print("Training Complete.")
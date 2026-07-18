import torch
import torch.nn as nn

class DeepLOB_Simplified(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Convolution (The "Eye")
        # Input: (Batch, 1, 100, 40) -> 1 Channel (Grayscale), 100 Ticks, 40 Features
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2)), # Reduces width (40->20)
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1)), # Scans time (4 ticks)
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1, 2)), # Reduces width (20->10)
            nn.ReLU(),
        )
        
        # 2. LSTM (The "Memory")
        # Input size will be determined by the output of Conv block
        # 16 channels * 10 width = 160 features per time step
        self.lstm = nn.LSTM(input_size=160, hidden_size=64, batch_first=True)
        
        # 3. Fully Connected (The "Decision")
        self.fc = nn.Linear(64, 3) # Output: 3 probabilities (Sell, Hold, Buy)

    def forward(self, x):
        # x Shape: (Batch, 100, 40) -> We need (Batch, 1, 100, 40) for Conv2d
        x = x.unsqueeze(1) 
        
        # Run Convolutions
        h = self.conv_block(x) 
        # Output shape: (Batch, 16, 97, 10) roughly
        
        # Prepare for LSTM: (Batch, Time, Features)
        # Permute to: (Batch, Time, Channels, Width)
        h = h.permute(0, 2, 1, 3) 
        # Flatten last two dims: (Batch, Time, 16*10) -> (Batch, 97, 160)
        h = h.reshape(h.size(0), h.size(1), -1)
        
        # Run LSTM
        out, _ = self.lstm(h)
        
        # Take the very last timestamp's output
        last_hidden_state = out[:, -1, :] 
        
        # Make Prediction
        prediction = self.fc(last_hidden_state)
        return prediction

if __name__ == "__main__":
    model = DeepLOB_Simplified()
    print("Model created successfully.")
    
    # Create fake data: Batch of 32, 100 ticks, 40 features
    fake_data = torch.randn(32, 100, 40)
    output = model(fake_data)
    print(f"Input shape: {fake_data.shape}")
    print(f"Output shape: {output.shape} (Should be 32, 3)")
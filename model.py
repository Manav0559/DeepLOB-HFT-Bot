import torch
import torch.nn as nn

class DeepLOB(nn.Module):
    def __init__(self, y_len=3):
        super().__init__()
        self.y_len = y_len
        
        # Looks at small groups of orders (like a cluster of bids)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )

        # Combines the patterns to see the "Shape" of the book
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # The CNNs looked at specific snapshots. The LSTM watches.
        # It remembers what happened 10 seconds ago to understand context.
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=1, batch_first=True)
        
        # Buy (2), Sell (0), or Hold (1)
        self.fc1 = nn.Linear(64, 3)

    def forward(self, x):
        # x shape: (Batch, 1, 100, 40)
        
        # Pass through CNN Blocks
        x = self.conv1(x)
        x = self.conv2(x)
        
        # x shape is now: (Batch, 32, 100, 10)
        # Reshape it for the LSTM
        # Combine Channels (32) and Height (10) -> 320 features? 
        # Actually, for this specific standard architecture:
        x = x.permute(0, 2, 1, 3) # (Batch, 100, 32, 10)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (Batch, 100, 320)
        
        # We force it down to size 10 linear projection before LSTM to save memory
        
        # Run LSTM
        # We just want the LAST output (the prediction for "now")
        lstm_out, _ = self.lstm(x) # Input needs to match. 
        # Wait, 32*10 = 320 input size. My definition above has input_size=10.
        
        pass 
    
    
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
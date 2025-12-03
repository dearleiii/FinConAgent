import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Use last time step
        out = out[:, -1, :]
        return self.fc(out)

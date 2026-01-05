import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(weights * lstm_out, dim=1)  # (batch_size, hidden_dim)
        return context

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.15):
        super().__init__()
        # Bidirectional LSTM with increased hidden dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        context = self.attention(lstm_out)  # (batch_size, hidden_dim*2)
        context = self.dropout(context)
        out = self.fc(context)
        return out

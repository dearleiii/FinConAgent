import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import LSTMClassifier
from dataloader import NpzDataset
from losses import FocalLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

datafile_path = "stock_sequences.npz"
dataset = NpzDataset(datafile_path, seq_len=30)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LSTMClassifier().to(device)
alpha = torch.tensor([0.05, 0.45, 0.5])  # must sum to 1 or close

# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(alpha=alpha, gamma=2)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,       # LR = LR * 0.5
    patience=3,       # wait 3 epochs with no improvement
    min_lr=1e-7
)

train_losses = []   # <--- store each step loss

for epoch in range(100):
    for seq, label in loader:
        seq = seq.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(seq)
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())   # <--- save loss each step

    print("epoch:", epoch, "loss:", loss.item())
    # Step scheduler *after epoch* with current loss
    scheduler.step(loss)


window = 50
smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
log_smoothed = np.log10(smoothed + 1e-8)

plt.figure(figsize=(8,4))
plt.plot(log_smoothed, label="Training loss", linewidth=1)
plt.xlabel("Training step")
plt.ylabel("Loss")
# plt.title("Training Loss Curve")
plt.plot(smoothed, label="Smoothed loss (50-step avg)")

plt.legend()
plt.grid(True)

plt.savefig("training_loss_curve.png", dpi=300, bbox_inches="tight")

plt.show()

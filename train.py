import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import LSTMClassifier
from dataloader import NpzDataset
from losses import FocalLoss
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

datafile_path = "stock_sequences.npz"
dataset = NpzDataset(datafile_path, seq_len=30)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LSTMClassifier(dropout=0.15).to(device)

# Compute class weights based on balanced dataset
data = np.load(datafile_path)
y_data = data['y']
unique, counts = np.unique(y_data, return_counts=True)
class_weights = 1.0 / counts  # Inverse frequency weighting
class_weights = class_weights / class_weights.sum()  # Normalize
alpha = torch.tensor(class_weights, dtype=torch.float32)

print(f"Class weights: {alpha}")

criterion = FocalLoss(alpha=alpha, gamma=2)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.7,       # LR = LR * 0.7 (gentler reduction)
    patience=5,       # wait 5 epochs
    min_lr=1e-8,
    verbose=True
)

train_losses = []   # <--- store each step loss
epoch_losses = []

for epoch in range(300):  # Increased epochs
    epoch_loss = 0
    batch_count = 0
    
    for seq, label in loader:
        seq = seq.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(seq)
        loss = criterion(pred, label)

        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses.append(loss.item())   # <--- save loss each step
        epoch_loss += loss.item()
        batch_count += 1

    avg_epoch_loss = epoch_loss / batch_count
    epoch_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch:3d} | Avg Loss: {avg_epoch_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Step scheduler *after epoch* with average epoch loss
    scheduler.step(avg_epoch_loss)



###########   Visualization   ###########
window = 50
smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,4))
plt.plot(smoothed, label="Smoothed loss (50-step avg)", linewidth=1)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.legend()
plt.grid(True)

plt.savefig("training_loss_curve.png", dpi=300, bbox_inches="tight")

plt.show()

# -----------------------------
# 4. Evaluate per-class accuracy
# -----------------------------
test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
correct_per_class = defaultdict(int)
total_per_class = defaultdict(int)

all_labels = []
all_predictions = []

with torch.no_grad():
    for seq, label in test_loader:
        seq = seq.to(device)
        label = label.to(device)

        pred = model(seq)
        predicted_class = pred.argmax(dim=1)

        # Store all labels and predictions for F1 score calculation
        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predicted_class.cpu().numpy())

        for l, p in zip(label, predicted_class):
            total_per_class[l.item()] += 1
            if l.item() == p.item():
                correct_per_class[l.item()] += 1

# -----------------------------
# 5. Print per-class accuracy
# -----------------------------
for cls in range(3):
    correct = correct_per_class.get(cls, 0)
    total = total_per_class.get(cls, 0)
    acc = 100 * correct / total if total > 0 else 0
    print(f"Class {cls}: Accuracy {acc:.2f}% ({correct}/{total})")

# -----------------------------
# 6. Compute and print F1 scores
# -----------------------------
print("\n" + "="*50)
print("F1 Score Metrics")
print("="*50)

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Macro F1 (average of per-class F1 scores, treats all classes equally)
f1_macro = f1_score(all_labels, all_predictions, average='macro')
print(f"\nMacro F1 Score: {f1_macro:.4f}")

# Weighted F1 (weighted by class support)
f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
print(f"Weighted F1 Score: {f1_weighted:.4f}")

# Micro F1 (global average)
f1_micro = f1_score(all_labels, all_predictions, average='micro')
print(f"Micro F1 Score: {f1_micro:.4f}")

# Per-class F1 scores
f1_per_class = f1_score(all_labels, all_predictions, average=None)
precision_per_class = precision_score(all_labels, all_predictions, average=None)
recall_per_class = recall_score(all_labels, all_predictions, average=None)

print("\nPer-Class Metrics:")
print("-" * 50)
for cls in range(3):
    print(f"Class {cls}:")
    print(f"  Precision: {precision_per_class[cls]:.4f}")
    print(f"  Recall:    {recall_per_class[cls]:.4f}")
    print(f"  F1 Score:  {f1_per_class[cls]:.4f}")

# Detailed classification report
print("\n" + "="*50)
print("Detailed Classification Report")
print("="*50)
print(classification_report(all_labels, all_predictions, digits=4))
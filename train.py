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

model = LSTMClassifier(input_dim=8, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.15).to(device)

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

# ==========================================
# STEP 5: Visualize Prediction Accuracy
# ==========================================
print("\n" + "="*50)
print("Generating Prediction Visualizations")
print("="*50)

# Create correct/incorrect mask
correct_mask = all_labels == all_predictions
incorrect_mask = ~correct_mask

# Label names and colors
label_names = {0: 'No', 1: 'Up', 2: 'Down'}
class_colors = {0: 'gray', 1: 'green', 2: 'red'}

# Plot 1: Correct vs Incorrect by Class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart - Accuracy per class
ax = axes[0]
class_accuracies = []
for cls in range(3):
    mask = all_labels == cls
    if mask.sum() > 0:
        acc = (correct_mask[mask].sum() / mask.sum()) * 100
    else:
        acc = 0
    class_accuracies.append(acc)

bars = ax.bar(
    [label_names[i] for i in range(3)],
    class_accuracies,
    color=[class_colors[i] for i in range(3)],
    alpha=0.7,
    edgecolor='black',
    linewidth=2
)

ax.set_title("Per-Class Prediction Accuracy", fontsize=12, fontweight='bold')
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars, class_accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{acc:.1f}%',
            ha='center', va='bottom', fontweight='bold')

# Pie chart - Correct vs Incorrect overall
ax = axes[1]
correct_count = correct_mask.sum()
incorrect_count = incorrect_mask.sum()

wedges, texts, autotexts = ax.pie(
    [correct_count, incorrect_count],
    labels=['Correct', 'Incorrect'],
    colors=['green', 'red'],
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)

ax.set_title(f"Overall Prediction Accuracy ({correct_count}/{len(all_labels)})", 
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("prediction_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Prediction Distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for cls in range(3):
    ax = axes[cls]
    mask = all_labels == cls
    
    if mask.sum() == 0:
        ax.text(0.5, 0.5, "No samples", ha='center', va='center', fontsize=12)
        ax.set_title(f"Class {cls} ({label_names[cls]})", fontweight='bold')
        continue
    
    # Get predictions for this class
    pred_dist = all_predictions[mask]
    correct_preds = pred_dist[correct_mask[mask]]
    incorrect_preds = pred_dist[incorrect_mask[mask]]
    
    # Count correct predictions per predicted class
    correct_by_pred = np.bincount(correct_preds.astype(int), minlength=3)
    
    # Count incorrect predictions per predicted class
    incorrect_by_pred = np.bincount(incorrect_preds.astype(int), minlength=3)
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, correct_by_pred, width, label='Correct', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, incorrect_by_pred, width, label='Incorrect', color='red', alpha=0.7)
    
    ax.set_title(f"True Class {cls} ({label_names[cls]}) - {mask.sum()} samples", fontweight='bold')
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels([label_names[i] for i in range(3)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("prediction_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 3: Confusion Matrix Heatmap
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_predictions)
fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(cm, cmap='Blues', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))
ax.set_xticklabels([label_names[i] for i in range(3)])
ax.set_yticklabels([label_names[i] for i in range(3)])

# Labels
ax.set_xlabel('Predicted Label', fontweight='bold')
ax.set_ylabel('True Label', fontweight='bold')
ax.set_title('Confusion Matrix', fontweight='bold', fontsize=12)

# Add text annotations
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, cm[i, j],
                      ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                      fontweight='bold', fontsize=12)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Visualizations saved:")
print("   - prediction_accuracy.png")
print("   - prediction_distribution.png")
print("   - confusion_matrix.png")

# ==========================================
# STEP 6: Visualize Sequences with Predictions
# ==========================================
print("\n" + "="*50)
print("Generating Sequence Prediction Visualizations")
print("="*50)

# Load sequences
data = np.load(datafile_path)
X = data['X']
y = data['y']

print(f"Loaded {len(X)} sequences")

# Get predictions for all sequences (need to run model on full dataset)
all_seq_predictions = []
model.eval()
with torch.no_grad():
    for i in range(0, len(X), 32):
        batch_X = torch.tensor(X[i:i+32], dtype=torch.float32).to(device)
        batch_pred = model(batch_X)
        batch_pred_class = batch_pred.argmax(dim=1).cpu().numpy()
        all_seq_predictions.extend(batch_pred_class)

all_seq_predictions = np.array(all_seq_predictions)
seq_correct = (y == all_seq_predictions)
seq_incorrect = ~seq_correct

print(f"Correct predictions: {seq_correct.sum()}")
print(f"Incorrect predictions: {seq_incorrect.sum()}")

# Feature columns
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'MA10', 'MA20', 'vwap']
label_names = {0: 'No', 1: 'Up', 2: 'Down'}
class_colors = {0: 'gray', 1: 'green', 2: 'red'}

# Plot 1: Sample Correct Predictions
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Sample Correct Predictions (True Label vs Predicted)', fontsize=14, fontweight='bold')

correct_indices = np.where(seq_correct)[0]
sample_correct = np.random.choice(correct_indices, min(6, len(correct_indices)), replace=False)

for idx, seq_idx in enumerate(sample_correct):
    ax = axes[idx // 2, idx % 2]
    
    seq = X[seq_idx]
    true_label = int(y[seq_idx])
    pred_label = int(all_seq_predictions[seq_idx])
    
    # Plot close price (feature index 3)
    ax.plot(seq[:, 3], 'o-', linewidth=2, markersize=6, 
            color=class_colors[true_label], alpha=0.7, label='Close Price')
    
    # Add markers for true and predicted labels
    ax.scatter([len(seq)-1], [seq[-1, 3]], s=300, marker='*', 
              color=class_colors[true_label], edgecolors='black', linewidth=2,
              label=f'True: {label_names[true_label]}', zorder=5)
    
    ax.set_title(f"Seq {seq_idx} | True: {label_names[true_label]} → Pred: {label_names[pred_label]} ✓",
                fontweight='bold', color='green')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Close Price')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("correct_predictions_sequences.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Sample Incorrect Predictions
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Sample Incorrect Predictions (True vs Predicted)', fontsize=14, fontweight='bold')

incorrect_indices = np.where(seq_incorrect)[0]
sample_incorrect = np.random.choice(incorrect_indices, min(6, len(incorrect_indices)), replace=False)

for idx, seq_idx in enumerate(sample_incorrect):
    ax = axes[idx // 2, idx % 2]
    
    seq = X[seq_idx]
    true_label = int(y[seq_idx])
    pred_label = int(all_seq_predictions[seq_idx])
    
    # Plot close price
    ax.plot(seq[:, 3], 'o-', linewidth=2, markersize=6, 
            color=class_colors[true_label], alpha=0.7, label='Close Price')
    
    # Add markers for true and predicted labels
    ax.scatter([len(seq)-1], [seq[-1, 3]], s=300, marker='*', 
              color=class_colors[true_label], edgecolors='black', linewidth=2,
              label=f'True: {label_names[true_label]}', zorder=5)
    
    ax.scatter([len(seq)-1], [seq[-1, 3]], s=200, marker='x', 
              color=class_colors[pred_label], linewidth=3,
              label=f'Pred: {label_names[pred_label]}', zorder=4)
    
    ax.set_title(f"Seq {seq_idx} | True: {label_names[true_label]} → Pred: {label_names[pred_label]} ✗",
                fontweight='bold', color='red')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Close Price')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("incorrect_predictions_sequences.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 3: Feature Distribution for Correct vs Incorrect
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Feature Patterns: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')

feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'VWAP']

for feat_idx, feat_name in enumerate(feature_names):
    ax = axes[feat_idx // 4, feat_idx % 4]
    
    # Get last value of each feature for correct and incorrect predictions
    correct_feat_values = X[seq_correct, -1, feat_idx]
    incorrect_feat_values = X[seq_incorrect, -1, feat_idx]
    
    # Create violin plot
    parts = ax.violinplot([correct_feat_values, incorrect_feat_values], 
                          positions=[1, 2], showmeans=True)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Correct', 'Incorrect'])
    ax.set_ylabel(f'{feat_name} Value')
    ax.set_title(f'{feat_name} Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("feature_distribution_by_prediction.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 4: Prediction Correctness by Feature Values
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Prediction Correctness by Feature Values', fontsize=14, fontweight='bold')

for feat_idx, feat_name in enumerate(feature_names):
    ax = axes[feat_idx // 4, feat_idx % 4]
    
    # Get last value of each feature
    feat_values = X[:, -1, feat_idx]
    
    # Create scatter plot
    ax.scatter(feat_values[seq_correct], all_seq_predictions[seq_correct],
              alpha=0.5, c='green', s=50, label='Correct', marker='o')
    ax.scatter(feat_values[seq_incorrect], all_seq_predictions[seq_incorrect],
              alpha=0.5, c='red', s=50, label='Incorrect', marker='x')
    
    ax.set_xlabel(f'{feat_name} Value')
    ax.set_ylabel('Predicted Label')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(list(label_names.values()))
    ax.set_title(f'{feat_name} vs Prediction', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("feature_vs_prediction_correctness.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Sequence Visualizations saved:")
print("   - correct_predictions_sequences.png")
print("   - incorrect_predictions_sequences.png")
print("   - feature_distribution_by_prediction.png")
print("   - feature_vs_prediction_correctness.png")
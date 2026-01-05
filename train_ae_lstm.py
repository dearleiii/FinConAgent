"""
Training Experiment: LSTM Classifier with ConvAutoencoder Feature Extraction

Architecture:
1. ConvAutoencoder: Extracts latent features from raw sequences [batch, seq_len, features]
2. LSTMClassifier: Takes latent features and predicts stock price direction

This approach decomposes the problem into:
- Low-level feature learning (ConvAutoencoder)
- High-level temporal pattern recognition (LSTM)
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.lstm import LSTMClassifier
from models.rpca import ConvAutoencoder
from dataloader import NpzDataset
from losses import FocalLoss
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ==========================================
    # STEP 1: Data Loading (Optimized)
    # ==========================================
    datafile_path = "stock_sequences.npz"
    dataset = NpzDataset(datafile_path)  # Applies z-score normalization per feature

    # Optimized DataLoader: GPU pinning, larger batch
    # Note: num_workers=0 for Windows compatibility (multiprocessing issues)
    num_workers = 0  # Set to 0 for Windows; use 4 for Linux/Mac if desired
    loader = DataLoader(
        dataset,
        batch_size=64,  # Increased from 32 for better GPU utilization
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',  # Pin memory for faster GPU transfer
        drop_last=True  # Drop incomplete batches for consistent batch size
    )

    print(f"Dataset loaded: {len(dataset)} sequences")
    print(f"Sequence shape: {dataset.X.shape}")
    print(f"DataLoader config: batch_size={loader.batch_size}, num_workers={num_workers}, pin_memory={device.type == 'cuda'}")
    print(f"Data normalization: Z-score (per-feature mean=0, std=1)")
    print(f"  Feature means: {dataset.feature_mean}")
    print(f"  Feature stds:  {dataset.feature_std}")

    class LatentFeatureExtractor(nn.Module):
        """Wrapper to extract latent features from ConvAutoencoder"""
        def __init__(self, autoencoder):
            super().__init__()
            self.autoencoder = autoencoder

        def forward(self, x):
            """
            Extract latent features from raw sequences
            x: [batch, seq_len, features]
            Returns: [batch, latent_dim]
            """
            # Reshape to 4D for ConvAutoencoder
            x_4d = x.unsqueeze(1)  # [batch, 1, seq_len, features]

            # Encoder: [batch, 1, seq_len, features] -> [batch, 32, h, w]
            encoder_output = self.autoencoder.encoder(x_4d)

            # Flatten: [batch, 32*h*w]
            batch_size = encoder_output.shape[0]
            flattened = encoder_output.view(batch_size, -1)

            # FC layer: [batch, 32*h*w] -> [batch, latent_dim]
            latent = self.autoencoder.fc_enc(flattened)

            return latent  # [batch, latent_dim]


    # ==========================================
    # STEP 2: Initialize Models
    # ==========================================

    # Autoencoder for feature extraction
    # Note: Sequence length is 25 (from stock_sequences.npz shape: 581, 25, 8)
    autoencoder = ConvAutoencoder(latent_dim=3, input_shape=(1, 25, 8)).to(device)

    # Feature extractor wrapper
    feature_extractor = LatentFeatureExtractor(autoencoder).to(device)

    # LSTM classifier takes latent features
    # Input: [batch, seq_len=1, latent_dim=3] -> Output: [batch, num_classes=3]
    model = LSTMClassifier(input_dim=3, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.15).to(device)

    print("="*60)
    print("Model Architecture")
    print("="*60)
    print(f"ConvAutoencoder: Input (1, 25, 8) -> Latent dim: 3")
    print(f"LSTMClassifier: Input (seq_len=1, latent_dim=3) -> Output: 3 classes")
    print("="*60)

    # ==========================================
    # STEP 3: Training Setup
    # ==========================================

    # Compute class weights
    data = np.load(datafile_path)
    y_data = data['y']
    unique, counts = np.unique(y_data, return_counts=True)
    class_weights = 1.0 / counts
    class_weights = class_weights / class_weights.sum()
    alpha = torch.tensor(class_weights, dtype=torch.float32)

    print(f"Class weights: {alpha}")

    criterion = FocalLoss(alpha=alpha, gamma=2)

    # ==========================================
    # STEP 4: Pretrain ConvAutoencoder
    # ==========================================

    print("\n" + "="*60)
    print("PHASE 1: Pretraining ConvAutoencoder")
    print("="*60)
    print("Training objective: Minimize reconstruction loss (MSE)")
    print("Input data: Z-score normalized (already applied by dataloader)")
    print("="*60 + "\n")

    # Optimizer for autoencoder pretraining
    # Note: Higher learning rate (1e-2) for faster initial learning
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-2, weight_decay=1e-5)
    ae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ae_optimizer,
        mode='min',
        factor=0.5,  # More aggressive decay
        patience=10,  # More patience before decay
        min_lr=1e-6,
        verbose=True  # Show scheduler updates
    )

    ae_losses = []
    ae_epoch_losses = []

    # Train autoencoder to minimize reconstruction loss
    autoencoder.train()

    # Pre-compute padding size to avoid computing each iteration
    dummy_input = torch.zeros(1, 1, 25, 8).to(device)
    dummy_output = autoencoder(dummy_input)
    pad_h = dummy_output.shape[2] - 25
    pad_w = dummy_output.shape[3] - 8
    needs_padding = (pad_h > 0) or (pad_w > 0)  # Check once

    for epoch in range(600):  # 1000 epochs for better convergence on full dataset
        epoch_loss = 0.0
        batch_count = 0
        grad_norms = []

        for seq, label in loader:
            seq = seq.to(device, non_blocking=True)  # [batch, seq_len, features] - async GPU transfer

            # Reshape to 4D for autoencoder
            seq_4d = seq.unsqueeze(1)  # [batch, 1, seq_len, features]

            # Forward pass: reconstruct input
            reconstructed = autoencoder(seq_4d)  # [batch, 1, seq_len, features_out]

            # Handle shape mismatch: pad target to match reconstructed shape (optimized)
            if needs_padding:
                seq_padded = F.pad(seq_4d, (0, pad_w, 0, pad_h))
            else:
                seq_padded = seq_4d

            # Reconstruction loss (MSE on normalized data)
            loss = F.mse_loss(reconstructed, seq_padded)

            ae_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping and monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            grad_norms.append(grad_norm.item())
            
            ae_optimizer.step()

            ae_losses.append(loss.item())
            epoch_loss += loss.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count
        avg_grad_norm = np.mean(grad_norms)
        ae_epoch_losses.append(avg_epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | AE Recon Loss: {avg_epoch_loss:.6f} | Grad Norm: {avg_grad_norm:.6f} | LR: {ae_optimizer.param_groups[0]['lr']:.2e}")

        ae_scheduler.step(avg_epoch_loss)

    print("\n✅ Autoencoder Pretraining Complete")
    print(f"Final reconstruction loss: {ae_epoch_losses[-1]:.6f}")

    # ==========================================
    # Test Reconstruction Quality
    # ==========================================

    print("\n" + "="*60)
    print("Testing Reconstruction Quality")
    print("="*60)

    autoencoder.eval()
    with torch.no_grad():
        # Get a batch for testing
        test_seq, test_label = next(iter(loader))
        test_seq = test_seq.to(device)

        # Forward pass
        test_seq_4d = test_seq.unsqueeze(1)
        reconstructed = autoencoder(test_seq_4d)

        # Crop reconstruction to match input size
        reconstructed = reconstructed[:, :, :test_seq_4d.shape[2], :test_seq_4d.shape[3]].squeeze(1)

        # Denormalize for comparison (use dataset normalization stats)
        test_seq_denorm = test_seq.cpu().numpy() * dataset.feature_std + dataset.feature_mean
        reconstructed_denorm = reconstructed.cpu().numpy() * dataset.feature_std + dataset.feature_mean

        # Compute errors
        normalized_mse = F.mse_loss(reconstructed, test_seq).item()
        denormalized_mse = np.mean((test_seq_denorm - reconstructed_denorm) ** 2)

        print(f"\nReconstruction Metrics (on test batch):")
        print(f"  Normalized MSE:   {normalized_mse:.6f}")
        print(f"  Denormalized MSE: {denormalized_mse:.6f}")

        # Per-feature analysis
        print(f"\nPer-Feature Denormalized MAE:")
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'VWAP']
        for feat_idx in range(test_seq_denorm.shape[2]):
            mae = np.mean(np.abs(test_seq_denorm[:, :, feat_idx] - reconstructed_denorm[:, :, feat_idx]))
            print(f"  {feature_names[feat_idx]:8s}: {mae:.6f}")

        # Sample comparison
        sample_idx = 0
        close_idx = 3  # Close price feature index
        print(f"\nSample {sample_idx} Close Price Comparison (Feature index: {close_idx}):")
        print(f"  Original normalized:     {test_seq[sample_idx, 0, close_idx]:.6f} → denorm: {test_seq_denorm[sample_idx, 0, close_idx]:.6f}")
        print(f"  Reconstructed normalized: {reconstructed[sample_idx, 0, close_idx]:.6f} → denorm: {reconstructed_denorm[sample_idx, 0, close_idx]:.6f}")
        print(f"  Mean error (denormalized): {np.mean(np.abs(test_seq_denorm[sample_idx, :, close_idx] - reconstructed_denorm[sample_idx, :, close_idx])):.6f}")

    print("="*60)

    # Freeze autoencoder weights
    for param in autoencoder.parameters():
        param.requires_grad = False

    print("✅ Autoencoder weights frozen for use as feature extractor\n")

    # ==========================================
    # STEP 5: Train LSTM Classifier
    # ==========================================

    print("="*60)
    print("PHASE 2: Training LSTM Classifier with Frozen AE Features")
    print("="*60 + "\n")

    # Optimizer for LSTM (autoencoder is frozen as feature extractor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=5,
        min_lr=1e-8,
        verbose=True
    )

    train_losses = []
    epoch_losses = []

    feature_extractor.eval()  # Freeze autoencoder as feature extractor

    for epoch in range(300):
        epoch_loss = 0.0
        batch_count = 0

        for seq, label in loader:
            seq = seq.to(device, non_blocking=True)  # [batch, seq_len, features] - async GPU transfer
            label = label.to(device, non_blocking=True)

            # Extract latent features from autoencoder (no gradients) - inlined for speed
            with torch.no_grad():
                seq_4d = seq.unsqueeze(1)
                encoder_output = feature_extractor.autoencoder.encoder(seq_4d)
                batch_size = encoder_output.shape[0]
                flattened = encoder_output.view(batch_size, -1)
                latent_features = feature_extractor.autoencoder.fc_enc(flattened)
                latent_features = latent_features.unsqueeze(1)  # [batch, 1, latent_dim]

            optimizer.zero_grad()
            pred = model(latent_features)  # [batch, num_classes]
            loss = criterion(pred, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            epoch_loss += loss.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count
        epoch_losses.append(avg_epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Avg Loss: {avg_epoch_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(avg_epoch_loss)

    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)

    # ==========================================
    # STEP 6: Visualize Training Loss (AE + LSTM)
    # ==========================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Autoencoder Pretraining Loss
    ax = axes[0]
    ae_window = 20
    ae_smoothed = np.convolve(ae_losses, np.ones(ae_window)/ae_window, mode='valid')
    ax.plot(ae_smoothed, label="Smoothed AE recon loss (20-step avg)", linewidth=2, color='blue')
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reconstruction Loss (MSE)")
    ax.set_title("PHASE 1: Autoencoder Pretraining Loss")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: LSTM Training Loss
    ax = axes[1]
    lstm_window = 50
    lstm_smoothed = np.convolve(train_losses, np.ones(lstm_window)/lstm_window, mode='valid')
    ax.plot(lstm_smoothed, label="Smoothed LSTM loss (50-step avg)", linewidth=2, color='green')
    ax.set_xlabel("Training step")
    ax.set_ylabel("Focal Loss")
    ax.set_title("PHASE 2: LSTM Classifier Training Loss")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_loss_ae_lstm.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n✅ Loss visualization saved: training_loss_ae_lstm.png")

    # ==========================================
    # STEP 7: Evaluate per-class accuracy
    # ==========================================

    print("\n" + "="*60)
    print("Evaluating Model Performance")
    print("="*60)

    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    all_labels = []
    all_predictions = []

    feature_extractor.eval()
    model.eval()

    with torch.no_grad():
        for seq, label in test_loader:
            seq = seq.to(device)
            label = label.to(device)

            # Extract latent features
            latent_features = feature_extractor(seq)  # [batch, latent_dim]
            latent_features = latent_features.unsqueeze(1)  # [batch, 1, latent_dim]

            pred = model(latent_features)
            predicted_class = pred.argmax(dim=1)

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

            for l, p in zip(label, predicted_class):
                total_per_class[l.item()] += 1
                if l.item() == p.item():
                    correct_per_class[l.item()] += 1

    # ==========================================
    # STEP 8: Print per-class accuracy
    # ==========================================

    print("\nPer-Class Accuracy:")
    print("-" * 50)
    for cls in range(3):
        correct = correct_per_class.get(cls, 0)
        total = total_per_class.get(cls, 0)
        acc = 100 * correct / total if total > 0 else 0
        print(f"Class {cls}: Accuracy {acc:.2f}% ({correct}/{total})")

    # ==========================================
    # STEP 9: Compute and print F1 scores
    # ==========================================

    print("\n" + "="*60)
    print("F1 Score Metrics")
    print("="*60)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    f1_micro = f1_score(all_labels, all_predictions, average='micro')

    print(f"\nMacro F1 Score:    {f1_macro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Micro F1 Score:    {f1_micro:.4f}")

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

    print("\n" + "="*60)
    print("Detailed Classification Report")
    print("="*60)
    print(classification_report(all_labels, all_predictions, digits=4))

    # ==========================================
    # STEP 10: Visualize Prediction Accuracy
    # ==========================================

    print("\n" + "="*60)
    print("Generating Prediction Visualizations")
    print("="*60)

    # Create correct/incorrect mask
    correct_mask = all_labels == all_predictions
    incorrect_mask = ~correct_mask

    # Label names and colors
    label_names = {0: 'No', 1: 'Up', 2: 'Down'}
    class_colors = {0: 'gray', 1: 'green', 2: 'red'}

    # Plot 1: Accuracy per class + Overall pie chart
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
    plt.savefig("prediction_accuracy_ae_lstm.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Plot 2: Prediction Distribution by True Class
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
    plt.savefig("prediction_distribution_ae_lstm.png", dpi=150, bbox_inches='tight')
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
    plt.savefig("confusion_matrix_ae_lstm.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✅ Visualizations saved:")
    print("   - prediction_accuracy_ae_lstm.png")
    print("   - prediction_distribution_ae_lstm.png")
    print("   - confusion_matrix_ae_lstm.png")

    print("\n" + "="*60)
    print("✅ Training experiment complete!")
    print("="*60)
    print(f"\nResults saved:")
    print("  - training_loss_ae_lstm.png")
    print("  - prediction_accuracy_ae_lstm.png")
    print("  - prediction_distribution_ae_lstm.png")
    print("  - confusion_matrix_ae_lstm.png")


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.rpca import ConvAutoencoder, soft_threshold
from dataloader import NpzDataset
from losses import FocalLoss
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

datafile_path = "stock_sequences.npz"
dataset = NpzDataset(datafile_path)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# model = LSTMClassifier(input_dim=8, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.15).to(device)
model = ConvAutoencoder(latent_dim=3, input_shape=(1, 25,8)).to(device)

# Compute class weights based on balanced dataset
data = np.load(datafile_path)
y_data = data['y']
unique, counts = np.unique(data, return_counts=True)


### 3️⃣ Denoising AE Training Step (for sequential data) ###
def train_autoencoder(model, X_input, X_target, device, epochs=5, lr=1e-3, verbose=False):
    """
    Train autoencoder on sequential data
    X_input: [batch_size, seq_len, features] - will be reshaped to [batch_size, 1, seq_len, features]
    X_target: [batch_size, seq_len, features]
    epochs: number of training epochs
    verbose: whether to print loss per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    
    # Reshape input to 4D for ConvAutoencoder [batch, 1, seq_len, features]
    #  i.e. torch.Size([32, 1, 25, 8]) for both input, target
    X_input_4d = X_input.unsqueeze(1)
    X_target_4d = X_target.unsqueeze(1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_input_4d)
        
        # Pad target to match output shape if needed
        if output.shape != X_target_4d.shape:
            pad_h = output.shape[2] - X_target_4d.shape[2]
            pad_w = output.shape[3] - X_target_4d.shape[3]
            X_target_padded = F.pad(X_target_4d, (0, pad_w, 0, pad_h))
        else:
            X_target_padded = X_target_4d
        
        loss = criterion(output, X_target_padded)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if verbose and epoch % max(1, epochs // 3) == 0:
            print(f"    AE Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    # Return reconstructed output, squeezed back to [batch, seq_len, features]
    model.eval()
    with torch.no_grad():
        reconstruction = model(X_input_4d)
        # Crop reconstruction back to original size for consistency
        reconstruction = reconstruction[:, :, :X_target_4d.shape[2], :X_target_4d.shape[3]].squeeze(1)
    return reconstruction.detach()


### Deep RPCA Alternating Optimization (for sequential data) ###
def deep_rpca(X, model, lam, device, mu=0.1, iters=5, ae_epochs=1000, verbose=True):
    """
    Deep RPCA for sequential data decomposition
    X: [batch_size, seq_len, features] normalized sequential data
    model: ConvAutoencoder (must be in train mode)
    lam: sparsity regularization parameter (controls sparsity sensitivity)
    ae_epochs: number of autoencoder training epochs per RPCA iteration (default=1000)
    iters: number of alternating optimization iterations
    verbose: whether to print convergence information
    
    Returns: L_hat (low-rank/reconstructed), N (sparse/noise component)
    
    Tuning Guide:
    - Increase ae_epochs (1000-5000) for better reconstruction quality
    - Increase iters (3-10) for better convergence
    - Increase lam (0.1-1.0) to detect more anomalies
    """
    # Start with no noise
    N = torch.zeros_like(X)

    for i in range(iters):
        # Low-rank update via autoencoder:  L = low-rank (clean signal)
        # N = sparse noise / anomalies
        L_input = X - N

        # L_input is passed as both X_input and X_target because the autoencoder is being used
        #  as a self-supervised denoiser / low-rank projector,
        #  not as a predictor of a different target: “Given this data, learn to reconstruct only the structured (low-rank) part of it.”
        L_hat = train_autoencoder(model, L_input, L_input, device, epochs=ae_epochs, lr=1e-4, verbose=verbose)

        # Sparse update via soft-thresholding: Keeps only large residuals
        residual = X - L_hat
        N = soft_threshold(residual, lam)

        sparsity = (N != 0).sum().item() / N.numel()
        mse_loss = torch.nn.functional.mse_loss(L_hat, X).item()
        if verbose:
            print(f"  RPCA Iter {i} | Sparsity: {sparsity:.4f} | MSE: {mse_loss:.6f} | Nonzeros: {(N!=0).sum().item()}")

    return L_hat, N



### ==========================================
### STEP 7: Test Deep RPCA Decomposition
### ==========================================
print("\n" + "="*50)
print("Testing Deep RPCA Decomposition")
print("="*50)

# Load all data from dataset
data = np.load(datafile_path)
X_np = data['X']  # Shape: [num_sequences, seq_len, features]

# Normalize the data using dataset's normalization
feature_mean = X_np.mean(axis=(0, 1))
feature_std = X_np.std(axis=(0, 1)) + 1e-8
X_normalized = (X_np - feature_mean) / feature_std

# Convert to tensor and take a batch for testing
batch_size = min(32, len(X_normalized))
X_batch = torch.tensor(X_normalized[:batch_size], dtype=torch.float32).to(device)

print(f"Input shape: {X_batch.shape}")  # Should be [batch_size, seq_len, features]
# Input shape: torch.Size([32, 25, 8])

# Reinitialize model for decomposition
model_rpca = ConvAutoencoder(latent_dim=3, input_shape=(1, 25,8)).to(device)
model_rpca.train()  # Set to training mode for RPCA iterations
print("RPCA model initialized for testing.")

# Test different lambda values
lamda_set = [0.0, 0.01, 0.1, 0.5, 1.0, 10.0]

print("\nTesting RPCA with different lambda values:")
print("-" * 50)

for lam in lamda_set:
    print(f"\nλ = {lam}")
    L, S = deep_rpca(X_batch, model_rpca, lam, device=device, iters=3, ae_epochs=1000, verbose=False)

    # Calculate MSE on flattened tensors
    mse = F.mse_loss(L.reshape(L.size(0), -1), 
                     X_batch.reshape(X_batch.size(0), -1))
    
    # Calculate reconstruction ratio
    recon_ratio = torch.norm(L) / torch.norm(X_batch)
    sparse_ratio = torch.norm(S) / torch.norm(X_batch)
    
    print(f"  MSE: {mse.item():.6f}")
    print(f"  Reconstruction ratio: {recon_ratio.item():.6f}")
    print(f"  Sparse ratio: {sparse_ratio.item():.6f}")


### ==========================================
### STEP 8: Visualize RPCA Decomposition Results
### ==========================================
print("\n" + "="*50)
print("Visualizing RPCA Decomposition Results")
print("="*50)

# Store RPCA results for all lambda values
rpca_results = {}

for lam in lamda_set:
    print(f"Processing λ = {lam}...")
    L, S = deep_rpca(X_batch, model_rpca, lam, device=device, iters=3, ae_epochs=1000, verbose=False)
    rpca_results[lam] = {
        'L': L.cpu().numpy(),
        'S': S.cpu().numpy(),
        'anomaly_scores': np.abs(S.cpu().numpy()).sum(axis=2)
    }

# Feature names
feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'VWAP']
label_names = {0: 'No', 1: 'Up', 2: 'Down'}

# Plot 1: Compare Reconstructions Across Lambda Values for Sample Sequence
print("\nGenerating multi-lambda reconstruction comparison...")
sample_idx = 0
X_sample = X_batch[sample_idx].cpu().numpy()

fig, axes = plt.subplots(len(lamda_set), 3, figsize=(16, 4*len(lamda_set)))
if len(lamda_set) == 1:
    axes = axes.reshape(1, -1)

fig.suptitle(f'RPCA Decomposition: Original vs Reconstructed vs Sparse (Sequence {sample_idx})', 
             fontsize=14, fontweight='bold', y=0.995)

for row, lam in enumerate(lamda_set):
    L_sample = rpca_results[lam]['L'][sample_idx]
    S_sample = rpca_results[lam]['S'][sample_idx]
    
    # Plot 1: Original Signal (normalize per-feature for better visualization)
    ax = axes[row, 0]
    X_normalized_viz = X_sample.copy()
    for feat_idx in range(X_sample.shape[1]):
        feat_min = X_sample[:, feat_idx].min()
        feat_max = X_sample[:, feat_idx].max()
        if feat_max > feat_min:
            X_normalized_viz[:, feat_idx] = (X_sample[:, feat_idx] - feat_min) / (feat_max - feat_min)
    im = ax.imshow(X_normalized_viz.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title(f'λ={lam:.2f} | Original Signal (per-feature normalized)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized [0-1]')
    
    # Plot 2: Reconstructed Signal (normalize per-feature)
    ax = axes[row, 1]
    L_normalized_viz = L_sample.copy()
    for feat_idx in range(L_sample.shape[1]):
        feat_min = L_sample[:, feat_idx].min()
        feat_max = L_sample[:, feat_idx].max()
        if feat_max > feat_min:
            L_normalized_viz[:, feat_idx] = (L_sample[:, feat_idx] - feat_min) / (feat_max - feat_min)
    im = ax.imshow(L_normalized_viz.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title(f'λ={lam:.2f} | Reconstructed L (per-feature normalized)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized [0-1]')
    
    # Plot 3: Sparse Component (Anomalies) - use absolute value for better visibility
    ax = axes[row, 2]
    S_abs = np.abs(S_sample)
    S_normalized_viz = S_abs.copy()
    for feat_idx in range(S_abs.shape[1]):
        feat_max = S_abs[:, feat_idx].max()
        if feat_max > 0:
            S_normalized_viz[:, feat_idx] = S_abs[:, feat_idx] / feat_max
    im = ax.imshow(S_normalized_viz.T, aspect='auto', cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title(f'λ={lam:.2f} | Sparse |S| (Anomalies)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Magnitude')

plt.tight_layout()
plt.savefig("rpca_decomposition_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Anomaly Score Heatmap (Time × Lambda)
print("\nGenerating anomaly score heatmap across lambda values...")
fig, axes = plt.subplots(1, len(lamda_set), figsize=(4*len(lamda_set), 8))
if len(lamda_set) == 1:
    axes = [axes]

fig.suptitle('Anomaly Scores (Sum |S| across features) vs Lambda Values', 
             fontsize=14, fontweight='bold')

for col, lam in enumerate(lamda_set):
    ax = axes[col]
    anomaly_scores = rpca_results[lam]['anomaly_scores']
    
    im = ax.imshow(anomaly_scores, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Sequence Index')
    ax.set_title(f'λ = {lam:.2f}', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Anomaly Score')

plt.tight_layout()
plt.savefig("rpca_anomaly_scores_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 3: Time Series Decomposition with Denormalization
print("\nGenerating denormalized time series decomposition...")
sample_idx = 0
X_sample_norm = X_batch[sample_idx].cpu().numpy()
X_sample = X_sample_norm * feature_std + feature_mean  # Denormalize

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, len(lamda_set), hspace=0.35, wspace=0.35)

fig.suptitle(f'✅ RPCA Decomposition: Original vs Reconstructed | Sparse Heatmap | Anomaly Score (Sequence {sample_idx})', 
             fontsize=14, fontweight='bold')

for lam_idx, lam in enumerate(lamda_set):
    L_sample_norm = rpca_results[lam]['L'][sample_idx]
    S_sample_norm = rpca_results[lam]['S'][sample_idx]
    
    # Denormalize
    L_sample = L_sample_norm * feature_std + feature_mean
    
    # Calculate anomaly score per timestep
    anomaly_scores = np.abs(S_sample_norm).sum(axis=1)
    sparsity_pct = (S_sample_norm != 0).sum() / S_sample_norm.size * 100
    
    time_steps = np.arange(len(X_sample))
    
    # Row 0: Original vs Reconstructed (Close Price)
    ax = fig.add_subplot(gs[0, lam_idx])
    ax.plot(time_steps, X_sample[:, 3], 'b-', linewidth=2.5, marker='o', markersize=5, 
            label='✅ Original', alpha=0.9)
    ax.plot(time_steps, L_sample[:, 3], 'g--', linewidth=2, marker='s', markersize=5,
            label='✅ Reconstructed', alpha=0.8)
    ax.fill_between(time_steps, X_sample[:, 3], L_sample[:, 3], alpha=0.15, color='orange')
    ax.set_ylabel('Close Price', fontsize=9, fontweight='bold')
    ax.set_title(f'λ={lam:.2f} | Original vs Reconstructed', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    
    # Row 1: Sparse Component Heatmap
    ax = fig.add_subplot(gs[1, lam_idx])
    S_abs = np.abs(S_sample_norm)
    im = ax.imshow(S_abs.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_ylabel('Feature', fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title(f'✅ Sparse Component |S|', fontsize=10, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, label='|S|')
    cbar.ax.tick_params(labelsize=7)
    ax.tick_params(labelsize=8)
    
    # Row 2: Anomaly Score Over Time
    ax = fig.add_subplot(gs[2, lam_idx])
    colors = ['red' if score > 0 else 'lightgray' for score in anomaly_scores]
    ax.bar(time_steps, anomaly_scores, color=colors, alpha=0.8, edgecolor='darkred', linewidth=0.5)
    ax.plot(time_steps, anomaly_scores, 'r-', linewidth=1.5, alpha=0.6)
    ax.set_ylabel('Anomaly Score', fontsize=9, fontweight='bold')
    ax.set_title(f'✅ Anomaly Score (Sparsity: {sparsity_pct:.1f}%)', 
                fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=8)
    
    # Row 3: Residuals
    ax = fig.add_subplot(gs[3, lam_idx])
    residuals = X_sample[:, 3] - L_sample[:, 3]
    colors_res = ['red' if res > 0 else 'blue' for res in residuals]
    ax.bar(time_steps, residuals, color=colors_res, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Residual', fontsize=9, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=9, fontweight='bold')
    ax.set_title(f'Close Price Residuals', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=8)

plt.savefig("rpca_timeseries_decomposition.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 4: Statistics Across Lambda Values
print("\nGenerating lambda tuning statistics...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RPCA Metrics vs Lambda', fontsize=14, fontweight='bold')

sparsity_by_lambda = []
recon_error_by_lambda = []
sparse_norm_by_lambda = []

for lam in lamda_set:
    L = torch.from_numpy(rpca_results[lam]['L']).to(device)
    S = torch.from_numpy(rpca_results[lam]['S']).to(device)
    
    sparsity = (S != 0).sum().item() / S.numel()
    recon_error = F.mse_loss(L, X_batch).item()
    sparse_norm = torch.norm(S).item()
    
    sparsity_by_lambda.append(sparsity)
    recon_error_by_lambda.append(recon_error)
    sparse_norm_by_lambda.append(sparse_norm)

# Sparsity vs Lambda
ax = axes[0, 0]
ax.plot(lamda_set, sparsity_by_lambda, 'o-', linewidth=2.5, markersize=8, color='red')
ax.set_xlabel('Lambda (λ)', fontweight='bold')
ax.set_ylabel('Sparsity Ratio', fontweight='bold')
ax.set_xscale('log')
ax.set_title('Sparsity vs Lambda', fontweight='bold')
ax.grid(True, alpha=0.3)

# Reconstruction Error vs Lambda
ax = axes[0, 1]
ax.plot(lamda_set, recon_error_by_lambda, 'o-', linewidth=2.5, markersize=8, color='green')
ax.set_xlabel('Lambda (λ)', fontweight='bold')
ax.set_ylabel('Reconstruction MSE', fontweight='bold')
ax.set_xscale('log')
ax.set_title('Reconstruction Error vs Lambda', fontweight='bold')
ax.grid(True, alpha=0.3)

# Sparse Component Norm vs Lambda
ax = axes[1, 0]
ax.plot(lamda_set, sparse_norm_by_lambda, 'o-', linewidth=2.5, markersize=8, color='orange')
ax.set_xlabel('Lambda (λ)', fontweight='bold')
ax.set_ylabel('||S|| (Frobenius norm)', fontweight='bold')
ax.set_xscale('log')
ax.set_title('Sparse Component Magnitude vs Lambda', fontweight='bold')
ax.grid(True, alpha=0.3)

# Trade-off curve
ax = axes[1, 1]
ax.plot(sparsity_by_lambda, recon_error_by_lambda, 'o-', linewidth=2.5, markersize=8, color='purple')
for lam, sparsity, error in zip(lamda_set, sparsity_by_lambda, recon_error_by_lambda):
    ax.annotate(f'λ={lam:.2f}', (sparsity, error), fontsize=9, alpha=0.7)
ax.set_xlabel('Sparsity', fontweight='bold')
ax.set_ylabel('Reconstruction MSE', fontweight='bold')
ax.set_title('Sparsity vs Reconstruction Trade-off', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rpca_lambda_statistics.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ RPCA Visualizations saved:")
print("   - rpca_decomposition_comparison.png")
print("   - rpca_anomaly_scores_heatmap.png")
print("   - rpca_timeseries_decomposition.png")
print("   - rpca_lambda_statistics.png")
import numpy as np
import torch
import torch.nn.functional as F

from models.rpca import ConvAutoencoder, soft_threshold
from dataloader import NpzDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load and normalize data
datafile_path = "stock_sequences.npz"
data = np.load(datafile_path)
X_np = data['X']  # Shape: [num_sequences, seq_len, features]

feature_mean = X_np.mean(axis=(0, 1))
feature_std = X_np.std(axis=(0, 1)) + 1e-8
X_normalized = (X_np - feature_mean) / feature_std

batch_size = min(32, len(X_normalized))
X_batch = torch.tensor(X_normalized[:batch_size], dtype=torch.float32).to(device)

print(f"Input shape: {X_batch.shape}")
print(f"Feature mean shape: {feature_mean.shape}")
print(f"Feature std shape: {feature_std.shape}")

# Initialize model
model = ConvAutoencoder(latent_dim=3, input_shape=(1, 25, 8)).to(device)
model.train()
print("Model initialized in train mode")

# Try simple autoencoder training
print("\nTesting autoencoder training...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

X_input_4d = X_batch.unsqueeze(1)
X_target_4d = X_batch.unsqueeze(1)

for epoch in range(20):
    optimizer.zero_grad()
    output = model(X_input_4d)
    
    # Pad target if needed
    if output.shape != X_target_4d.shape:
        pad_h = output.shape[2] - X_target_4d.shape[2]
        pad_w = output.shape[3] - X_target_4d.shape[3]
        X_target_padded = torch.nn.functional.pad(X_target_4d, (0, pad_w, 0, pad_h))
    else:
        X_target_padded = X_target_4d
    
    loss = criterion(output, X_target_padded)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/20 | Loss: {loss.item():.6f}")

# Test reconstruction
model.eval()
with torch.no_grad():
    reconstruction = model(X_input_4d)
    reconstruction = reconstruction[:, :, :X_target_4d.shape[2], :X_target_4d.shape[3]].squeeze(1)

mse = F.mse_loss(reconstruction, X_batch).item()
print(f"\nReconstruction MSE: {mse:.6f}")

# Denormalize and compare
sample_idx = 0
X_orig = X_batch[sample_idx].cpu().numpy()
X_recon = reconstruction[sample_idx].cpu().numpy()

X_orig_denorm = X_orig * feature_std + feature_mean
X_recon_denorm = X_recon * feature_std + feature_mean

print(f"\nSample 0 comparison (Feature: Close, idx=3):")
print(f"Original normalized:     {X_orig[0, 3]:.6f} -> denorm: {X_orig_denorm[0, 3]:.6f}")
print(f"Reconstructed normalized: {X_recon[0, 3]:.6f} -> denorm: {X_recon_denorm[0, 3]:.6f}")
print(f"Mean error (denormalized): {np.mean(np.abs(X_orig_denorm[:, 3] - X_recon_denorm[:, 3])):.6f}")

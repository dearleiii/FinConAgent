import numpy as np

# Load data
data = np.load("stock_sequences.npz")
X_np = data['X']

feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'VWAP']

print("Feature Statistics (Raw Data):")
print("="*80)
for i, feat in enumerate(feature_names):
    mean = X_np[:, :, i].mean()
    std = X_np[:, :, i].std()
    min_val = X_np[:, :, i].min()
    max_val = X_np[:, :, i].max()
    print(f"{feat:10} | Mean: {mean:12.2f} | Std: {std:12.2f} | Range: [{min_val:12.2f}, {max_val:12.2f}]")

print("\n" + "="*80)
print("After Global Normalization (across all features):")
print("="*80)

feature_mean = X_np.mean(axis=(0, 1))
feature_std = X_np.std(axis=(0, 1)) + 1e-8

for i, feat in enumerate(feature_names):
    X_norm = (X_np[:, :, i] - feature_mean[i]) / feature_std[i]
    print(f"{feat:10} | Mean: {X_norm.mean():7.4f} | Std: {X_norm.std():7.4f} | Range: [{X_norm.min():7.4f}, {X_norm.max():7.4f}]")

print("\n" + "="*80)
print("Key Insight:")
print("="*80)
print("Volume has much larger raw values and standard deviation.")
print("This causes it to have different normalized ranges than price features.")
print("In heatmap visualization, this leads to different color scales for Volume.")

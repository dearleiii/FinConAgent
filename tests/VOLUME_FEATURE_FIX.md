# Volume Feature Visualization Issue - Analysis & Fix

## Problem
In the `rpca_decomposition_comparison.png` heatmaps, the Volume feature (row 5) appeared visually very different from other features, with much larger value ranges that dominated the color scale.

## Root Cause Analysis

### Feature Scale Comparison
```
Raw Data Statistics:
- Price features (Open/High/Low/Close/MA10/MA20/VWAP): Mean ~259, Std ~68.5
- Volume: Mean ~128,697, Std ~111,396 (1000x larger!)

After Normalization:
- Price features: Range [-3.79, 0.53]
- Volume: Range [-1.15, 12.63] (much larger!)
```

**The Problem**: Although z-score normalization standardizes means/stds to 0/1, Volume still has:
1. Different min/max range due to its different distribution
2. Larger outlier values (up to ±12.6 vs ±0.5)
3. This dominates the heatmap color scale, making price features appear uniform and volume appears extreme

## Solution Implemented

Changed the heatmap visualization from **global normalization** to **per-feature normalization**:

### Before (Global normalization - problematic):
```python
im = ax.imshow(X_sample.T, aspect='auto', cmap='RdYlBu_r')
# Color scale: [-1.15 to 12.63], Volume dominates
```

### After (Per-feature min-max normalization - fixed):
```python
X_normalized_viz = X_sample.copy()
for feat_idx in range(X_sample.shape[1]):
    feat_min = X_sample[:, feat_idx].min()
    feat_max = X_sample[:, feat_idx].max()
    if feat_max > feat_min:
        X_normalized_viz[:, feat_idx] = (X_sample[:, feat_idx] - feat_min) / (feat_max - feat_min)

im = ax.imshow(X_normalized_viz.T, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
# Color scale: [0, 1] for ALL features - fair comparison
```

## Applied Changes

Updated 3 heatmap panels in `rpca_decomposition_comparison()`:

1. **Original Signal** - Per-feature normalized to [0, 1]
2. **Reconstructed L** - Per-feature normalized to [0, 1]
3. **Sparse Component |S|** - Per-feature normalized relative to each feature's max anomaly

All 3 now have consistent visualization where:
- ✅ Volume is no longer visually dominant
- ✅ All features use [0, 1] scale for fair comparison
- ✅ Temporal patterns are visible for all features equally
- ✅ Anomalies in each feature are highlighted relative to that feature's baseline

## Impact

The updated `rpca_decomposition_comparison.png` will now show:
- All features with similar visual prominence
- Volume's actual sparse anomalies clearly visible (not hidden by scale)
- Better comparison across lambda values since all features are on the same [0,1] scale
- More interpretable results for anomaly detection in all features

## Note on Raw Values

The actual underlying data is still the raw normalized values (z-scores). The visualization is just adjusted for interpretability. For precise numerical analysis, use the raw `rpca_results[lam]['L']` and `rpca_results[lam]['S']` values directly.

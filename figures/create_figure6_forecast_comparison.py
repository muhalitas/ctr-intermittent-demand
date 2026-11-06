"""
Create Figure 6 - Forecast comparison on representative SKUs
Panels by sparsity level showing true demand vs forecasts
Highlights false positives & missed events
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

# Load data
print("Loading prediction files...")
croston_df = pd.read_csv('./Son2.py_sonuclar/croston_predictions.csv')
sba_df = pd.read_csv('./Son2.py_sonuclar/sba_predictions.csv')
lstm_df = pd.read_csv('./Son2.py_sonuclar/lstm_predictions.csv')
ctr_df = pd.read_csv('./Son2.py_sonuclar/ctr_predictions.csv')
combined_df = pd.read_csv('./Son2.py_sonuclar/combined_results.csv')

# Get unique products with their sparsity levels
product_sparsity = combined_df[['Product_ID', 'Sparsity_Level']].drop_duplicates()
product_sparsity = product_sparsity.set_index('Product_ID')['Sparsity_Level'].to_dict()

# Define sparsity bins for selection (combining multiple bins)
sparsity_bins = {
    '20-40%': ['[0.2, 0.3)', '[0.3, 0.4)'],
    '40-60%': ['[0.4, 0.5)', '[0.5, 0.6)'],
    '60-80%': ['[0.6, 0.7)', '[0.7, 0.8)'],
    '80-100%': ['[0.8, 1.0)']
}

# Function to select representative SKU from a sparsity bin
def select_representative_sku(sparsity_levels, min_test_periods=15):
    """Select a representative SKU with sufficient test periods and some non-zero demand"""
    # Get all candidates from the specified sparsity levels
    candidates = [pid for pid, sp in product_sparsity.items() 
                  if sp in sparsity_levels]
    
    if not candidates:
        return None
    
    # Check each candidate
    for pid in candidates:
        # Get test data for this product
        test_data = croston_df[croston_df['Product_ID'] == pid]
        if len(test_data) < min_test_periods:
            continue
        
        # Check if there's at least some non-zero demand (relaxed for high sparsity)
        non_zero_count = (test_data['y_test'] > 0).sum()
        if non_zero_count < 1:  # Need at least 1 non-zero period
            continue
        
        # Check if there's variation in predictions (not all constant)
        ctr_preds = ctr_df[ctr_df['Product_ID'] == pid]['CTR_Predictions'].values
        if len(ctr_preds) > 0 and np.std(ctr_preds) > 0.01:
            return pid
    
    # If no ideal candidate, return first with sufficient periods
    for pid in candidates:
        test_data = croston_df[croston_df['Product_ID'] == pid]
        if len(test_data) >= min_test_periods:
            return pid
    
    return None

# Select representative SKUs
selected_skus = {}
for bin_name, sparsity_levels in sparsity_bins.items():
    sku = select_representative_sku(sparsity_levels)
    if sku:
        selected_skus[bin_name] = sku
        actual_sparsity = product_sparsity.get(sku, 'Unknown')
        print(f"Selected SKU {sku} for sparsity bin {bin_name} (actual: {actual_sparsity})")

# Create figure with subplots
fig, axes = plt.subplots(len(selected_skus), 4, figsize=(20, 5*len(selected_skus)))
if len(selected_skus) == 1:
    axes = axes.reshape(1, -1)

models = {
    'Croston': {'df': croston_df, 'col': 'Croston_Predictions', 'color': '#FF6B6B'},
    'SBA': {'df': sba_df, 'col': 'SBA_Predictions', 'color': '#4ECDC4'},
    'LSTM': {'df': lstm_df, 'col': 'LSTM_Predictions', 'color': '#95E1D3'},
    'CTR': {'df': ctr_df, 'col': 'CTR_Predictions', 'color': '#F38181'}
}

# Plot each sparsity level
for row_idx, (bin_name, sku_id) in enumerate(selected_skus.items()):
    # Get data for this SKU
    croston_data = croston_df[croston_df['Product_ID'] == sku_id].copy()
    sba_data = sba_df[sba_df['Product_ID'] == sku_id].copy()
    lstm_data = lstm_df[lstm_df['Product_ID'] == sku_id].copy()
    ctr_data = ctr_df[ctr_df['Product_ID'] == sku_id].copy()
    
    # Ensure all have same length and order
    min_len = min(len(croston_data), len(sba_data), len(lstm_data), len(ctr_data))
    croston_data = croston_data.iloc[:min_len].reset_index(drop=True)
    sba_data = sba_data.iloc[:min_len].reset_index(drop=True)
    lstm_data = lstm_data.iloc[:min_len].reset_index(drop=True)
    ctr_data = ctr_data.iloc[:min_len].reset_index(drop=True)
    
    y_true = croston_data['y_test'].values
    time_points = np.arange(len(y_true))
    
    # Plot each model
    for col_idx, (model_name, model_info) in enumerate(models.items()):
        ax = axes[row_idx, col_idx]
        
        # Get predictions
        if model_name == 'Croston':
            y_pred = croston_data[model_info['col']].values
        elif model_name == 'SBA':
            y_pred = sba_data[model_info['col']].values
        elif model_name == 'LSTM':
            y_pred = lstm_data[model_info['col']].values
        else:  # CTR
            y_pred = ctr_data[model_info['col']].values
        
        # Plot true demand
        ax.plot(time_points, y_true, 'ko-', markersize=4, linewidth=1.5, 
                label='True Demand', alpha=0.7, zorder=3)
        
        # Plot predictions
        ax.plot(time_points, y_pred, '-', color=model_info['color'], 
                linewidth=2, label=f'{model_name} Forecast', alpha=0.8, zorder=2)
        
        # Highlight false positives (predicted > 0 when actual = 0)
        # Use a small threshold relative to the scale of predictions
        threshold = max(0.01, np.percentile(y_pred[y_pred > 0], 5) if (y_pred > 0).any() else 0.01)
        false_pos_mask = (y_true == 0) & (y_pred > threshold)
        
        # Highlight missed events (predicted = 0 or very low when actual > 0)
        missed_mask = (y_true > 0) & (y_pred < threshold)
        
        # Build legend items dynamically
        legend_items = ['True Demand', f'{model_name} Forecast']
        
        if false_pos_mask.any():
            ax.scatter(time_points[false_pos_mask], y_pred[false_pos_mask], 
                      color='red', marker='x', s=120, linewidths=3, 
                      label='False Positive', zorder=4)
            legend_items.append('False Positive')
        
        if missed_mask.any():
            ax.scatter(time_points[missed_mask], y_true[missed_mask], 
                      color='orange', marker='^', s=120, linewidths=2, 
                      label='Missed Event', zorder=4, edgecolors='darkorange')
            legend_items.append('Missed Event')
        
        # Formatting
        ax.set_xlabel('Time Period', fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(f'Demand\n({bin_name} sparsity)', fontsize=11, fontweight='bold')
        # Add SKU ID to title
        ax.set_title(f'{model_name}\nSKU: {sku_id}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=1)
        
        # Set y-axis to start from 0 or slightly below
        y_max = max(y_true.max(), y_pred.max()) * 1.1
        ax.set_ylim(-0.05 * y_max, y_max)
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# Add overall title
fig.suptitle('Figure 6 — Forecast Comparison on Representative SKUs', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save
output_path = './Son2.py_sonuclar/figure6_forecast_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to: {output_path}")

output_path_pdf = './Son2.py_sonuclar/figure6_forecast_comparison.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_path_pdf}")

plt.close()

print("\nFigure 6 generation complete!")
print(f"Selected SKUs: {selected_skus}")

"""
Create Figure 8 - Sparsity vs Optimal Threshold
Scatter plot with linear fit showing relationship between sparsity and optimal alpha threshold
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Load combined results to get Product_ID, Best_Alpha_CTR, and Sparsity_Level
print("Loading data...")
combined_df = pd.read_csv('./Son2.py_sonuclar/combined_results.csv')

# Get CTR model rows with valid alpha and sparsity
ctr_data = combined_df[combined_df['Model'] == 'CTR'].copy()
ctr_data = ctr_data[ctr_data['Best_Alpha_CTR'].notna()]
ctr_data = ctr_data[ctr_data['Sparsity_Level'].notna()]

# Calculate actual sparsity values from original data
print("Calculating sparsity values...")
all_intermittent_products = pd.read_csv('./Veri/filtered_data.csv')

def calculate_sparsity(df, product_col='URUNID', demand_col='Satis Adet'):
    sparsity_dict = {}
    for pid in df[product_col].unique():
        product_data = df[df[product_col] == pid][demand_col]
        total_periods = len(product_data)
        zero_periods = (product_data == 0).sum()
        sparsity = zero_periods / total_periods if total_periods > 0 else 0
        sparsity_dict[pid] = sparsity
    return sparsity_dict

sparsity_dict = calculate_sparsity(all_intermittent_products)

# Add actual sparsity to CTR data
ctr_data['Sparsity'] = ctr_data['Product_ID'].map(sparsity_dict)
ctr_data = ctr_data[ctr_data['Sparsity'].notna()]

# Prepare data for plotting
x = ctr_data['Sparsity'].values
y = ctr_data['Best_Alpha_CTR'].values

print(f"Number of data points: {len(x)}")
print(f"Sparsity range: {x.min():.3f} - {x.max():.3f}")
print(f"Threshold range: {y.min():.3f} - {y.max():.3f}")

# Fit linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter plot
scatter = ax.scatter(x, y, alpha=0.5, s=30, color='#F38181', edgecolors='darkred', linewidths=0.5, zorder=3, label='Product (SKU)')

# Linear fit line
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'Linear Fit (R² = {r_value**2:.3f})', zorder=2)

# Formatting
ax.set_xlabel('Sparsity (% Zero Periods)', fontsize=12, fontweight='bold')
ax.set_ylabel('Optimal Threshold (α)', fontsize=12, fontweight='bold')
ax.set_title('Figure 8 — Sparsity vs Optimal Threshold', 
             fontsize=14, fontweight='bold', pad=15)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
ax.set_axisbelow(True)

# Add equation text
equation_text = f'y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}'
ax.text(0.05, 0.95, equation_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        zorder=4)

# Add note explaining what each point represents
note_text = 'Each point represents a product (SKU)\nwith its sparsity and optimal α threshold'
ax.text(0.95, 0.05, note_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        zorder=4)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Set axis limits with some padding
ax.set_xlim(max(0, x.min() - 0.02), min(1, x.max() + 0.02))
ax.set_ylim(max(0, y.min() - 0.02), min(1, y.max() + 0.02))

plt.tight_layout()

# Save
output_path = './Son2.py_sonuclar/figure8_sparsity_threshold.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to: {output_path}")

output_path_pdf = './Son2.py_sonuclar/figure8_sparsity_threshold.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_path_pdf}")

plt.close()

# Print statistics
print(f"\n=== Linear Regression Statistics ===")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R²: {r_value**2:.4f}")
print(f"p-value: {p_value:.2e}")
print(f"Standard Error: {std_err:.4f}")

print("\nFigure 8 generation complete!")

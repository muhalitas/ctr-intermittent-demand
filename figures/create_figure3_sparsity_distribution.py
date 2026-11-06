"""
Create Figure 3 - Distribution of Sparsity (% zero periods)
Shows histogram/KDE of sparsity distribution with bin boundaries overlaid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
print("Loading data...")
all_intermittent_products = pd.read_csv('./Veri/filtered_data.csv')

# Calculate sparsity for each product
def calculate_sparsity(df, product_col='URUNID', demand_col='Satis Adet'):
    sparsity_dict = {}
    for pid in df[product_col].unique():
        product_data = df[df[product_col] == pid][demand_col]
        total_periods = len(product_data)
        zero_periods = (product_data == 0).sum()
        sparsity = zero_periods / total_periods if total_periods > 0 else 0
        sparsity_dict[pid] = sparsity
    return sparsity_dict

# Calculate sparsity
sparsity_dict = calculate_sparsity(all_intermittent_products)
sparsity_df = pd.DataFrame.from_dict(sparsity_dict, orient='index', columns=['Sparsity'])
sparsity_df.index.name = 'URUNID'

# Define sparsity bins (same as used in analysis)
sparsity_bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

print(f"Total products: {len(sparsity_df)}")
print(f"Sparsity range: {sparsity_df['Sparsity'].min():.3f} - {sparsity_df['Sparsity'].max():.3f}")
print(f"Mean sparsity: {sparsity_df['Sparsity'].mean():.3f}")
print(f"Median sparsity: {sparsity_df['Sparsity'].median():.3f}")

# Create figure with both histogram and KDE
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Top subplot: Histogram
ax1 = axes[0]
n, bins, patches = ax1.hist(sparsity_df['Sparsity'], bins=50, alpha=0.7, color='steelblue', 
                             edgecolor='black', linewidth=0.5)

# Overlay bin boundaries
for bin_edge in sparsity_bins:
    ax1.axvline(x=bin_edge, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label='Bin boundaries' if bin_edge == sparsity_bins[0] else '')

ax1.set_xlabel('Sparsity (% zero periods)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
ax1.set_title('Figure 3a — Histogram of Sparsity Distribution', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)

# Add statistics text
stats_text = f'n = {len(sparsity_df)}\nMean = {sparsity_df["Sparsity"].mean():.3f}\nMedian = {sparsity_df["Sparsity"].median():.3f}'
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Bottom subplot: KDE (Kernel Density Estimation)
ax2 = axes[1]

# Create KDE
kde = stats.gaussian_kde(sparsity_df['Sparsity'])
x_kde = np.linspace(0, 1, 1000)
y_kde = kde(x_kde)

# Plot KDE
ax2.plot(x_kde, y_kde, linewidth=2.5, color='darkblue', label='KDE')

# Fill under curve
ax2.fill_between(x_kde, y_kde, alpha=0.3, color='steelblue')

# Overlay bin boundaries
for bin_edge in sparsity_bins:
    ax2.axvline(x=bin_edge, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='Bin boundaries' if bin_edge == sparsity_bins[0] else '')

ax2.set_xlabel('Sparsity (% zero periods)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
ax2.set_title('Figure 3b — Kernel Density Estimation (KDE) of Sparsity Distribution', 
              fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlim(0, 1)

# Add bin labels
bin_labels = ['[0.0, 0.2)', '[0.2, 0.3)', '[0.3, 0.4)', '[0.4, 0.5)', 
              '[0.5, 0.6)', '[0.6, 0.7)', '[0.7, 0.8)', '[0.8, 1.0)']
for i, (bin_start, bin_end) in enumerate(zip(sparsity_bins[:-1], sparsity_bins[1:])):
    bin_center = (bin_start + bin_end) / 2
    count = len(sparsity_df[(sparsity_df['Sparsity'] >= bin_start) & 
                           (sparsity_df['Sparsity'] < bin_end)])
    if i < len(bin_labels):
        # Add text annotation for bin counts
        ax1.text(bin_center, ax1.get_ylim()[1] * 0.9, f'{count}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()

# Save figure
output_path = './Son2.py_sonuclar/figure3_sparsity_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Also save as PDF for publication
output_path_pdf = './Son2.py_sonuclar/figure3_sparsity_distribution.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Figure saved to: {output_path_pdf}")

# Print bin statistics
print("\n" + "=" * 80)
print("Sparsity Bin Statistics:")
print("=" * 80)
for i, (bin_start, bin_end) in enumerate(zip(sparsity_bins[:-1], sparsity_bins[1:])):
    mask = (sparsity_df['Sparsity'] >= bin_start) & (sparsity_df['Sparsity'] < bin_end)
    count = mask.sum()
    percentage = (count / len(sparsity_df)) * 100
    print(f"{bin_labels[i]:<15} {bin_start:>4.1f}-{bin_end:<4.1f}: {count:>4d} products ({percentage:>5.2f}%)")

plt.show()

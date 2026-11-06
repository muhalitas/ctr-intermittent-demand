"""
Create Figure 7 - Relative improvement vs sparsity
Line chart showing % MAE improvement by sparsity class
Separate lines for CTR, LSTM, SBA
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load sparsity metrics
print("Loading sparsity metrics...")
df = pd.read_csv('./Son2.py_sonuclar/sparsity_metrics.csv')

# Filter for models we need: CTR, LSTM, SBA, and SES (baseline)
models_to_plot = ['CTR', 'LSTM Model', 'SBA Method']
baseline_model = 'Simple Exponential Smoothing'

# Filter data
plot_data = df[df['Model'].isin(models_to_plot + [baseline_model])].copy()

# Remove rows with missing MAE
plot_data = plot_data[plot_data['MAE'].notna()]

# Get baseline MAE by sparsity level
baseline_mae = plot_data[plot_data['Model'] == baseline_model].set_index('Sparsity_Level')['MAE']

# Calculate % improvement for each model
improvements = {}
for model in models_to_plot:
    model_data = plot_data[plot_data['Model'] == model].set_index('Sparsity_Level')
    improvements[model] = {}
    
    for sparsity_level in model_data.index:
        if sparsity_level in baseline_mae.index:
            model_mae = model_data.loc[sparsity_level, 'MAE']
            baseline_mae_val = baseline_mae.loc[sparsity_level]
            
            if baseline_mae_val > 0:
                # % improvement = (baseline - model) / baseline * 100
                improvement = ((baseline_mae_val - model_mae) / baseline_mae_val) * 100
                improvements[model][sparsity_level] = improvement
            else:
                improvements[model][sparsity_level] = np.nan
        else:
            improvements[model][sparsity_level] = np.nan

# Prepare data for plotting
sparsity_levels = sorted([sl for sl in baseline_mae.index if pd.notna(sl)])
# Filter out empty sparsity levels
sparsity_levels = [sl for sl in sparsity_levels if sl and str(sl) != 'nan']

# Create cleaner labels for x-axis
sparsity_labels = []
for sl in sparsity_levels:
    if isinstance(sl, str):
        # Convert [0.2, 0.3) to "20-30%"
        if '[' in sl and ',' in sl:
            parts = sl.replace('[', '').replace(')', '').split(',')
            if len(parts) == 2:
                lower = float(parts[0].strip()) * 100
                upper = float(parts[1].strip()) * 100
                sparsity_labels.append(f"{int(lower)}-{int(upper)}%")
            else:
                sparsity_labels.append(sl)
        else:
            sparsity_labels.append(sl)
    else:
        sparsity_labels.append(str(sl))

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Define colors and styles
model_styles = {
    'CTR': {'color': '#F38181', 'marker': 'o', 'linewidth': 2.5, 'markersize': 8, 'label': 'CTR'},
    'LSTM Model': {'color': '#95E1D3', 'marker': 's', 'linewidth': 2.5, 'markersize': 8, 'label': 'LSTM'},
    'SBA Method': {'color': '#4ECDC4', 'marker': '^', 'linewidth': 2.5, 'markersize': 8, 'label': 'SBA'}
}

# Plot each model
x_positions = np.arange(len(sparsity_levels))
for model in models_to_plot:
    y_values = []
    for sl in sparsity_levels:
        if sl in improvements[model]:
            y_values.append(improvements[model][sl])
        else:
            y_values.append(np.nan)
    
    style = model_styles[model]
    ax.plot(x_positions, y_values, 
            color=style['color'], 
            marker=style['marker'],
            linewidth=style['linewidth'],
            markersize=style['markersize'],
            label=style['label'],
            alpha=0.9,
            zorder=3)

# Add zero line
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

# Formatting
ax.set_xlabel('Sparsity Class (% Zero Periods)', fontsize=12, fontweight='bold')
ax.set_ylabel('% MAE Improvement\n(relative to SES)', fontsize=12, fontweight='bold')
ax.set_title('Figure 7 — Relative Improvement vs Sparsity', 
             fontsize=14, fontweight='bold', pad=15)

# Set x-axis
ax.set_xticks(x_positions)
ax.set_xticklabels(sparsity_labels, rotation=45, ha='right', fontsize=10)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
ax.set_axisbelow(True)

# Set y-axis limits to show improvements clearly
y_min = min([min([v for v in improvements[m].values() if not np.isnan(v)]) 
             for m in models_to_plot] + [0]) - 5
y_max = max([max([v for v in improvements[m].values() if not np.isnan(v)]) 
             for m in models_to_plot] + [0]) + 5
ax.set_ylim(y_min, y_max)

# Legend
ax.legend(loc='best', fontsize=11, framealpha=0.9, ncol=3)

# Adjust layout
plt.tight_layout()

# Save
output_path = './Son2.py_sonuclar/figure7_relative_improvement.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to: {output_path}")

output_path_pdf = './Son2.py_sonuclar/figure7_relative_improvement.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_path_pdf}")

plt.close()

# Print summary statistics
print("\n=== Improvement Summary ===")
for model in models_to_plot:
    print(f"\n{model}:")
    for sl, label in zip(sparsity_levels, sparsity_labels):
        if sl in improvements[model] and not np.isnan(improvements[model][sl]):
            print(f"  {label}: {improvements[model][sl]:.2f}%")

print("\nFigure 7 generation complete!")

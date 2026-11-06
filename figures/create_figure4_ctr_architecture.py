"""
Create Figure 4 - CTR Model Architecture Diagram
Shows the two-stage classification-then-regression architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
input_color = '#E8F4F8'
lstm_color = '#FFE5B4'
class_color = '#FFB6C1'
reg_color = '#90EE90'
decision_color = '#DDA0DD'
output_color = '#FFD700'

# Define positions
y_top = 11
y_input = 9.5
y_lstm = 7
y_branch = 5
y_decision = 2.5
y_output = 0.5

# 1. INPUT LAYER
input_box = FancyBboxPatch((0.5, y_input-0.8), 9, 1.6, 
                           boxstyle="round,pad=0.1", 
                           facecolor=input_color, 
                           edgecolor='black', linewidth=2)
ax.add_patch(input_box)
ax.text(5, y_input, 'Input Features', ha='center', va='center', 
        fontsize=14, fontweight='bold')

# Input features list
features = [
    'URUNID', 'Day', 'Month', 'Year', 
    'DayOfWeek', 'IsWeekend',
    'Lag_1', 'Lag_2', 'Lag_3',
    'Rolling_Mean_3', 'Rolling_Mean_7'
]
feature_text = ' | '.join(features)
ax.text(5, y_input-0.4, feature_text, ha='center', va='center', 
        fontsize=9, style='italic')

# Arrow from input to LSTM
arrow1 = FancyArrowPatch((5, y_input-0.8), (5, y_lstm+0.5),
                        arrowstyle='->', lw=2.5, color='black')
ax.add_patch(arrow1)

# 2. SHARED LSTM ENCODER
lstm_box = FancyBboxPatch((3, y_lstm-0.6), 4, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor=lstm_color,
                          edgecolor='black', linewidth=2)
ax.add_patch(lstm_box)
ax.text(5, y_lstm, 'Shared LSTM Encoder\n(32 units)', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Dense layers
dense1_box = FancyBboxPatch((3.2, y_lstm-0.4), 3.6, 0.3,
                           boxstyle="round,pad=0.05",
                           facecolor=lstm_color,
                           edgecolor='gray', linewidth=1)
ax.add_patch(dense1_box)
ax.text(5, y_lstm-0.25, 'Dense(64) + Dropout(0.1)', ha='center', va='center',
        fontsize=9)

dense2_box = FancyBboxPatch((3.2, y_lstm-0.7), 3.6, 0.3,
                           boxstyle="round,pad=0.05",
                           facecolor=lstm_color,
                           edgecolor='gray', linewidth=1)
ax.add_patch(dense2_box)
ax.text(5, y_lstm-0.55, 'Dense(32)', ha='center', va='center',
        fontsize=9)

# Arrow from LSTM to branches
arrow2 = FancyArrowPatch((5, y_lstm-0.6), (5, y_branch+0.5),
                        arrowstyle='->', lw=2.5, color='black')
ax.add_patch(arrow2)

# 3. BRANCH POINT
branch_circle = Circle((5, y_branch), 0.3, facecolor='white', 
                       edgecolor='black', linewidth=2)
ax.add_patch(branch_circle)
ax.text(5, y_branch, '•', ha='center', va='center', fontsize=20)

# 4. BRANCH 1: CLASSIFICATION HEAD
class_box = FancyBboxPatch((1, y_branch-0.8), 2.5, 1.6,
                           boxstyle="round,pad=0.1",
                           facecolor=class_color,
                           edgecolor='black', linewidth=2)
ax.add_patch(class_box)
ax.text(2.25, y_branch, 'Classification\nHead', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Classification details
class_dense = FancyBboxPatch((1.2, y_branch-0.6), 2.1, 0.3,
                            boxstyle="round,pad=0.05",
                            facecolor=class_color,
                            edgecolor='gray', linewidth=1)
ax.add_patch(class_dense)
ax.text(2.25, y_branch-0.45, 'Dense(1, sigmoid)', ha='center', va='center',
        fontsize=9)

ax.text(2.25, y_branch-0.8, 'P(demand > 0)', ha='center', va='center',
        fontsize=9, style='italic')

# Arrow to classification
arrow_class = FancyArrowPatch((3.5, y_branch), (2.5, y_branch),
                             arrowstyle='->', lw=2, color='black')
ax.add_patch(arrow_class)

# 5. BRANCH 2: REGRESSION HEAD
reg_box = FancyBboxPatch((6.5, y_branch-0.8), 2.5, 1.6,
                         boxstyle="round,pad=0.1",
                         facecolor=reg_color,
                         edgecolor='black', linewidth=2)
ax.add_patch(reg_box)
ax.text(7.75, y_branch, 'Regression\nHead', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Regression details
reg_dense = FancyBboxPatch((6.7, y_branch-0.6), 2.1, 0.3,
                          boxstyle="round,pad=0.05",
                          facecolor=reg_color,
                          edgecolor='gray', linewidth=1)
ax.add_patch(reg_dense)
ax.text(7.75, y_branch-0.45, 'Dense(1, linear)', ha='center', va='center',
        fontsize=9)

ax.text(7.75, y_branch-0.8, 'Demand value', ha='center', va='center',
        fontsize=9, style='italic')

# Arrow to regression
arrow_reg = FancyArrowPatch((6.5, y_branch), (7.5, y_branch),
                           arrowstyle='->', lw=2, color='black')
ax.add_patch(arrow_reg)

# 6. DECISION LOGIC
decision_box = FancyBboxPatch((2.5, y_decision-0.6), 5, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=decision_color,
                              edgecolor='black', linewidth=2)
ax.add_patch(decision_box)

# Decision logic text
decision_text = 'Decision Logic:\nif P(demand > 0) > α: output = regression_value\nelse: output = 0'
ax.text(5, y_decision, decision_text, ha='center', va='center',
        fontsize=10, fontweight='bold')

# Arrows from branches to decision
arrow_class_to_dec = FancyArrowPatch((2.25, y_branch-0.8), (3.5, y_decision+0.6),
                                    arrowstyle='->', lw=2, color='black',
                                    connectionstyle="arc3,rad=0.2")
ax.add_patch(arrow_class_to_dec)

arrow_reg_to_dec = FancyArrowPatch((7.75, y_branch-0.8), (6.5, y_decision+0.6),
                                   arrowstyle='->', lw=2, color='black',
                                   connectionstyle="arc3,rad=-0.2")
ax.add_patch(arrow_reg_to_dec)

# 7. OUTPUT
output_box = FancyBboxPatch((3.5, y_output-0.4), 3, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=output_color,
                           edgecolor='black', linewidth=2)
ax.add_patch(output_box)
ax.text(5, y_output, 'Final Prediction\n(Demand Forecast)', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Arrow from decision to output
arrow_output = FancyArrowPatch((5, y_decision-0.6), (5, y_output+0.4),
                              arrowstyle='->', lw=2.5, color='black')
ax.add_patch(arrow_output)

# Add title
ax.text(5, y_top, 'Figure 4 — CTR Model Architecture', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Add sequence indicator
ax.text(0.2, y_input, '1', ha='center', va='center', 
        fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
ax.text(0.2, y_lstm, '2', ha='center', va='center', 
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
ax.text(0.2, y_branch, '3', ha='center', va='center', 
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
ax.text(0.2, y_decision, '4', ha='center', va='center', 
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
ax.text(0.2, y_output, '5', ha='center', va='center', 
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

# Add note about alpha threshold
note_text = 'Note: α (threshold) is optimized per product on validation set'
ax.text(5, 0.1, note_text, ha='center', va='bottom', 
        fontsize=9, style='italic', color='gray')

plt.tight_layout()

# Save figure
output_path = './Son2.py_sonuclar/figure4_ctr_architecture.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_path}")

# Also save as PDF
output_path_pdf = './Son2.py_sonuclar/figure4_ctr_architecture.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_path_pdf}")

plt.show()

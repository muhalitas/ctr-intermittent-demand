"""
Create Figure 5 - Workflow of Classify-Then-Regress Pipeline (Improved Version)
Flowchart showing the two-stage decision process with better layout
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')

# Professional color scheme
input_color = '#BBDEFB'
process_color = '#FFE0B2'
decision_color = '#E1BEE7'
regression_color = '#A5D6A7'
output_color = '#FFF59D'
suppress_color = '#EF9A9A'

# Y positions (better spacing)
y_input = 9.5
y_classify = 7.5
y_decision = 5.5
y_branches = 3
y_output = 1

# X positions
x_center = 7
x_yes = 11
x_no = 3

# 1. INPUT SEQUENCE
input_box = FancyBboxPatch((x_center-2, y_input-0.5), 4, 1,
                           boxstyle="round,pad=0.15",
                           facecolor=input_color,
                           edgecolor='#1976D2', linewidth=2.5)
ax.add_patch(input_box)
ax.text(x_center, y_input, 'Input Sequence\n(Time series features)', 
        ha='center', va='center', fontsize=12, fontweight='bold')

# Arrow to classification
arrow1 = FancyArrowPatch((x_center, y_input-0.5), (x_center, y_classify+0.5),
                        arrowstyle='->', lw=3, color='#424242',
                        mutation_scale=20)
ax.add_patch(arrow1)

# 2. CLASSIFICATION STAGE
classify_box = FancyBboxPatch((x_center-2.5, y_classify-0.5), 5, 1,
                              boxstyle="round,pad=0.15",
                              facecolor=process_color,
                              edgecolor='#F57C00', linewidth=2.5)
ax.add_patch(classify_box)
ax.text(x_center, y_classify, 'Classification Stage\nCompute P(demand > 0)', 
        ha='center', va='center', fontsize=12, fontweight='bold')

# Arrow to decision
arrow2 = FancyArrowPatch((x_center, y_classify-0.5), (x_center, y_decision+0.6),
                        arrowstyle='->', lw=3, color='#424242',
                        mutation_scale=20)
ax.add_patch(arrow2)

# 3. DECISION DIAMOND (larger, clearer)
diamond_size = 1.5
diamond_points = np.array([
    [x_center, y_decision+diamond_size],      # top
    [x_center+diamond_size, y_decision],     # right
    [x_center, y_decision-diamond_size],     # bottom
    [x_center-diamond_size, y_decision]       # left
])
decision_diamond = Polygon(diamond_points, 
                          facecolor=decision_color,
                          edgecolor='#7B1FA2', linewidth=3)
ax.add_patch(decision_diamond)
ax.text(x_center, y_decision, 'P(demand > 0)\n> α?', 
        ha='center', va='center', fontsize=12, fontweight='bold')

# 4a. YES PATH → REGRESSION (right side)
yes_box = FancyBboxPatch((x_yes-1.5, y_branches-0.6), 3, 1.2,
                         boxstyle="round,pad=0.15",
                         facecolor=regression_color,
                         edgecolor='#2E7D32', linewidth=2.5)
ax.add_patch(yes_box)
ax.text(x_yes, y_branches+0.1, 'Non-zero Path', 
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(x_yes, y_branches-0.2, '→ Regression', 
        ha='center', va='center', fontsize=11, fontweight='bold', color='#1B5E20')

# Regression detail
reg_detail = FancyBboxPatch((x_yes-1.4, y_branches-0.45), 2.8, 0.3,
                           boxstyle="round,pad=0.05",
                           facecolor='white',
                           edgecolor='#66BB6A', linewidth=1.5)
ax.add_patch(reg_detail)
ax.text(x_yes, y_branches-0.3, 'Predict demand value', 
        ha='center', va='center', fontsize=9, style='italic')

# Arrow from decision to regression (YES path)
arrow_yes = FancyArrowPatch((x_center+diamond_size, y_decision), (x_yes-1.5, y_branches+0.1),
                           arrowstyle='->', lw=3, color='#2E7D32',
                           connectionstyle="arc3,rad=0.25",
                           mutation_scale=20)
ax.add_patch(arrow_yes)
# YES label with box
yes_label_box = FancyBboxPatch((x_center+0.8, y_decision-0.3), 0.8, 0.4,
                               boxstyle="round,pad=0.1",
                               facecolor='white',
                               edgecolor='#2E7D32', linewidth=2)
ax.add_patch(yes_label_box)
ax.text(x_center+1.2, y_decision-0.1, 'YES', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#2E7D32')

# 4b. NO PATH → ZERO SUPPRESSION (left side)
no_box = FancyBboxPatch((x_no-1.5, y_branches-0.6), 3, 1.2,
                        boxstyle="round,pad=0.15",
                        facecolor=suppress_color,
                        edgecolor='#C62828', linewidth=2.5)
ax.add_patch(no_box)
ax.text(x_no, y_branches+0.1, 'Zero-demand', 
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(x_no, y_branches-0.2, 'Suppression', 
        ha='center', va='center', fontsize=11, fontweight='bold', color='#B71C1C')

# Suppression detail
suppress_detail = FancyBboxPatch((x_no-1.4, y_branches-0.45), 2.8, 0.3,
                                boxstyle="round,pad=0.05",
                                facecolor='white',
                                edgecolor='#EF5350', linewidth=1.5)
ax.add_patch(suppress_detail)
ax.text(x_no, y_branches-0.3, 'Output = 0', 
        ha='center', va='center', fontsize=9, style='italic', color='#C62828', fontweight='bold')

# Arrow from decision to suppression (NO path)
arrow_no = FancyArrowPatch((x_center-diamond_size, y_decision), (x_no+1.5, y_branches+0.1),
                          arrowstyle='->', lw=3, color='#C62828',
                          connectionstyle="arc3,rad=-0.25",
                          mutation_scale=20)
ax.add_patch(arrow_no)
# NO label with box
no_label_box = FancyBboxPatch((x_center-1.6, y_decision-0.3), 0.8, 0.4,
                             boxstyle="round,pad=0.1",
                             facecolor='white',
                             edgecolor='#C62828', linewidth=2)
ax.add_patch(no_label_box)
ax.text(x_center-1.2, y_decision-0.1, 'NO', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#C62828')

# Arrows from both paths to output
arrow_reg_out = FancyArrowPatch((x_yes, y_branches-0.6), (x_center+0.8, y_output+0.5),
                               arrowstyle='->', lw=2.5, color='#424242',
                               connectionstyle="arc3,rad=-0.2",
                               mutation_scale=15)
ax.add_patch(arrow_reg_out)

arrow_suppress_out = FancyArrowPatch((x_no, y_branches-0.6), (x_center-0.8, y_output+0.5),
                                    arrowstyle='->', lw=2.5, color='#424242',
                                    connectionstyle="arc3,rad=0.2",
                                    mutation_scale=15)
ax.add_patch(arrow_suppress_out)

# 5. FINAL OUTPUT
output_box = FancyBboxPatch((x_center-2.2, y_output-0.5), 4.4, 1,
                           boxstyle="round,pad=0.15",
                           facecolor=output_color,
                           edgecolor='#F57F17', linewidth=3)
ax.add_patch(output_box)
ax.text(x_center, y_output, 'Final Prediction\n(Demand Forecast)', 
        ha='center', va='center', fontsize=13, fontweight='bold')

# Add title
ax.text(x_center, 10.6, 'Figure 5 — Workflow of Classify-Then-Regress Pipeline', 
        ha='center', va='center', fontsize=17, fontweight='bold')

# Add explanatory notes
note_box = FancyBboxPatch((1, 0.1), 12, 0.6,
                         boxstyle="round,pad=0.1",
                         facecolor='#FAFAFA',
                         edgecolor='#BDBDBD', linewidth=1)
ax.add_patch(note_box)
note1 = 'α: Classification threshold (optimized per product on validation set)'
note2 = 'Two-stage approach: Classify first, then regress only when non-zero demand is predicted'
ax.text(x_center, 0.45, note1, ha='center', va='center', 
        fontsize=9, style='italic', color='#424242')
ax.text(x_center, 0.2, note2, ha='center', va='center', 
        fontsize=9, style='italic', color='#424242')

# Add step numbers in circles (left side)
step_y_positions = [y_input, y_classify, y_decision, y_branches, y_output]
step_labels = ['1', '2', '3', '4', '5']
step_x = 0.8

for i, (y_pos, label) in enumerate(zip(step_y_positions, step_labels)):
    if i == 3:  # For branches, add both 4a and 4b
        # 4a
        circle = Circle((x_yes+1.8, y_branches), 0.25, 
                       facecolor='white', edgecolor='black', linewidth=2.5, zorder=10)
        ax.add_patch(circle)
        ax.text(x_yes+1.8, y_branches, '4a', ha='center', va='center',
               fontsize=10, fontweight='bold', zorder=11)
        # 4b
        circle = Circle((x_no-1.8, y_branches), 0.25, 
                       facecolor='white', edgecolor='black', linewidth=2.5, zorder=10)
        ax.add_patch(circle)
        ax.text(x_no-1.8, y_branches, '4b', ha='center', va='center',
               fontsize=10, fontweight='bold', zorder=11)
    else:
        circle = Circle((step_x, y_pos), 0.25, 
                       facecolor='white', edgecolor='black', linewidth=2.5, zorder=10)
        ax.add_patch(circle)
        ax.text(step_x, y_pos, label, ha='center', va='center',
               fontsize=11, fontweight='bold', zorder=11)

plt.tight_layout()

# Save figure
output_path = './Son2.py_sonuclar/figure5_ctr_workflow.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_path}")

# Also save as PDF
output_path_pdf = './Son2.py_sonuclar/figure5_ctr_workflow.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_path_pdf}")

plt.close()

"""
Quick visualization of droop from segment analysis CSV.
Shows the Z-profile of the bridge to see droop.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load segment data
df = pd.read_csv('Results_bridge_physics/bridge_segment_analysis.csv')

# Filter to bridge segments (unsupported)
bridge = df[df['has_support'] == False].copy()

print(f"Total segments: {len(df)}")
print(f"Unsupported segments: {len(bridge)}")
print(f"Maximum droop: {bridge['droop_mm'].max():.3f} mm")
print(f"Average droop: {bridge['droop_mm'].mean():.3f} mm")
print(f"Min Z position: {bridge['end_z'].min():.3f} mm")
print(f"Max Z position: {bridge['end_z'].max():.3f} mm")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. XY view - top down
ax = axes[0, 0]
ax.scatter(df['end_x'], df['end_y'], c='lightgray', s=1, alpha=0.5, label='All segments')
scatter = ax.scatter(bridge['end_x'], bridge['end_y'], 
                     c=bridge['droop_mm'], cmap='coolwarm', s=10)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title('Top View - Bridge Segments Colored by Droop')
ax.set_aspect('equal')
plt.colorbar(scatter, ax=ax, label='Droop (mm)')
ax.legend()

# 2. XZ view - side view showing droop
ax = axes[0, 1]
ax.scatter(df['end_x'], df['end_z'], c='lightgray', s=1, alpha=0.5, label='All segments')
scatter = ax.scatter(bridge['end_x'], bridge['end_z'], 
                     c=bridge['droop_mm'], cmap='coolwarm', s=10)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Z (mm)')
ax.set_title('Side View (X-Z) - Showing Vertical Droop')
ax.legend()
plt.colorbar(scatter, ax=ax, label='Droop (mm)')

# 3. YZ view - another side view
ax = axes[1, 0]
ax.scatter(df['end_y'], df['end_z'], c='lightgray', s=1, alpha=0.5, label='All segments')
scatter = ax.scatter(bridge['end_y'], bridge['end_z'], 
                     c=bridge['droop_mm'], cmap='coolwarm', s=10)
ax.set_xlabel('Y (mm)')
ax.set_ylabel('Z (mm)')
ax.set_title('Side View (Y-Z) - Showing Vertical Droop')
ax.legend()
plt.colorbar(scatter, ax=ax, label='Droop (mm)')

# 4. Droop histogram
ax = axes[1, 1]
ax.hist(bridge['droop_mm'], bins=50, edgecolor='black')
ax.set_xlabel('Droop (mm)')
ax.set_ylabel('Count')
ax.set_title(f'Droop Distribution (max={bridge["droop_mm"].max():.1f}mm)')
ax.axvline(bridge['droop_mm'].mean(), color='red', linestyle='--', 
           label=f'Mean: {bridge["droop_mm"].mean():.1f}mm')
ax.legend()

plt.tight_layout()
plt.savefig('Results_bridge_physics/droop_visualization.png', dpi=150)
print("\nVisualization saved to: Results_bridge_physics/droop_visualization.png")
plt.show()

# Show the actual bridge segment details
print("\n" + "="*70)
print("BRIDGE SEGMENT DETAILS (First 10 with significant droop):")
print("="*70)
significant_droop = bridge[bridge['droop_mm'] > 1.0].sort_values('droop_mm', ascending=False)
print(significant_droop[['segment_id', 'end_x', 'end_y', 'end_z', 'droop_mm', 
                         'temperature', 'unsupported_length']].head(10).to_string())

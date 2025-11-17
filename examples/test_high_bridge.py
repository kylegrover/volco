"""Test bridge at high Z to see droop above the bed"""

import json
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from volco import run_simulation

print("="*70)
print("HIGH BRIDGE TEST - Bridge at Z=50mm")
print("="*70)

# Run with physics enabled
output = run_simulation(
    gcode="examples/bridge_high.gcode",
    printer_config="examples/printer_settings.json",
    sim_config={
        "enable_thermal_simulation": True,
        "enable_droop_simulation": True
    },
    output_dir="Results_high_bridge"
)

print(f"\nSTL written to: {output['stl_path']}")

print("\n" + "="*70)
print("DONE! Check Results_high_bridge/")
print("="*70)
print("\nIf droop is working, the bridge segments should sag below Z=50mm")
print("Check the STL in a viewer - you should see the bridge drooping!")

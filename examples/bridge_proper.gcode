; Simple bridge test - 40mm bridge at Z=2.2mm
G21 ; millimeters
G90 ; absolute positioning
M82 ; absolute extrusion
G92 E0 ; reset extrusion

; Build first tower to Z=2.0mm at X=70
G1 X70 Y50 Z0.4 E0.5 F1800
G1 X70 Y50 Z0.8 E1.0
G1 X70 Y50 Z1.2 E1.5
G1 X70 Y50 Z1.6 E2.0
G1 X70 Y50 Z2.0 E2.5

; Travel to second tower (no extrusion)
G1 X110 Y50 Z2.0 F3000

; Build second tower to Z=2.0mm at X=110
G1 X110 Y50 Z0.4 E3.0 F1800
G1 X110 Y50 Z0.8 E3.5
G1 X110 Y50 Z1.2 E4.0
G1 X110 Y50 Z1.6 E4.5
G1 X110 Y50 Z2.0 E5.0

; Move up to bridge height Z=2.2mm (0.2mm gap below bridge)
G1 X110 Y50 Z2.2 F3000

; THE BRIDGE - 40mm horizontal bridge from X=110 to X=70 at Z=2.2
G1 X70 Y50 Z2.2 E7.0 F1800

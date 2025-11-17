; Bridge test at Z=50mm (high enough to see droop above bed)
; Two towers with 10mm bridge between them

; Heating
M140 S50 ; set bed temp and continue
M104 S210 ; set hotend temp and continue
M190 S50 ; set bed temp and wait
M109 S210 ; set hotend temp and wait
M106 S255 ; set fan speed

; First tower (X=70)
G1 Z0.2 F5000
G1 X70 Y50 F3000
G1 E1 F300
G1 X70 Y50 Z50 E100 F1200 ; Spiral up to 50mm

; Second tower (X=80)  
G1 X80 Y50 Z0.2 E1 F3000
G1 X80 Y50 Z50 E100 F1200 ; Spiral up to 50mm

; THE BRIDGE - 10mm span at Z=50mm (high up!)
G1 X70 Y50 Z50 F3000
G1 X80 Y50 Z50 E10 F600 ; Bridge across at constant Z=50

; Done
M104 S0
M140 S0
M106 S0

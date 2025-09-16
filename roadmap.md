to do:

1. optimise speed of fea
1. add periodic boundary conditions for volco simulation
    1. identify input data format
    1. add iteration mechanism for each periodic filament deposition: first sphere set to final sphere size, iterating until difference between first and last sphere size is below tolerance
    1. handle crop-data
    1. consider implications of bottom surface being the top surface. the top surface cannot be printed before the intermitten lines have been printed. This means any filament that is on the z-periodic interface requires iteration of the whole unit cell?
1. demo ability to iterate something easily localised in volco (e.g. extrusion width for all lines) to achieve a target mechanical property (effectively iterate gcode->volco->fea multiple times)



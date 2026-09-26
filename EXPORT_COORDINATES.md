# Voxel export coordinate contract

## Frames

Low-level mesh and streaming STL functions use grid-local **cell corners**:
cell `[i,j,k]` spans `[i*h,(i+1)*h]` on each axis, with centre `(i+0.5)*h`.
Their optional `origin=(0,0,0)` is the position of the cell-zero lower corner.
This intentionally replaces the historical convention that centred cell zero at zero.

`SimulationOutput` exports in input G-code/world coordinates (millimetres):

```
translation = [x_offset - min_printed_x, y_offset - min_printed_y, 0]
world_origin = crop_start * voxel_size - translation
world_corner = local_integer_corner * voxel_size + world_origin
```

`crop_start` is the integer XYZ start of the cropped array in the original grid.
No recentering occurs. Sphere Z offset changes deposition, not the export frame.
Mesh-object, binary STL and ASCII STL share this same corner calculation.
No viewer correction is required. Existing consumers relying on the old half-cell
shift or cropped-local placement must adapt explicitly; GUI integration is separate.

## Crop intervals

Each configured axis pair is a **half-open world interval `[lower, upper)`**.
`all` at either end means that grid boundary. Retain whole cells having a
positive-width intersection with the interval: floor the lower coordinate and
ceil the upper coordinate in grid units, then clamp to the grid. Values within
`1e-9` cell units of an integer are snapped to that integer to avoid translation
cancellation admitting an extra cell. Partial cells are not geometrically cut;
a cropped surface can extend outward to the surrounding cell boundary.

Nonfinite, reversed, zero-width and wholly out-of-grid intervals fail explicitly.
A crop cutting through occupied cells exposes their retained boundary faces.
A crop snapshots only its subarray; it does not mutate deposition occupancy.
Every recrop invalidates the previous mesh, origin and cropped array, even if the
new request fails. Repeated crops refer to the original grid, not the prior crop.

## Precision and representation

Integer corners and the origin transform are evaluated before a single float32
conversion, matching binary STL precision. ASCII writes sufficient digits to
round-trip those same values through a float64 reader. Translated shared corners
therefore remain identical across adjacent faces and exporter chunks.

STL still has finite precision: exceptionally large world coordinates relative
to tiny voxels may collapse distinct float32 positions. Tests cover small solids
at ordinary printer-scale offsets, not all magnitude/resolution combinations.
STL has no explicit unit metadata; millimetres are this application's convention.

Meshing failure no longer silently falls back to marching cubes, which would
change the surface, volume and coordinate convention. Empty occupancy retains
the existing empty-output behavior; a valid crop is not proof of deposited material.

## Evidence

Failing-first coordinate tests plus the updated low-level known-solid bounds:
8 failed / 10 passed before implementation. Final engine suite: **85 passed**.
Tests cover XYZ crops, nonzero translations, both signs of world coordinates,
non-binary voxel pitch, mesh/binary/ASCII parity, normals/watertightness/volume,
repeated and invalid crops, cut surfaces, and translated binary slice seams in
all three axes. These are export correctness checks, not physical validation.

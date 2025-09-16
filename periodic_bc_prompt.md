Add an option for periodic boundary conditions (PUC) in the voxel FEA solver. Follow the existing coding style and package structure. Do not import any new packages beyond the Python standard library or ones already in use. Keep the implementation simple, clear, and future-proof.
Requirements:
API: Add a PeriodicBC configuration (or equivalent kwargs) with fields for enabling/disabling and specifying macroscopic strains:
Exx, Eyy, Ezz (normal strains)
Exy, Exz, Eyz (shear strains; default to engineering shear with γ = 2*Exy etc.)
Options for rigid body removal: "fix-corner" (fix one corner node DOFs) or "zero-mean".
Option for method: "elimination" (preferred) or "lagrange".
Constraints: For a rectangular voxel grid of size (Nx, Ny, Nz) with lengths (Lx, Ly, Lz), enforce periodic displacements with affine jumps:
Pair nodes across opposite faces:
(0,j,k) ↔ (Nx,j,k) with Δu = E·a1
(i,0,k) ↔ (i,Ny,k) with Δu = E·a2
(i,j,0) ↔ (i,j,Nz) with Δu = E·a3
where a1 = [Lx,0,0], a2 = [0,Ly,0], a3 = [0,0,Lz].
Only pair positive faces (x+, y+, z+) to avoid duplicate constraints.
Expand E·a into component equations, e.g. for x-face:
ux(r) − ux(l) = Exx*Lx
uy(r) − uy(l) = Exy*Lx
uz(r) − uz(l) = Exz*Lx.
Implementation:
Build a face-pair map from integer indices.
Build corresponding constant jumps from the chosen strain components.
Apply constraints either by DOF elimination (merge DOFs with constant offset) or by Lagrange multipliers if already supported.
Provide back-substitution to recover the full displacement field after elimination.
Rigid modes:
Default: fix (0,0,0) node DOFs to zero.
Alternative: zero-mean displacement constraints.
Testing:
Uniaxial Exx test: check opposite x-faces differ by Exx*Lx in ux.
Pure shear Exy test: check opposite y-faces differ by 0.5γLy in ux if engineering shear.
Verify solver runs without over-constraint and that rigid body motion is removed.
Documentation:
Add clear docstrings explaining affine periodic constraints and strain component conventions.
Inline comments for why only positive faces are paired.

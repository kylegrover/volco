import numpy as np

from volco_fea import analyze_voxel_matrix


def _grid_index_map(nodes: np.ndarray, voxel_size: float):
    """
    Map node coordinates to integer grid indices based on voxel_size.
    Returns:
        idx_ijk: (N,3) integer indices
        index_map: dict[(ix,iy,iz)] = node_index
        Nxyz: (Nx,Ny,Nz) counts of voxels along axes
        Lxyz: (Lx,Ly,Lz) lengths along axes
        xyz_min: (x_min, y_min, z_min)
    """
    xyz_min = np.min(nodes, axis=0)
    xyz_max = np.max(nodes, axis=0)
    Lx, Ly, Lz = xyz_max - xyz_min
    Nx = int(round(Lx / voxel_size))
    Ny = int(round(Ly / voxel_size))
    Nz = int(round(Lz / voxel_size))
    rel = (nodes - xyz_min[None, :]) / float(voxel_size)
    idx_ijk = np.rint(rel).astype(int)
    index_map = {(ix, iy, iz): i for i, (ix, iy, iz) in enumerate(idx_ijk)}
    return idx_ijk, index_map, (Nx, Ny, Nz), (Lx, Ly, Lz), tuple(xyz_min.tolist())


def _make_full_brick(nx: int, ny: int, nz: int) -> np.ndarray:
    """
    Build a solid voxel brick with ones of shape (nx, ny, nz).
    """
    return np.ones((nx, ny, nz), dtype=np.uint8)


def _make_hollow_brick(nx: int, ny: int, nz: int, wall: int = 1) -> np.ndarray:
    """
    Build a hollow voxel brick: a box frame with wall thickness 'wall'.
    Ensures single connectivity if wall >= 1.
    """
    vm = np.ones((nx, ny, nz), dtype=np.uint8)
    if nx > 2 * wall and ny > 2 * wall and nz > 2 * wall:
        vm[wall:nx - wall, wall:ny - wall, wall:nz - wall] = 0
    return vm


def _make_perforated_slab(nx: int, ny: int, nz: int) -> np.ndarray:
    """
    Build a perforated slab: remove a central region in one middle layer,
    keeping boundary walls intact to preserve connectivity.
    """
    vm = np.ones((nx, ny, nz), dtype=np.uint8)
    if nz >= 3 and nx >= 3 and ny >= 3:
        mid = nz // 2
        vm[1:-1, 1:-1, mid] = 0  # perforation in middle layer
    return vm


def _average_strain(strains: np.ndarray) -> np.ndarray:
    """
    Compute volume-averaged strains from element strains.
    Elements are uniform voxels, so simple mean is volume-weighted mean.
    Voigt order returned by solver is [exx, eyy, ezz, gxy, gyz, gxz] with engineering shear.
    """
    return np.mean(strains, axis=0)


def test_periodic_uniaxial_Exx():
    """
    Uniaxial Exx test:
    For PBC with Exx != 0 (others 0), the opposite x-faces should differ by Exx*Lx in ux,
    and have ~0 jump in uy, uz.
    """
    voxel_size = 1.0
    nx, ny, nz = 2, 2, 2  # produces nodes on a 3x3x3 grid
    vm = _make_full_brick(nx, ny, nz)

    Exx = 0.01  # 1% strain in x
    bc = {
        "periodic": {
            "enabled": True,
            "Exx": Exx,  # normal strain
            "rigid_removal": "fix-corner",
            "method": "elimination",
        }
    }

    results = analyze_voxel_matrix(
        voxel_matrix=vm,
        voxel_size=voxel_size,
        boundary_conditions=bc,
        visualization=False,
    )

    nodes = results["nodes"]
    U = results["displacements"]

    # Sanity: no NaNs
    assert np.isfinite(U).all()

    # Build integer index map
    _, idx_map, (Nx, Ny, Nz), (Lx, Ly, Lz), _ = _grid_index_map(nodes, voxel_size)

    tol = 1e-6
    # Only pair positive faces (x+, y+, z+)
    for iy in range(0, Ny + 1):
        for iz in range(0, Nz + 1):
            l = idx_map[(0, iy, iz)]
            r = idx_map[(Nx, iy, iz)]
            jump = U[r] - U[l]
            # Expected jumps for x-face:
            # [Exx*Lx, Exy*Lx, Exz*Lx] with Exy=Exz=0 here
            assert abs(jump[0] - Exx * Lx) < 1e-6
            assert abs(jump[1] - 0.0) < tol
            assert abs(jump[2] - 0.0) < tol


def test_periodic_pure_shear_Exy_engineering():
    """
    Pure shear Exy test (engineering shear):
    With engineering shear gamma_xy provided via Exy and engineering_shear=True (default),
    the tensor component is E_xy = gamma/2. For y-face (+y vs -y), the affine jump is:
      Δu = [E_xy*Ly, E_yy*Ly, E_yz*Ly] = [0.5*gamma*Ly, 0, 0]
    so ux jump should be 0.5*gamma*Ly; other components ~0.
    """
    voxel_size = 1.0
    nx, ny, nz = 2, 2, 2
    vm = _make_full_brick(nx, ny, nz)

    gamma = 0.02  # engineering shear gamma_xy
    bc = {
        "periodic": {
            "enabled": True,
            "Exy": gamma,  # interpreted as engineering shear; internally converted to tensor Exy=gamma/2
            "rigid_removal": "fix-corner",
            "method": "elimination",
        }
    }

    results = analyze_voxel_matrix(
        voxel_matrix=vm,
        voxel_size=voxel_size,
        boundary_conditions=bc,
        visualization=False,
    )

    nodes = results["nodes"]
    U = results["displacements"]

    # Sanity
    assert np.isfinite(U).all()

    _, idx_map, (Nx, Ny, Nz), (Lx, Ly, Lz), _ = _grid_index_map(nodes, voxel_size)

    tol = 1e-6
    for ix in range(0, Nx + 1):
        for iz in range(0, Nz + 1):
            l = idx_map[(ix, 0, iz)]
            r = idx_map[(ix, Ny, iz)]
            jump = U[r] - U[l]
            # Expected jumps for y-face:
            # [Exy*Ly, Eyy*Ly, Eyz*Ly] with Exy (tensor) = gamma/2 and Eyy=Eyz=0
            expected_ux_jump = 0.5 * gamma * (Ly)
            assert abs(jump[0] - expected_ux_jump) < 1e-6
            assert abs(jump[1] - 0.0) < tol
            assert abs(jump[2] - 0.0) < tol


def test_periodic_average_strain_matches_macro_hollow_box():
    """
    Hollow box geometry: verify volume-averaged strains equal imposed macroscopic strains.
    Tests combined Exx and engineering Exy, with others zero.
    Note: element strains are sampled at element centers; allow small discretization error.
    """
    voxel_size = 1.0
    nx, ny, nz = 4, 4, 3
    vm = _make_hollow_brick(nx, ny, nz, wall=1)

    Exx = 0.015
    gamma_xy = 0.03  # engineering shear

    bc = {
        "periodic": {
            "enabled": True,
            "Exx": Exx,
            "Exy": gamma_xy,  # engineering; internally Exy_tensor = gamma/2
            "rigid_removal": "fix-corner",
            "method": "elimination",
        }
    }

    results = analyze_voxel_matrix(
        voxel_matrix=vm,
        voxel_size=voxel_size,
        boundary_conditions=bc,
        visualization=False,
    )

    # Average strains over elements (engineering shear expected)
    avg_strain = _average_strain(results["strains"])
    exx, eyy, ezz, gxy, gyz, gxz = avg_strain

    # Allow small tolerance due to center-point sampling and discrete mesh
    tol_avg = 2e-3
    assert abs(exx - Exx) < tol_avg
    assert abs(gxy - gamma_xy) < tol_avg
    assert abs(eyy - 0.0) < tol_avg
    assert abs(ezz - 0.0) < tol_avg
    assert abs(gyz - 0.0) < tol_avg
    assert abs(gxz - 0.0) < tol_avg

    # Also verify face jumps for x and y faces numerically (tight tolerance)
    nodes = results["nodes"]
    U = results["displacements"]
    _, idx_map, (Nx, Ny, Nz), (Lx, Ly, Lz), _ = _grid_index_map(nodes, voxel_size)

    # x-face: ux jump = Exx*Lx ; uy jump = Exy_tensor*Lx = 0.5*gamma*Lx
    for iy in range(Ny + 1):
        for iz in range(Nz + 1):
            l = idx_map[(0, iy, iz)]
            r = idx_map[(Nx, iy, iz)]
            jump = U[r] - U[l]
            assert abs(jump[0] - Exx * Lx) < 1e-6
            assert abs(jump[1] - 0.5 * gamma_xy * Lx) < 1e-6
            assert abs(jump[2] - 0.0) < 1e-6

    # y-face: ux jump = Exy_tensor*Ly = 0.5*gamma*Ly ; uy jump = 0
    for ix in range(Nx + 1):
        for iz in range(Nz + 1):
            l = idx_map[(ix, 0, iz)]
            r = idx_map[(ix, Ny, iz)]
            jump = U[r] - U[l]
            assert abs(jump[0] - 0.5 * gamma_xy * Ly) < 1e-6
            assert abs(jump[1] - 0.0) < 1e-6
            assert abs(jump[2] - 0.0) < 1e-6


def test_periodic_zero_mean_translation_and_face_jump_consistency():
    """
    Verify zero-mean rigid removal enforces ~zero average displacement while preserving face jump values.
    """
    voxel_size = 1.0
    nx, ny, nz = 3, 3, 3
    vm = _make_hollow_brick(nx, ny, nz, wall=1)

    Exx = 0.02

    bc_zero_mean = {
        "periodic": {
            "enabled": True,
            "Exx": Exx,
            "rigid_removal": "zero-mean",
            "method": "elimination",
        }
    }
    bc_fix_corner = {
        "periodic": {
            "enabled": True,
            "Exx": Exx,
            "rigid_removal": "fix-corner",
            "method": "elimination",
        }
    }

    res_zero = analyze_voxel_matrix(vm, voxel_size, boundary_conditions=bc_zero_mean, visualization=False)
    res_fix = analyze_voxel_matrix(vm, voxel_size, boundary_conditions=bc_fix_corner, visualization=False)

    U_zero = res_zero["displacements"]
    U_fix = res_fix["displacements"]
    nodes = res_zero["nodes"]

    # Zero mean enforcement
    mean_zero = np.mean(U_zero, axis=0)
    assert np.linalg.norm(mean_zero) < 1e-6

    # Face jump equality for both rigid modes
    _, idx_map, (Nx, Ny, Nz), (Lx, Ly, Lz), _ = _grid_index_map(nodes, voxel_size)
    for iy in range(Ny + 1):
        for iz in range(Nz + 1):
            l = idx_map[(0, iy, iz)]
            r = idx_map[(Nx, iy, iz)]
            jump_zero = U_zero[r] - U_zero[l]
            jump_fix = U_fix[r] - U_fix[l]
            # Expected and consistency between modes
            assert abs(jump_zero[0] - Exx * Lx) < 1e-6
            assert abs(jump_fix[0] - Exx * Lx) < 1e-6
            assert np.allclose(jump_zero, jump_fix, atol=1e-6)


def test_periodic_perforated_slab_average_shear_and_face_jumps():
    """
    Perforated slab geometry: validate average engineering shear and face jump under Exy (engineering).
    Note: element-center sampling implies small discretization error; use moderate tolerance.
    """
    voxel_size = 1.0
    nx, ny, nz = 5, 4, 3
    vm = _make_perforated_slab(nx, ny, nz)

    gamma_xy = 0.04

    bc = {
        "periodic": {
            "enabled": True,
            "Exy": gamma_xy,
            "rigid_removal": "fix-corner",
            "method": "elimination",
        }
    }

    results = analyze_voxel_matrix(
        voxel_matrix=vm,
        voxel_size=voxel_size,
        boundary_conditions=bc,
        visualization=False,
    )

    avg_strain = _average_strain(results["strains"])
    exx, eyy, ezz, gxy, gyz, gxz = avg_strain

    tol_avg = 2e-3
    # Average engineering shear equals imposed gamma; others approx zero
    assert abs(gxy - gamma_xy) < tol_avg
    assert abs(exx - 0.0) < tol_avg
    assert abs(eyy - 0.0) < tol_avg
    assert abs(ezz - 0.0) < tol_avg
    assert abs(gyz - 0.0) < tol_avg
    assert abs(gxz - 0.0) < tol_avg

    # Face jumps along y faces (tight tolerance)
    nodes = results["nodes"]
    U = results["displacements"]
    _, idx_map, (Nx, Ny, Nz), (Lx, Ly, Lz), _ = _grid_index_map(nodes, voxel_size)
    for ix in range(Nx + 1):
        for iz in range(Nz + 1):
            l = idx_map[(ix, 0, iz)]
            r = idx_map[(ix, Ny, iz)]
            jump = U[r] - U[l]
            expected_ux = 0.5 * gamma_xy * Ly  # Exy_tensor = gamma/2
            assert abs(jump[0] - expected_ux) < 1e-6
            assert abs(jump[1]) < 1e-6
            assert abs(jump[2]) < 1e-6
import numpy as np

from volco_fea import analyze_voxel_matrix


def build_cross_columns_voxel(nx: int = 19, ny: int = 19, nz: int = 19) -> np.ndarray:
    """
    Create a 19x19x19 voxel grid with a single central voxel column in each
    of the three directions (x, y, z) intersecting at the center.
    The result is a single connected 'plus' structure.
    """
    vm = np.zeros((nx, ny, nz), dtype=np.uint8)
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    # Central columns along x, y, z through the grid center
    for i in range(nx):
        vm[i, cy, cz] = 1
    for j in range(ny):
        vm[cx, j, cz] = 1
    for k in range(nz):
        vm[cx, cy, k] = 1

    return vm


def lengths_from_nodes(nodes: np.ndarray) -> tuple[float, float, float]:
    xyz_min = np.min(nodes, axis=0)
    xyz_max = np.max(nodes, axis=0)
    Lx, Ly, Lz = xyz_max - xyz_min
    return float(Lx), float(Ly), float(Lz)


def average_normal_stresses(stresses: np.ndarray) -> tuple[float, float, float]:
    """
    Average normal stresses over all elements.
    Voigt order in solver: [sxx, syy, szz, sxy, syz, sxz]
    """
    sxx = float(np.mean(stresses[:, 0]))
    syy = float(np.mean(stresses[:, 1]))
    szz = float(np.mean(stresses[:, 2]))
    return sxx, syy, szz


def average_normal_strains(strains: np.ndarray) -> tuple[float, float, float]:
    """
    Average normal strains over all elements.
    Voigt order in solver: [exx, eyy, ezz, gxy, gyz, gxz] with engineering shear.
    """
    exx = float(np.mean(strains[:, 0]))
    eyy = float(np.mean(strains[:, 1]))
    ezz = float(np.mean(strains[:, 2]))
    return exx, eyy, ezz


def element_centers(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    Compute element centers as the average of corner node coordinates.
    """
    return np.mean(nodes[elements], axis=1)


def slice_force(res: dict, voxel_size: float, axis: int) -> float:
    """
    Compute reaction-like force across the mid-plane normal to 'axis' by integrating
    the corresponding normal stress over that plane using element-center sampling.

    Args:
        res: results dict from analyze_voxel_matrix
        voxel_size: grid spacing
        axis: 0 for x, 1 for y, 2 for z

    Returns:
        Scalar force (N) obtained by summing σ_nn * area over elements whose centers
        lie on the mid-plane normal to 'axis'. Only the normal stress component to the
        chosen surface is used (σ_xx, σ_yy, or σ_zz).
    """
    nodes = res["nodes"]
    elements = res["elements"]
    stresses = res["stresses"]

    centers = element_centers(nodes, elements)
    xyz_min = np.min(nodes, axis=0)
    xyz_max = np.max(nodes, axis=0)
    mid = 0.5 * (xyz_min[axis] + xyz_max[axis])

    # Select elements whose center lies on the mid-plane. Start with tight tol then broaden if empty.
    tol = 1e-6
    mask = np.abs(centers[:, axis] - mid) < tol
    if not np.any(mask):
        tol = voxel_size * 0.51  # broaden to half a voxel to capture the layer
        mask = np.abs(centers[:, axis] - mid) < tol

    # Pick the corresponding normal stress component
    stress_idx = {0: 0, 1: 1, 2: 2}[axis]
    sigma = stresses[mask, stress_idx]  # MPa (N/mm^2)

    area = voxel_size * voxel_size  # mm^2 per element face
    force = float(np.sum(sigma) * area)  # N
    return force


def test_periodic_force_equivalence_cross_columns_19():
    """
    Build a symmetric cross-columns PUC (19^3 voxels) and run three independent periodic tests:
    - Exx = 0.10 (10% macroscopic strain)
    - Eyy = 0.10
    - Ezz = 0.10

    Compute the reaction force from the FEA as:
      F_dir = <σ_dir,avg> * face_area_perpendicular_to_dir

    Because the geometry is symmetric under x/y/z permutation and the material is isotropic,
    the required force should be the same in all three directions within a small tolerance.
    """
    voxel_size = 1.0
    vm = build_cross_columns_voxel(19, 19, 19)

    # Specific material properties
    E = 1500.0  # MPa
    nu = 0.30
    strain = 0.10  # 10%

    material = {"young_modulus": E, "poisson_ratio": nu}

    # Periodic BCs for each loading direction (affine periodic strain)
    bc_x = {"periodic": {"enabled": True, "Exx": strain, "rigid_removal": "fix-corner", "method": "elimination"}}
    bc_y = {"periodic": {"enabled": True, "Eyy": strain, "rigid_removal": "fix-corner", "method": "elimination"}}
    bc_z = {"periodic": {"enabled": True, "Ezz": strain, "rigid_removal": "fix-corner", "method": "elimination"}}

    # Run analyses
    res_x = analyze_voxel_matrix(vm, voxel_size, material_properties=material, boundary_conditions=bc_x, visualization=False)
    res_y = analyze_voxel_matrix(vm, voxel_size, material_properties=material, boundary_conditions=bc_y, visualization=False)
    res_z = analyze_voxel_matrix(vm, voxel_size, material_properties=material, boundary_conditions=bc_z, visualization=False)

    # Geometry lengths (same for all three)
    Lx, Ly, Lz = lengths_from_nodes(res_x["nodes"])

    # Average normal stresses (MPa = N/mm^2)
    sxx_avg, _, _ = average_normal_stresses(res_x["stresses"])
    _, syy_avg, _ = average_normal_stresses(res_y["stresses"])
    _, _, szz_avg = average_normal_stresses(res_z["stresses"])

    # Compute reaction forces by integrating normal stress across the mid-plane
    # normal to the loading direction. Only the normal component is considered.
    Fx = slice_force(res_x, voxel_size, axis=0)
    Fy = slice_force(res_y, voxel_size, axis=1)
    Fz = slice_force(res_z, voxel_size, axis=2)

    # Forces should be equal within a small relative tolerance due to symmetry
    rtol = 2e-2
    assert np.isfinite([Fx, Fy, Fz]).all()
    assert np.allclose(Fx, Fy, rtol=rtol, atol=0.0)
    assert np.allclose(Fx, Fz, rtol=rtol, atol=0.0)
    assert np.allclose(Fy, Fz, rtol=rtol, atol=0.0)
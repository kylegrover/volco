"""
Analysis utilities for voxel FEA results.

These helpers expose simple, reusable functions to compute:
- Mid-plane resultant force in a chosen direction (normal component only)
- Opposite-face average displacement jump in a chosen direction
- Cell lengths, volume, and effective stress/modulus based on force and imposed strain
- Average strain component (element-center sampling)

Design:
- Keep dependencies minimal (NumPy only)
- Accept standard 'results' dict produced by analyze_voxel_matrix for convenience
- Use voxel_size to determine per-element face area and mid-plane selection tolerance
"""

from typing import Tuple
import numpy as np


def get_cell_lengths(nodes: np.ndarray) -> Tuple[float, float, float]:
    """
    Return (Lx, Ly, Lz) extents from node coordinates.
    """
    xyz_min = np.min(nodes, axis=0)
    xyz_max = np.max(nodes, axis=0)
    Lx, Ly, Lz = (xyz_max - xyz_min).astype(float)
    return float(Lx), float(Ly), float(Lz)


def get_cell_volume(nodes: np.ndarray) -> float:
    """
    Return cell volume V_cell = Lx * Ly * Lz from node coordinates.
    """
    Lx, Ly, Lz = get_cell_lengths(nodes)
    return float(Lx * Ly * Lz)


def _element_centers(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    Compute element centers as the average of corner node coordinates.
    """
    return np.mean(nodes[elements], axis=1)


def compute_midplane_force(results: dict, voxel_size: float, axis: int = 2) -> float:
    """
    Compute resultant force across the mid-plane normal to 'axis' by integrating the
    corresponding normal stress component over elements whose centers lie near that plane.

    Args:
        results: dict from analyze_voxel_matrix
        voxel_size: grid spacing (mm)
        axis: 0=x, 1=y, 2=z

    Returns:
        Force in N (MPa * mm^2).
    """
    nodes = results["nodes"]
    elements = results["elements"]
    stresses = results["stresses"]

    centers = _element_centers(nodes, elements)
    xyz_min = np.min(nodes, axis=0)
    xyz_max = np.max(nodes, axis=0)
    mid = 0.5 * (xyz_min[axis] + xyz_max[axis])

    # Select elements near the mid-plane. If none, broaden tolerance to half a voxel.
    tol = 1e-6
    mask = np.abs(centers[:, axis] - mid) < tol
    if not np.any(mask):
        tol = voxel_size * 0.51
        mask = np.abs(centers[:, axis] - mid) < tol

    # Normal component index matches axis
    sigma = stresses[mask, axis]  # MPa (N/mm^2)
    area = voxel_size * voxel_size  # mm^2 per element face
    return float(np.sum(sigma) * area)  # N


def compute_face_displacement_jump(results: dict, voxel_size: float, axis: int = 2) -> float:
    """
    Compute the average opposite-face displacement jump along the specified axis.
    Only pair positive faces to avoid duplicates.

    Examples:
      axis=2 (z): average of uz(r) - uz(l) across pairs on (iz_max vs iz_min)
      axis=1 (y): average of uy(r) - uy(l), across (iy_max vs iy_min)
      axis=0 (x): average of ux(r) - ux(l), across (ix_max vs ix_min)

    Args:
        results: dict from analyze_voxel_matrix
        voxel_size: grid spacing
        axis: 0=x, 1=y, 2=z

    Returns:
        Average jump (float). NaN if no pairs were found.
    """
    nodes = results["nodes"]
    U = results["displacements"]

    # Map to integer grid indices tolerant to rounding
    rel = (nodes - np.min(nodes, axis=0)[None, :]) / float(voxel_size)
    ijk = np.rint(rel).astype(int)

    # Extract min/max layer indices along the chosen axis
    a_min = int(np.min(ijk[:, axis]))
    a_max = int(np.max(ijk[:, axis]))

    # Build dictionary (ix,iy,iz) -> node index
    idx_map = {(ix, iy, iz): i for i, (ix, iy, iz) in enumerate(ijk)}

    # Build pairing loops over the two transverse axes
    axes = [0, 1, 2]
    axes.remove(axis)
    t0, t1 = axes[0], axes[1]

    t0_vals = range(int(np.min(ijk[:, t0])), int(np.max(ijk[:, t0])) + 1)
    t1_vals = range(int(np.min(ijk[:, t1])), int(np.max(ijk[:, t1])) + 1)

    # Component index for displacement equals axis
    comp = axis
    jumps = []
    for a in [(t0_val, t1_val) for t0_val in t0_vals for t1_val in t1_vals]:
        coords_min = [None, None, None]
        coords_max = [None, None, None]
        coords_min[axis] = a_min
        coords_max[axis] = a_max
        coords_min[t0] = a[0]
        coords_max[t0] = a[0]
        coords_min[t1] = a[1]
        coords_max[t1] = a[1]
        l = idx_map.get(tuple(coords_min))
        r = idx_map.get(tuple(coords_max))
        if l is None or r is None:
            continue
        jumps.append(U[r, comp] - U[l, comp])

    return float(np.mean(jumps)) if len(jumps) > 0 else float("nan")


def compute_effective_stress(force: float, nodes: np.ndarray, axis: int = 2) -> float:
    """
    Compute effective macroscopic stress: sigma_eff = F / A_perp,
    where A_perp is the area perpendicular to 'axis' (e.g., for axis=z, A = Lx * Ly).

    Args:
        force: resultant force (N)
        nodes: array of node coordinates
        axis: 0=x, 1=y, 2=z

    Returns:
        Effective stress in MPa (N/mm^2). Note: inputs are in mm units (consistent with codebase).
    """
    Lx, Ly, Lz = get_cell_lengths(nodes)
    if axis == 0:
        area = Ly * Lz
    elif axis == 1:
        area = Lx * Lz
    else:
        area = Lx * Ly
    if area == 0.0:
        return float("nan")
    return float(force / area)  # N / mm^2 = MPa


def compute_effective_modulus(force: float, nodes: np.ndarray, imposed_strain: float, axis: int = 2) -> float:
    """
    Compute effective modulus E_eff = sigma_eff / epsilon_macro
    where sigma_eff = F / A_perp and epsilon_macro is the imposed macroscopic strain.

    Args:
        force: resultant force (N)
        nodes: array of node coordinates
        imposed_strain: scalar macro strain applied along 'axis' (e.g., -0.05)
        axis: 0=x, 1=y, 2=z

    Returns:
        Effective modulus in MPa.
    """
    if imposed_strain == 0.0:
        return float("nan")
    sigma_eff = compute_effective_stress(force, nodes, axis=axis)
    return float(sigma_eff / imposed_strain)


def average_strain_component(results: dict, component_index: int = 2) -> float:
    """
    Average a strain component over all elements using element-center sampling.

    Args:
        results: dict from analyze_voxel_matrix (contains 'strains')
        component_index: 0=exx, 1=eyy, 2=ezz, 3=gxy, 4=gyz, 5=gxz

    Returns:
        Average component value (float).
    """
    strains = results["strains"]
    return float(np.mean(strains[:, component_index]))
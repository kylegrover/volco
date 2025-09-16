"""
Periodic boundary conditions (PUC) support for voxel FEA.

This module implements affine periodic boundary constraints for a rectangular voxel grid.
It provides:
- Pairing of nodes across opposite faces (only positive faces are paired to avoid duplicates)
- Construction of affine displacement jumps from macroscopic strain tensor E
- DOF elimination by expressing slave DOFs in terms of master DOFs plus constant offsets
- Back-substitution to recover full-field displacements
- Rigid body removal options: "fix-corner" (default) and "zero-mean"

Shear strain convention:
- By default engineering shear is assumed with γ = 2*E_xy etc.
- Inputs Exy, Exz, Eyz are interpreted as engineering shear if `engineering_shear=True`.
- Internally constraints are built using tensor shear components E_xy = γ/2.

Limitations:
- Assumes the voxelized model spans a rectangular grid and all boundary nodes exist.
- Only translational DOFs are present (consistent with current solver).
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix

# Reuse core solver building blocks to remain consistent with the codebase
from .solver import (
    assemble_system,
    solve_system,
    calculate_stresses_and_strains,
    apply_boundary_conditions as apply_dirichlet_bc,  # reuse for "fix-corner"
)


def _normalize_periodic_cfg(cfg_in: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize PeriodicBC configuration with defaults."""
    cfg = dict(cfg_in or {})
    cfg.setdefault("enabled", False)
    cfg.setdefault("method", "elimination")  # "elimination" | "lagrange" (not implemented)
    cfg.setdefault("rigid_removal", "fix-corner")  # "fix-corner" | "zero-mean"
    cfg.setdefault("engineering_shear", True)

    # Macroscopic strain components (tensor components by default)
    for k in ["Exx", "Eyy", "Ezz", "Exy", "Exz", "Eyz"]:
        cfg.setdefault(k, 0.0)

    # If engineering shear specified, convert to tensor components
    if cfg.get("engineering_shear", True):
        # Exy, Exz, Eyz provided as engineering gamma; convert to E tensor
        cfg["Exy"] = 0.5 * cfg["Exy"]
        cfg["Exz"] = 0.5 * cfg["Exz"]
        cfg["Eyz"] = 0.5 * cfg["Eyz"]

    return cfg


def _build_integer_grid_indices(nodes: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], int], Tuple[int, int, int], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Map node coordinates to integer grid indices (ix, iy, iz) based on voxel_size.

    Returns:
        - idx_ijk: (N, 3) array of integer indices for each node
        - index_map: dict mapping (ix, iy, iz) -> node_index
        - Nxyz: tuple of voxel counts along each axis (Nx, Ny, Nz)
        - xyz_min: tuple (x_min, y_min, z_min)
        - Lxyz: tuple (Lx, Ly, Lz)
    """
    xyz_min = tuple(np.min(nodes, axis=0).tolist())
    xyz_max = tuple(np.max(nodes, axis=0).tolist())
    Lx, Ly, Lz = xyz_max[0] - xyz_min[0], xyz_max[1] - xyz_min[1], xyz_max[2] - xyz_min[2]

    # Compute expected integer counts (Nx, Ny, Nz) of voxel cells from span/voxel_size
    # Use rounding to tolerate minor floating errors
    Nx = int(round(Lx / voxel_size))
    Ny = int(round(Ly / voxel_size))
    Nz = int(round(Lz / voxel_size))

    # Map nodes to integer coordinates on the grid
    rel = (nodes - np.array(xyz_min)[None, :]) / float(voxel_size)
    idx_ijk = np.rint(rel).astype(int)

    # Build index_map
    index_map: Dict[Tuple[int, int, int], int] = {}
    for i, (ix, iy, iz) in enumerate(idx_ijk):
        index_map[(ix, iy, iz)] = i

    return idx_ijk, index_map, (Nx, Ny, Nz), xyz_min, (Lx, Ly, Lz)


def _affine_jump_components(E: Dict[str, float], Lxyz: Tuple[float, float, float]) -> Dict[str, np.ndarray]:
    """
    Expand E·a into component jump vectors for each paired face.

    For x-face: Δu = [Exx*Lx, Exy*Lx, Exz*Lx]
    For y-face: Δu = [Exy*Ly, Eyy*Ly, Eyz*Ly]
    For z-face: Δu = [Exz*Lz, Eyz*Lz, Ezz*Lz]
    """
    Lx, Ly, Lz = Lxyz
    Exx, Eyy, Ezz = E["Exx"], E["Eyy"], E["Ezz"]
    Exy, Exz, Eyz = E["Exy"], E["Exz"], E["Eyz"]

    return {
        "x": np.array([Exx * Lx, Exy * Lx, Exz * Lx], dtype=float),
        "y": np.array([Exy * Ly, Eyy * Ly, Eyz * Ly], dtype=float),
        "z": np.array([Exz * Lz, Eyz * Lz, Ezz * Lz], dtype=float),
    }


def _build_face_pairs(index_map: Dict[Tuple[int, int, int], int], Nxyz: Tuple[int, int, int]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build face-pair node mappings for positive faces only.

    Returns:
        dict with keys 'x', 'y', 'z' and lists of (left, right) node index pairs:
        - x: (ix=0, iy, iz)  <-> (ix=Nx, iy, iz)
        - y: (ix, iy=0, iz)  <-> (ix, iy=Ny, iz)
        - z: (ix, iy, iz=0)  <-> (ix, iy, iz=Nz)
    """
    Nx, Ny, Nz = Nxyz
    pairs = {"x": [], "y": [], "z": []}

    # x+ face pairs
    for iy in range(0, Ny + 1):
        for iz in range(0, Nz + 1):
            l = index_map.get((0, iy, iz))
            r = index_map.get((Nx, iy, iz))
            if l is not None and r is not None:
                pairs["x"].append((l, r))

    # y+ face pairs
    for ix in range(0, Nx + 1):
        for iz in range(0, Nz + 1):
            l = index_map.get((ix, 0, iz))
            r = index_map.get((ix, Ny, iz))
            if l is not None and r is not None:
                pairs["y"].append((l, r))

    # z+ face pairs
    for ix in range(0, Nx + 1):
        for iy in range(0, Ny + 1):
            l = index_map.get((ix, iy, 0))
            r = index_map.get((ix, iy, Nz))
            if l is not None and r is not None:
                pairs["z"].append((l, r))

    return pairs


def _union_find_offsets(num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize parent and shift arrays for union-find with affine offsets.

    parent[i] = parent representative index
    shift[i] = 3-vector offset from node i to its parent such that:
        u_i = u_parent[i] + shift[i]
    """
    parent = np.arange(num_nodes, dtype=int)
    shift = np.zeros((num_nodes, 3), dtype=float)
    return parent, shift


def _find_root_with_total_shift(i: int, parent: np.ndarray, shift: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Find root representative of node i with path compression, accumulating total shift to root.

    Returns:
        (root_index, total_shift_vector) where total_shift_vector satisfies:
        u_i = u_root + total_shift_vector
    """
    if parent[i] == i:
        return i, shift[i].copy()
    # Recurse
    root, s_to_root = _find_root_with_total_shift(parent[i], parent, shift)
    # Accumulate and compress path
    shift[i] = shift[i] + s_to_root
    parent[i] = root
    return parent[i], shift[i].copy()


def _unify_slave_to_master(slave: int, master: int, c_vec: np.ndarray, parent: np.ndarray, shift: np.ndarray) -> None:
    """
    Impose constraint u_slave = u_master + c_vec by union in the forest.

    Note: We always direct parents from positive-face nodes (slaves) to negative-face nodes (masters),
    so cycles cannot form with the prescribed processing order.
    """
    # If already same set, nothing to do; otherwise attach slave under master with shift
    # Accumulate current roots to avoid overriding valid chains
    r_slave, t_slave = _find_root_with_total_shift(slave, parent, shift)
    r_master, t_master = _find_root_with_total_shift(master, parent, shift)

    if r_slave == r_master:
        return

    # We want: u_slave = u_master + c_vec
    # Currently: u_slave = u_r_slave + t_slave ; u_master = u_r_master + t_master
    # Enforce: u_r_slave + t_slave = u_r_master + t_master + c_vec
    # Attach r_slave under r_master with shift_delta such that:
    # u_r_slave = u_r_master + (t_master + c_vec - t_slave)
    attach_shift = (t_master + c_vec) - t_slave

    parent[r_slave] = r_master
    shift[r_slave] = attach_shift


def _build_mapping_and_offsets(nodes: np.ndarray, voxel_size: float, pairs: Dict[str, List[Tuple[int, int]]], jumps: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build mapping from each node to its master (root) and the constant offset from root.

    Returns:
        - master_node: (N,) array of root node indices
        - offset_from_master: (N,3) array with u_i = u_master + offset_from_master[i]
    """
    N = nodes.shape[0]
    parent, shift = _union_find_offsets(N)

    # Process positive faces in order x, y, z for predictable roots toward (x-, y-, z-)
    for face in ("x", "y", "z"):
        c = jumps[face]
        for l, r in pairs[face]:
            # constrain right (positive face) to left (negative face) with jump
            _unify_slave_to_master(slave=r, master=l, c_vec=c, parent=parent, shift=shift)

    # Compress and compute offsets to root
    master_node = np.empty(N, dtype=int)
    offset_from_master = np.zeros((N, 3), dtype=float)
    for i in range(N):
        root, t = _find_root_with_total_shift(i, parent, shift)
        master_node[i] = root
        offset_from_master[i] = t

    return master_node, offset_from_master


def _build_projection_and_constant(master_node: np.ndarray, offset_from_master: np.ndarray) -> Tuple[coo_matrix, np.ndarray, np.ndarray]:
    """
    Construct projection matrix P and constant c such that:
        u_full = P * u_master + c
    Also returns:
        master_list: unique sorted array of master node indices
    """
    N = master_node.shape[0]
    dofs_full = N * 3

    # Unique masters and mapping to reduced indices
    master_list, inv = np.unique(master_node, return_inverse=True)
    M = master_list.shape[0]  # number of master nodes
    dofs_red = M * 3

    # Row/col/data for P in COO
    rows = np.arange(dofs_full, dtype=int)
    cols = np.empty(dofs_full, dtype=int)
    data = np.ones(dofs_full, dtype=float)

    # For each node i and component d, map to its master node reduced index
    # inv[i] gives index of master_list for node i
    # reduced_col = (inv[i] * 3 + d)
    inv_repeat = np.repeat(inv, 3)
    comp = np.tile(np.array([0, 1, 2], dtype=int), N)
    cols = inv_repeat * 3 + comp

    P = coo_matrix((data, (rows, cols)), shape=(dofs_full, dofs_red))

    # Constant c (dofs_full,) from offsets
    c = offset_from_master.reshape(dofs_full)

    return P, c, master_list


def _apply_rigid_body_removal(
    K_r: lil_matrix,
    f_r: np.ndarray,
    master_list: np.ndarray,
    rigid_mode: str,
    nodes: np.ndarray,
    voxel_size: float,
) -> Tuple[lil_matrix, np.ndarray]:
    """
    Apply rigid body removal on the reduced system.

    - "fix-corner": fix the (x_min, y_min, z_min) corner master node (ux=uy=uz=0) using Dirichlet.
    - "zero-mean": no conditioning here; handled post back-substitution by subtracting mean.
    """
    if rigid_mode == "zero-mean":
        return K_r, f_r

    # Find a robust "corner" node index in full nodes.
    # Prefer exact (min_x, min_y, min_z); if none, fall back to lexicographic minimum (x, then y, then z).
    xyz_min = np.min(nodes, axis=0)
    tol = 1e-9
    candidates = np.where(
        (np.abs(nodes[:, 0] - xyz_min[0]) < tol) &
        (np.abs(nodes[:, 1] - xyz_min[1]) < tol) &
        (np.abs(nodes[:, 2] - xyz_min[2]) < tol)
    )[0]
    if candidates.size > 0:
        origin_full_idx = int(candidates[0])
    else:
        order = np.lexsort((nodes[:, 2], nodes[:, 1], nodes[:, 0]))
        origin_full_idx = int(order[0])

    # Identify its master
    # master_list contains the node indices of masters; find if origin_full_idx is a master
    try:
        origin_master_pos = int(np.where(master_list == origin_full_idx)[0][0])
    except IndexError:
        # If for some reason origin is not a master (shouldn't happen with our pairing), pick the first master
        origin_master_pos = 0

    # Build reduced DOFs dict: one node fixed (ux, uy, uz) = 0
    num_nodes_red = master_list.shape[0]
    dofs_red = {origin_master_pos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    K_c, f_c = apply_dirichlet_bc(K_r, f_r, dofs_red, num_nodes_red)
    return K_c, f_c


def solve_static_problem_periodic(
    nodes: np.ndarray,
    elements: np.ndarray,
    voxel_size: float,
    material_properties: Dict[str, float],
    periodic_cfg_in: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve linear static problem under affine periodic boundary conditions using DOF elimination.

    Args:
        nodes: (N x 3) node coordinates
        elements: (E x 8) element connectivity
        voxel_size: grid spacing
        material_properties: material dictionary {'young_modulus', 'poisson_ratio'}
        periodic_cfg_in: dict with keys:
          - enabled: bool
          - Exx, Eyy, Ezz, Exy, Exz, Eyz: float (see shear convention)
          - engineering_shear: bool (default True; if True, Exy=γ_xy, internally converted to tensor)
          - method: 'elimination' (default) | 'lagrange' (not implemented)
          - rigid_removal: 'fix-corner' (default) | 'zero-mean'

    Returns:
        displacements (N x 3), stresses (E x 6), strains (E x 6), von_mises (E,)
    """
    cfg = _normalize_periodic_cfg(periodic_cfg_in)
    if not cfg.get("enabled", False):
        raise ValueError("PeriodicBC enabled=False; call the standard path instead.")

    if cfg.get("method", "elimination") != "elimination":
        raise NotImplementedError("Periodic Lagrange multiplier method is not implemented in this solver.")

    # Assemble full system (no external forces for PBC unless provided elsewhere)
    forces = np.zeros((nodes.shape[0], 6), dtype=float)
    K, f = assemble_system(nodes, elements, forces, material_properties["young_modulus"], material_properties["poisson_ratio"])

    # Build grid and pair maps
    _, index_map, Nxyz, xyz_min, Lxyz = _build_integer_grid_indices(nodes, voxel_size)
    pairs = _build_face_pairs(index_map, Nxyz)
    jumps = _affine_jump_components(
        {
            "Exx": cfg["Exx"],
            "Eyy": cfg["Eyy"],
            "Ezz": cfg["Ezz"],
            "Exy": cfg["Exy"],
            "Exz": cfg["Exz"],
            "Eyz": cfg["Eyz"],
        },
        Lxyz,
    )

    # Build mapping and constant offsets
    master_node, offset_from_master = _build_mapping_and_offsets(nodes, voxel_size, pairs, jumps)

    # Build projection P and constant c s.t. u_full = P u_red + c
    P, c_const, master_list = _build_projection_and_constant(master_node, offset_from_master)

    # Reduce system: (P^T K P) u_r = P^T (f - K c)
    K_csr = K.tocsr()
    P_csr = P.tocsr()
    Kr = (P_csr.T @ K_csr @ P_csr).tolil()
    fr = (P_csr.T @ (f - K_csr @ c_const)).astype(float)

    # Rigid body removal on reduced system
    Kr_c, fr_c = _apply_rigid_body_removal(Kr, fr, master_list, cfg.get("rigid_removal", "fix-corner"), nodes, voxel_size)

    # Solve reduced system
    u_red = solve_system(Kr_c, fr_c)

    # Back-substitute full field
    u_full = (P_csr @ u_red) + c_const
    displacements = u_full.reshape(nodes.shape[0], 3)

    # Optional zero-mean displacement removal (translation only)
    if cfg.get("rigid_removal") == "zero-mean":
        mean_u = np.mean(displacements, axis=0, keepdims=True)
        displacements = displacements - mean_u

    # Post-processing
    strains, stresses, von_mises = calculate_stresses_and_strains(
        nodes, elements, displacements, material_properties["young_modulus"], material_properties["poisson_ratio"]
    )

    return displacements, stresses, strains, von_mises
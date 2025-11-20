import logging
import trimesh
import numpy as np
from skimage import measure
import scipy.sparse
import multiprocessing
from scipy.sparse import coo_matrix
import tracemalloc
import time
import struct

logger = logging.getLogger(__name__)

# Log NumPy version for debugging purposes
logger.info(f"[Mesh]: Using NumPy {np.__version__}")

def create_mesh_vectorized(voxel_space, voxel_size):
    """
    Create a mesh from a voxel space using vectorized operations.
    This is much faster than iterating over voxels in Python.
    
    Parameters:
    -----------
    voxel_space : numpy.ndarray
        The 3D voxel space array
    voxel_size : float
        The size of each voxel
        
    Returns:
    --------
    trimesh.Trimesh
        A mesh containing only visible faces of filled voxels
    """
    logger.info("[Profile]: Starting mesh generation (vectorized)")
    tracemalloc.start()
    t0 = time.perf_counter()
    
    # Cube vertices (relative to center)
    cube = voxel_size * np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ], dtype=np.float32)
    
    # Face definitions: (vertex indices, normal vector)
    # Fixed -Z face winding order to [0, 3, 2, 1] for correct normal
    faces = [
        ([0, 3, 2, 1], np.array([0, 0, -1], dtype=np.float32)),  # -Z
        ([4, 5, 6, 7], np.array([0, 0, 1], dtype=np.float32)),   # +Z
        ([0, 1, 5, 4], np.array([0, -1, 0], dtype=np.float32)),  # -Y
        ([2, 3, 7, 6], np.array([0, 1, 0], dtype=np.float32)),   # +Y
        ([0, 4, 7, 3], np.array([-1, 0, 0], dtype=np.float32)),  # -X
        ([1, 2, 6, 5], np.array([1, 0, 0], dtype=np.float32))    # +X
    ]

    filled = (voxel_space > 0)
    all_vertices = []
    all_faces = []
    vertex_count = 0
    
    for face_idx, (face_verts, normal) in enumerate(faces):
        # Build surface mask for this face direction
        mask = np.zeros_like(filled, dtype=bool)
        
        if face_idx == 0:  # -Z face
            mask[:, :, :-1] = filled[:, :, :-1] & ~filled[:, :, 1:]
        elif face_idx == 1:  # +Z face
            mask[:, :, 1:] = filled[:, :, 1:] & ~filled[:, :, :-1]
        elif face_idx == 2:  # -Y face
            mask[:, :-1, :] = filled[:, :-1, :] & ~filled[:, 1:, :]
        elif face_idx == 3:  # +Y face
            mask[:, 1:, :] = filled[:, 1:, :] & ~filled[:, :-1, :]
        elif face_idx == 4:  # -X face
            mask[:-1, :, :] = filled[:-1, :, :] & ~filled[1:, :, :]
        elif face_idx == 5:  # +X face
            mask[1:, :, :] = filled[1:, :, :] & ~filled[:-1, :, :]
        
        # Get indices of voxels with visible faces
        indices = np.array(np.where(mask)).T
        
        if indices.size == 0:
            continue
        
        # Compute voxel centers
        centers = indices.astype(np.float32) * voxel_size
        
        # Precompute face vertices from cube
        face_vertices_template = cube[face_verts]  # shape (4, 3)
        
        # Build vertices for all voxels in this face direction: shape (M, 4, 3)
        # Broadcasting: (M, 1, 3) + (1, 4, 3) -> (M, 4, 3)
        verts = centers[:, np.newaxis, :] + face_vertices_template[np.newaxis, :, :]
        
        # Reshape to (M*4, 3) to add to vertex list
        verts_flat = verts.reshape(-1, 3)
        
        # Create faces
        # Each voxel face has 4 vertices (0, 1, 2, 3) relative to the start of its block
        # We want to make 2 triangles: [0, 1, 2] and [0, 2, 3]
        M = indices.shape[0]
        
        # Base indices for each voxel's vertices
        base_indices = np.arange(0, M * 4, 4) + vertex_count
        
        # Triangle 1: 0, 1, 2
        t1 = np.column_stack([base_indices, base_indices + 1, base_indices + 2])
        # Triangle 2: 0, 2, 3
        t2 = np.column_stack([base_indices, base_indices + 2, base_indices + 3])
        
        current_faces = np.vstack([t1, t2])
        
        all_vertices.append(verts_flat)
        all_faces.append(current_faces)
        
        vertex_count += M * 4

    t1 = time.perf_counter()
    mem1 = tracemalloc.get_traced_memory()
    logger.info(f"[Profile]: Mesh generation complete. Time: {t1-t0:.2f}s, Mem: current={mem1[0]/1e6:.2f}MB, peak={mem1[1]/1e6:.2f}MB")
    tracemalloc.stop()
    
    if all_vertices:
        vertices_array = np.vstack(all_vertices)
        faces_array = np.vstack(all_faces)
        mesh = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)
        return mesh
    else:
        return trimesh.Scene()

def generate_mesh_from_voxels(voxel_space, voxel_size):
    """
    Generate a 3D mesh from the voxel data using optimized box representation.
    This implementation only creates triangles for voxel faces that are either:
    1. At the boundary of the voxel matrix
    2. Adjacent to an empty voxel
    
    Parameters:
    -----------
    voxel_space : numpy.ndarray
        The 3D voxel space array
    voxel_size : float
        The size of each voxel
        
    Returns:
    --------
    trimesh.Trimesh
        The generated mesh
    """
    if voxel_space is None:
        logger.warning("[Mesh]: No voxel space provided.")
        return None
    
    try:
        logger.info("[Mesh]: Using optimized box representation")
        voxels = create_mesh_vectorized(voxel_space, voxel_size)
        return voxels
    except Exception as e:
        logger.error(f"[Mesh]: Failed to generate mesh with optimized box representation: {e}")
        
        # Fall back to using marching cubes
        try:
            logger.warning("[Mesh]: Falling back to marching cubes method")
            mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_space, pitch=voxel_size)
            return mesh
        except Exception as e:
            logger.warning(f"[Mesh]: Failed to generate mesh with trimesh: {e}")
            
            # Fall back to using skimage directly
            try:
                logger.warning("[Mesh]: Falling back to skimage marching cubes")
                verts, faces, normals, values = measure.marching_cubes(voxel_space, level=0.5)
                verts = verts * voxel_size
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                return mesh
            except Exception as e:
                logger.error(f"[Mesh]: Failed to generate mesh with skimage: {e}")
                return None

def export_mesh_to_stl(mesh, file_path, ascii_format=True):
    """
    Export the mesh to an STL file.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh or trimesh.Scene
        The mesh to export
    file_path : str
        The path to save the STL file
    ascii_format : bool
        Whether to export in ASCII format (True) or binary format (False)
            
    Returns:
    --------
    str
        The path to the exported STL file
    """
    if mesh is None:
        logger.warning("[Mesh]: No mesh provided for export.")
        return None
        
    try:
        format_type = "ASCII" if ascii_format else "binary"
        logger.info(f"[Mesh]: Exporting STL in {format_type} format to path {file_path}...")
        
        export_options = {'file_type': 'stl_ascii' if ascii_format else 'stl'}
        trimesh.exchange.export.export_mesh(mesh, file_path, **export_options)
            
        logger.info(f"[Mesh]: STL exported in {format_type} format!")
        return file_path
    except Exception as e:
        logger.error(f"[Mesh]: Failed to export STL: {e}")
        return None

def export_voxel_stl_streaming_binary(voxel_space, voxel_size, file_path):
    """
    Export a voxel space as a binary STL file using streaming (low-memory) logic.
    Only surface faces are written, and no full mesh is held in RAM.
    
    This optimized version uses vectorized operations to process chunks of voxels
    at once, dramatically improving performance for large voxel grids.
    
    Parameters:
    -----------
    voxel_space : numpy.ndarray
        The 3D voxel space array
    voxel_size : float
        The size of each voxel
    file_path : str
        The path to save the binary STL file
        
    Returns:
    --------
    str
        The path to the exported STL file
    """
    logger.info("[Profile]: Starting vectorized streaming binary STL export")
    tracemalloc.start()
    t0 = time.perf_counter()

    # Cube vertices (relative to center)
    cube = voxel_size * np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ], dtype=np.float32)
    
    # Face definitions: (vertex indices, normal vector)
    # Fixed -Z face winding order to [0, 3, 2, 1] for correct normal
    faces = [
        ([0, 3, 2, 1], np.array([0, 0, -1], dtype=np.float32)),  # -Z
        ([4, 5, 6, 7], np.array([0, 0, 1], dtype=np.float32)),   # +Z
        ([0, 1, 5, 4], np.array([0, -1, 0], dtype=np.float32)),  # -Y
        ([2, 3, 7, 6], np.array([0, 1, 0], dtype=np.float32)),   # +Y
        ([0, 4, 7, 3], np.array([-1, 0, 0], dtype=np.float32)),  # -X
        ([1, 2, 6, 5], np.array([1, 0, 0], dtype=np.float32))    # +X
    ]

    max_i, max_j, max_k = voxel_space.shape
    filled = (voxel_space > 0)
    
    with open(file_path, 'wb') as f:
        # Write STL header (80 bytes) + placeholder for triangle count
        header = b'Binary STL generated by Volco'.ljust(80, b'\0')
        f.write(header)
        f.write(struct.pack('<I', 0))  # Placeholder for triangle count

        triangle_count = 0
        
        for face_idx, (face_verts, normal) in enumerate(faces):
            # Build surface mask for this face direction
            mask = np.zeros_like(filled, dtype=bool)
            
            if face_idx == 0:  # -Z face
                mask[:, :, :-1] = filled[:, :, :-1] & ~filled[:, :, 1:]
            elif face_idx == 1:  # +Z face
                mask[:, :, 1:] = filled[:, :, 1:] & ~filled[:, :, :-1]
            elif face_idx == 2:  # -Y face
                mask[:, :-1, :] = filled[:, :-1, :] & ~filled[:, 1:, :]
            elif face_idx == 3:  # +Y face
                mask[:, 1:, :] = filled[:, 1:, :] & ~filled[:, :-1, :]
            elif face_idx == 4:  # -X face
                mask[:-1, :, :] = filled[:-1, :, :] & ~filled[1:, :, :]
            elif face_idx == 5:  # +X face
                mask[1:, :, :] = filled[1:, :, :] & ~filled[:-1, :, :]
            
            # Get indices of voxels with visible faces
            indices = np.array(np.where(mask)).T
            
            if indices.size == 0:
                continue
            
            N = indices.shape[0]
            chunk_size = 100000  # Process in large chunks
            
            # Precompute face vertices from cube
            face_vertices = cube[face_verts]  # shape (4, 3)
            
            # Define structured dtype for binary STL triangle (50 bytes)
            # normal(12) + v1(12) + v2(12) + v3(12) + attr(2)
            stl_dtype = np.dtype([
                ('normal', '<f4', (3,)),
                ('v1', '<f4', (3,)),
                ('v2', '<f4', (3,)),
                ('v3', '<f4', (3,)),
                ('attr', '<u2')
            ])
            
            for start in range(0, N, chunk_size):
                end = min(N, start + chunk_size)
                idx_chunk = indices[start:end]
                
                # Compute voxel centers
                centers = idx_chunk.astype(np.float32) * voxel_size
                
                # Build vertices for all voxels in chunk: shape (M, 4, 3)
                verts = centers[:, np.newaxis, :] + face_vertices[np.newaxis, :, :]
                
                # Create two triangles per face
                tri0 = verts[:, [0, 1, 2], :]  # shape (M, 3, 3)
                tri1 = verts[:, [0, 2, 3], :]  # shape (M, 3, 3)
                tris = np.vstack([tri0, tri1])  # shape (2*M, 3, 3)
                
                num_tris = tris.shape[0]
                
                # Create structured array for correct binary layout
                chunk_data = np.zeros(num_tris, dtype=stl_dtype)
                chunk_data['normal'] = normal
                chunk_data['v1'] = tris[:, 0, :]
                chunk_data['v2'] = tris[:, 1, :]
                chunk_data['v3'] = tris[:, 2, :]
                # 'attr' is initialized to 0 by np.zeros
                
                # Write chunk to file
                f.write(chunk_data.tobytes())
                
                triangle_count += num_tris

        # Update triangle count in header
        f.seek(80)
        f.write(struct.pack('<I', triangle_count))

    t1 = time.perf_counter()
    mem1 = tracemalloc.get_traced_memory()
    logger.info(f"[Profile]: Binary STL export complete. Time: {t1-t0:.2f}s, Mem: current={mem1[0]/1e6:.2f}MB, peak={mem1[1]/1e6:.2f}MB, Triangles: {triangle_count}")
    tracemalloc.stop()
    
    return file_path

def generate_and_export_mesh(voxel_space, voxel_size, file_path, binary=True):
    """
    Generate and export a mesh from voxel space using streaming approach.
    Supports both ASCII and binary STL formats.

    Parameters:
    -----------
    voxel_space : numpy.ndarray
        The dense 3D voxel space array.
    voxel_size : float
        The size of each voxel.
    file_path : str
        The path to save the STL file.
    binary : bool
        Whether to export in binary format (True) or ASCII format (False).
        Binary is strongly recommended for performance.

    Returns:
    --------
    str
        The path to the exported STL file.
    """
    if voxel_space is None:
        logger.warning("[Mesh]: No voxel space provided.")
        return None
    
    try:
        if binary:
            logger.info("[Mesh]: Exporting STL in binary format (recommended)")
            format_type = "binary"
            result_path = export_voxel_stl_streaming_binary(voxel_space, voxel_size, file_path)
        else:
            logger.info("[Mesh]: Exporting STL in ASCII format (optimized streaming)")
            format_type = "ASCII"
            # For ASCII, we generate the mesh first (using vectorized method) and then use trimesh's exporter
            # This is faster than our custom streaming ASCII exporter was
            mesh = generate_mesh_from_voxels(voxel_space, voxel_size)
            result_path = export_mesh_to_stl(mesh, file_path, ascii_format=True)
        
        logger.info(f"[Mesh]: STL exported in {format_type} format!")
        return result_path
    except Exception as e:
        logger.error(f"[Mesh]: Failed to export STL: {e}")
        return None

# Sparse voxel space utilities (kept for potential future use)
def dense_to_sparse(voxel_space):
    """
    Convert a dense voxel space (numpy array) to a sparse representation.
    
    Parameters:
    -----------
    voxel_space : numpy.ndarray
        The dense 3D voxel space array.

    Returns:
    --------
    scipy.sparse.coo_matrix
        Sparse representation of the voxel space.
    """
    filled_indices = np.array(np.where(voxel_space > 0)).T
    data = np.ones(len(filled_indices))
    sparse_voxel_space = coo_matrix(
        (data, (filled_indices[:, 0], filled_indices[:, 1] * voxel_space.shape[2] + filled_indices[:, 2])),
        shape=(voxel_space.shape[0], voxel_space.shape[1] * voxel_space.shape[2])
    )
    return sparse_voxel_space
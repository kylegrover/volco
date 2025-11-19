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

def create_box_representation(voxel_space, voxel_size):
    """
    Create a box representation mesh from a voxel space.
    This is a custom implementation that doesn't rely on any trimesh functions
    that might use the `ptp` method, making it compatible with NumPy 2.x.
    
    This optimized version only creates triangles for voxel faces that are either:
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
        A mesh containing only visible faces of filled voxels
    """
    logger.info("[Profile]: Starting mesh generation (box representation)")
    tracemalloc.start()
    t0 = time.perf_counter()
    
    # Find filled voxels
    filled = (voxel_space > 0)
    if not np.any(filled):
        tracemalloc.stop()
        return trimesh.Scene()

    max_i, max_j, max_k = voxel_space.shape
    unit_cube_vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])
    
    # Define faces with their triangles and direction
    face_definitions = [
        ([[0, 2, 1], [0, 3, 2]], (0, 0, -1)),  # -Z face
        ([[4, 5, 6], [4, 6, 7]], (0, 0, 1)),   # +Z face
        ([[0, 1, 5], [0, 5, 4]], (0, -1, 0)),  # -Y face
        ([[2, 3, 7], [2, 7, 6]], (0, 1, 0)),   # +Y face
        ([[0, 4, 7], [0, 7, 3]], (-1, 0, 0)),  # -X face
        ([[1, 2, 6], [1, 6, 5]], (1, 0, 0))    # +X face
    ]

    all_vertices = []
    all_faces = []
    vertex_count = 0
    
    # Process each face direction
    for face_idx, (face_triangles, (di, dj, dk)) in enumerate(face_definitions):
        # Create mask for visible faces in this direction
        mask = np.zeros_like(filled, dtype=bool)
        
        if di != 0:
            if di > 0:  # +X direction
                mask[:-1, :, :] = filled[:-1, :, :] & ~filled[1:, :, :]
            else:  # -X direction
                mask[1:, :, :] = filled[1:, :, :] & ~filled[:-1, :, :]
        elif dj != 0:
            if dj > 0:  # +Y direction
                mask[:, :-1, :] = filled[:, :-1, :] & ~filled[:, 1:, :]
            else:  # -Y direction
                mask[:, 1:, :] = filled[:, 1:, :] & ~filled[:, :-1, :]
        elif dk != 0:
            if dk > 0:  # +Z direction
                mask[:, :, :-1] = filled[:, :, :-1] & ~filled[:, :, 1:]
            else:  # -Z direction
                mask[:, :, 1:] = filled[:, :, 1:] & ~filled[:, :, :-1]
        
        # Get indices of voxels with visible faces
        face_voxels = np.array(np.where(mask)).T
        
        # Process each voxel with a visible face
        for idx in face_voxels:
            i, j, k = idx
            center = np.array([i, j, k], dtype=np.float32) * voxel_size
            voxel_vertices = unit_cube_vertices * voxel_size + center
            
            v_start = vertex_count
            all_vertices.extend(voxel_vertices)
            vertex_count += 8
            
            for triangle in face_triangles:
                all_faces.append([v_start + t for t in triangle])
    
    t1 = time.perf_counter()
    mem1 = tracemalloc.get_traced_memory()
    logger.info(f"[Profile]: Mesh generation complete. Time: {t1-t0:.2f}s, Mem: current={mem1[0]/1e6:.2f}MB, peak={mem1[1]/1e6:.2f}MB")
    tracemalloc.stop()
    
    if all_vertices and all_faces:
        vertices_array = np.array(all_vertices, dtype=np.float32)
        faces_array = np.array(all_faces, dtype=np.int32)
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
        voxels = create_box_representation(voxel_space, voxel_size)
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

def export_voxel_stl_streaming_ascii(voxel_space, voxel_size, file_path):
    """
    Export a voxel space as an ASCII STL file using streaming (low-memory) logic.
    Super-optimized using NumPy vectorized string operations to avoid Python loops.
    
    Parameters:
    -----------
    voxel_space : numpy.ndarray
        The 3D voxel space array
    voxel_size : float
        The size of each voxel
    file_path : str
        The path to save the ASCII STL file
        
    Returns:
    --------
    str
        The path to the exported STL file
    """
    logger.info("[Profile]: Starting vectorized streaming ASCII STL export (NumPy optimized)")
    tracemalloc.start()
    t0 = time.perf_counter()
    
    # Cube vertices (relative to center)
    cube = voxel_size * np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ], dtype=np.float32)
    
    # Face definitions: (vertex indices, normal vector)
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
    triangle_count = 0

    with open(file_path, 'w', buffering=8*1024*1024) as f:  # 8MB buffer
        f.write("solid voxel\n")
        
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
            chunk_size = 20000  # Process in chunks to manage memory
            
            # Precompute face vertices from cube
            face_vertices = cube[face_verts]  # shape (4, 3)
            
            # Pre-format normal string
            nx, ny, nz = normal
            normal_str = f"  facet normal {nx:.6g} {ny:.6g} {nz:.6g}\n"
            outer_loop_str = "    outer loop\n"
            end_loop_str = "    endloop\n  endfacet\n"
            
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
                triangle_count += num_tris
                
                # --- Vectorized String Formatting ---
                # Format vertices to strings using numpy
                # We flatten to (num_tris * 3, 3) to format all vertices at once
                all_verts = tris.reshape(-1, 3)
                
                # Format X, Y, Z columns
                # %.6g is standard for STL
                xs = np.char.mod('%.6g', all_verts[:, 0])
                ys = np.char.mod('%.6g', all_verts[:, 1])
                zs = np.char.mod('%.6g', all_verts[:, 2])
                
                # Combine into "vertex X Y Z\n"
                # We use np.core.defchararray.add which is vectorized string concatenation
                v_str = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
                    np.array(["      vertex "]*len(xs)), xs), 
                    np.array([" "]*len(xs))), ys), 
                    np.array([" "]*len(xs))), zs)
                v_str = np.char.add(v_str, np.array(["\n"]*len(xs)))
                
                # Reshape back to (num_tris, 3)
                v_str = v_str.reshape(num_tris, 3)
                
                # Combine all parts
                # Each triangle: normal + outer + v1 + v2 + v3 + endloop
                
                # Create arrays for static parts
                normals = np.full(num_tris, normal_str)
                outers = np.full(num_tris, outer_loop_str)
                endloops = np.full(num_tris, end_loop_str)
                
                # Concatenate everything
                # This creates one huge string per triangle
                tri_strings = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
                    normals, outers), 
                    v_str[:, 0]), 
                    v_str[:, 1]), 
                    v_str[:, 2]), 
                    endloops)
                
                # Join all triangle strings and write
                f.write("".join(tri_strings))

        f.write("endsolid voxel\n")

    t1 = time.perf_counter()
    mem1 = tracemalloc.get_traced_memory()
    logger.info(f"[Profile]: ASCII STL export complete. Time: {t1-t0:.2f}s, Mem: current={mem1[0]/1e6:.2f}MB, peak={mem1[1]/1e6:.2f}MB, Triangles: {triangle_count}")
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
            result_path = export_voxel_stl_streaming_ascii(voxel_space, voxel_size, file_path)
        
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
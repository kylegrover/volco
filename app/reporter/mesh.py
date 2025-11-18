import logging
import trimesh
import numpy as np
from skimage import measure

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
    # Find filled voxels
    filled = (voxel_space > 0)
    if not np.any(filled):
        return trimesh.Scene()

    max_i, max_j, max_k = voxel_space.shape
    unit_cube_vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])
    face_definitions = [
        ([[0, 2, 1], [0, 3, 2]], (0, 0, -1)),
        ([[4, 5, 6], [4, 6, 7]], (0, 0, 1)),
        ([[0, 1, 5], [0, 5, 4]], (0, -1, 0)),
        ([[2, 3, 7], [2, 7, 6]], (0, 1, 0)),
        ([[0, 4, 7], [0, 7, 3]], (-1, 0, 0)),
        ([[1, 2, 6], [1, 6, 5]], (1, 0, 0))
    ]

    # Vectorized neighbor check for all filled voxels
    filled_indices = np.array(np.where(filled)).T
    all_vertices = []
    all_faces = []
    vertex_count = 0
    for face_idx, (face_triangles, (di, dj, dk)) in enumerate(face_definitions):
        # Use numpy slicing for neighbor checks
        if di != 0:
            if di > 0:
                mask = np.zeros_like(filled)
                mask[:-1, :, :] = filled[:-1, :, :] & (~filled[1:, :, :])
            else:
                mask = np.zeros_like(filled)
                mask[1:, :, :] = filled[1:, :, :] & (~filled[:-1, :, :])
        elif dj != 0:
            if dj > 0:
                mask = np.zeros_like(filled)
                mask[:, :-1, :] = filled[:, :-1, :] & (~filled[:, 1:, :])
            else:
                mask = np.zeros_like(filled)
                mask[:, 1:, :] = filled[:, 1:, :] & (~filled[:, :-1, :])
        elif dk != 0:
            if dk > 0:
                mask = np.zeros_like(filled)
                mask[:, :, :-1] = filled[:, :, :-1] & (~filled[:, :, 1:])
            else:
                mask = np.zeros_like(filled)
                mask[:, :, 1:] = filled[:, :, 1:] & (~filled[:, :, :-1])
        face_voxels = np.array(np.where(mask)).T
        for idx in face_voxels:
            i, j, k = idx
            center = np.array([i, j, k]) * voxel_size
            voxel_vertices = unit_cube_vertices * voxel_size + center
            v_start = vertex_count
            all_vertices.extend(voxel_vertices)
            vertex_count += 8
            for triangle in face_triangles:
                all_faces.append([v_start + t for t in triangle])
    if all_vertices and all_faces:
        vertices_array = np.array(all_vertices)
        faces_array = np.array(all_faces)
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
        # Use our optimized box representation function
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
                
                # Scale vertices by voxel size
                verts = verts * voxel_size
                
                # Create mesh from vertices and faces
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
        The mesh to export (can be a Trimesh object or a Scene object from as_boxes())
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
        
        # Set export options based on format
        export_options = {'file_type': 'stl_ascii' if ascii_format else 'stl'}
        
        # Use the generic export function which works for both Scene and Trimesh objects
        trimesh.exchange.export.export_mesh(mesh, file_path, **export_options)
        # mesh.export(file_path, **export_options)
            
        logger.info(f"[Mesh]: STL exported in {format_type} format!")
        return file_path
    except Exception as e:
        logger.error(f"[Mesh]: Failed to export STL: {e}")
        return None
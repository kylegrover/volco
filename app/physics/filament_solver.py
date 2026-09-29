import numpy as np
import logging

logger = logging.getLogger(__name__)

class FilamentNode:
    def __init__(self, position, velocity, mass, temperature, radius, rest_length=None):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.mass = mass
        self.temperature = temperature
        self.radius = radius
        self.fixed = False  # If True, this node has solidified/collided and is part of the grid
        self.rest_length = rest_length # Distance to previous node

class FilamentSolver:
    def __init__(self, config, physics_solver):
        self.config = config
        self.physics_solver = physics_solver # Reference to the Eulerian grid solver for collision
        self.nodes = [] # List of FilamentNode
        
        # Physics constants
        self.k = self.config.filament_spring_k
        self.damping = self.config.filament_damping
        self.gravity = np.array([0, 0, -9.81 * self.config.gravity_factor], dtype=np.float32)
        # self.segment_length is now dynamic per node

    def add_node(self, position, velocity, mass, temperature, radius):
        """
        Adds a new node to the active filament chain (at the nozzle).
        """
        rest_length = self.config.filament_segment_length
        
        # If we have a previous node, calculate the distance to it as the rest length
        # This assumes the filament is deposited at its natural length (no initial strain)
        if self.nodes:
            prev_node = self.nodes[-1]
            dist = np.linalg.norm(np.array(position) - prev_node.position)
            if dist > 0:
                rest_length = dist
        
        new_node = FilamentNode(position, velocity, mass, temperature, radius, rest_length)
        self.nodes.append(new_node)

    def step(self, dt):
        """
        Evolves the filament chain.
        Returns a list of segments (p1, p2, radius, mass, temp) that have 'frozen' and should be transferred to the grid.
        """
        if not self.nodes:
            return []

        # Stability: Sub-step if dt is too large for the current forces
        # With gravity_factor=100 (~1000 m/s^2), we need small steps.
        # Target dt around 0.0005s (0.5ms)
        max_substep = 0.0005
        n_substeps = int(np.ceil(dt / max_substep))
        sub_dt = dt / n_substeps
        
        for _ in range(n_substeps):
            # 1. Apply Forces (Gravity + Springs)
            forces = [np.zeros(3, dtype=np.float32) for _ in self.nodes]
            
            # Gravity
            for i, node in enumerate(self.nodes):
                if not node.fixed:
                    forces[i] += self.gravity * node.mass

            # Springs (Chain tension)
            # We iterate through segments (i, i+1)
            for i in range(len(self.nodes) - 1):
                n1 = self.nodes[i]
                n2 = self.nodes[i+1]
                
                if n1.fixed and n2.fixed:
                    continue

                # Vector from n1 to n2
                delta = n2.position - n1.position
                dist = np.linalg.norm(delta)
                
                if dist < 1e-6:
                    continue
                    
                direction = delta / dist
                
                # Use the rest length stored in the second node (n2)
                # n2.rest_length is the distance to n1
                rest_len = n2.rest_length if n2.rest_length is not None else self.config.filament_segment_length
                
                # Hooke's Law: F = -k * (x - L)
                stretch = dist - rest_len
                
                # Compression handling: If compressed (stretch < 0), we used to reduce stiffness
                # But for stacking (vertical printing), we need the filament to support its own weight.
                # So we keep full stiffness or even increase it?
                # Let's keep it symmetric for now.
                k_eff = self.k
                # if stretch < 0:
                #    k_eff *= 0.01 # 1% stiffness in compression
                
                # Safety: Cap the stretch to avoid explosive forces
                max_stretch = 1.5 * rest_len
                if abs(stretch) > max_stretch:
                    stretch = np.sign(stretch) * max_stretch

                force_mag = k_eff * stretch
                
                force_vec = force_mag * direction
                
                # Apply equal and opposite forces
                if not n1.fixed:
                    forces[i] += force_vec
                if not n2.fixed:
                    forces[i+1] -= force_vec

            # 2. Integrate (Euler)
            for i, node in enumerate(self.nodes):
                if node.fixed:
                    continue
                
                # Acceleration
                acc = forces[i] / node.mass
                
                # Velocity update
                node.velocity += acc * sub_dt
                node.velocity *= (1.0 - self.damping) # Damping
                
                # Safety: Clamp velocity
                speed = np.linalg.norm(node.velocity)
                if speed > 500.0: # 500 mm/s limit
                    node.velocity = (node.velocity / speed) * 500.0

                # Position update
                node.position += node.velocity * sub_dt
                
                # Safety: Check for NaN/Inf
                if not np.isfinite(node.position).all():
                    logger.warning(f"Node {i} position became non-finite: {node.position}. Resetting to previous valid state or freezing.")
                    node.fixed = True
                    continue

                # 3. Collision Detection
                # Check against Z=0 floor
                if node.position[2] <= node.radius:
                    node.position[2] = node.radius
                    node.velocity[:] = 0
                    node.fixed = True
                    node.temperature = 0 # Reset temperature upon collision
                    continue
                
                # Check against Voxel Grid
                # We check a box of voxels around the node to ensure we don't miss collisions
                # Radius in voxels
                r_vox = int(node.radius / self.config.voxel_size) + 1
                
                cx = int(node.position[0] / self.config.voxel_size)
                cy = int(node.position[1] / self.config.voxel_size)
                cz = int(node.position[2] / self.config.voxel_size)
                
                # Check bounds
                if (cx + r_vox < 0 or cx - r_vox >= self.physics_solver.shape[0] or
                    cy + r_vox < 0 or cy - r_vox >= self.physics_solver.shape[1] or
                    cz + r_vox < 0 or cz - r_vox >= self.physics_solver.shape[2]):
                    # Out of bounds, can't collide with grid (but might be flying off)
                    continue

                # Iterate over a small neighborhood
                collision_detected = False
                
                # Optimization: Check center first
                if (0 <= cx < self.physics_solver.shape[0] and
                    0 <= cy < self.physics_solver.shape[1] and
                    0 <= cz < self.physics_solver.shape[2]):
                    if self.physics_solver.mass[cx, cy, cz] > (self.config.voxel_size**3 * 0.5):
                        collision_detected = True
                
                if not collision_detected:
                    # Check neighborhood
                    # We only check if we are close to the surface?
                    # Full box check
                    x_min = max(0, cx - r_vox)
                    x_max = min(self.physics_solver.shape[0], cx + r_vox + 1)
                    y_min = max(0, cy - r_vox)
                    y_max = min(self.physics_solver.shape[1], cy + r_vox + 1)
                    z_min = max(0, cz - r_vox)
                    z_max = min(self.physics_solver.shape[2], cz + r_vox + 1)
                    
                    # Extract mass slice
                    mass_slice = self.physics_solver.mass[x_min:x_max, y_min:y_max, z_min:z_max]
                    if np.any(mass_slice > (self.config.voxel_size**3 * 0.5)):
                        collision_detected = True
                
                if collision_detected:
                    # Check if we should ignore this collision (e.g. overlapping with the anchor node)
                    ignore = False
                    if i > 0 and self.nodes[i-1].fixed:
                        # Calculate distance to the fixed neighbor
                        dist = np.linalg.norm(node.position - self.nodes[i-1].position)
                        # If we are within the combined radius (plus a small margin), ignore grid collision
                        # This allows the filament to "emerge" from the solidified structure
                        # Increased margin to 1.5 to prevent premature freezing on the anchor's voxel footprint
                        if dist < (node.radius + self.nodes[i-1].radius) * 1.5:
                            ignore = True

                    if not ignore:
                        # Collision!
                        node.velocity[:] = 0
                        node.fixed = True
                        node.temperature = 0 # Reset temperature upon collision

        # 4. Manage Lists & Generate Segments
        frozen_segments = []
        
        # Find the index of the last fixed node (the anchor)
        last_fixed_idx = -1
        for i in range(len(self.nodes) - 1, -1, -1):
            if self.nodes[i].fixed:
                last_fixed_idx = i
                break
        
        if last_fixed_idx != -1:
            # We have at least one fixed node.
            # The chain starts at last_fixed_idx.
            
            # Force freeze nodes that are being discarded (the segment behind the new anchor)
            for i in range(last_fixed_idx):
                node = self.nodes[i]
                if not node.fixed:
                    node.fixed = True
                    node.velocity[:] = 0
            
            # Generate segments for the frozen chain
            # We iterate from 1 to last_fixed_idx
            # Segment connects i-1 to i
            for i in range(1, last_fixed_idx + 1):
                n1 = self.nodes[i-1]
                n2 = self.nodes[i]
                # Segment properties from n2 (mass, temp, radius)
                frozen_segments.append((n1.position, n2.position, n2.radius, n2.mass, n2.temperature))

            self.nodes = self.nodes[last_fixed_idx:]
        
        return frozen_segments
    
    def shift_nodes(self, offset_vector):
        """
        Shifts all nodes by the given vector.
        Used when the voxel grid expands in the negative direction.
        """
        offset = np.array(offset_vector, dtype=np.float32)
        for node in self.nodes:
            node.position += offset

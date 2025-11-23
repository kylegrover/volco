import numpy as np
import logging

logger = logging.getLogger(__name__)

class PhysicsSolver:
    def __init__(self, shape, voxel_size, config):
        self.shape = shape
        self.voxel_size = voxel_size
        self.config = config
        
        # Physics Grids
        # Mass: 0.0 to 1.0 (or >1.0 for compression) representing material amount
        self.mass = np.zeros(shape, dtype=np.float32)
        
        # Temperature: In Celsius
        self.temperature = np.full(shape, config.ambient_temperature, dtype=np.float32)
        
        # Velocity: (3, x, y, z) - Optional, for more complex momentum-based flow
        # For now, we might stick to diffusion-based flow for simplicity and stability
        
        # Active region tracking to optimize updates
        self.active_bounds = {
            'x_min': 0, 'x_max': 0,
            'y_min': 0, 'y_max': 0,
            'z_min': 0, 'z_max': 0
        }
        self.has_active_material = False

    def add_material(self, center, radius, amount, temperature):
        """
        Injects hot material into the grid.
        """
        # Convert center to grid indices
        cx, cy, cz = center
        idx_x = int(cx / self.voxel_size)
        idx_y = int(cy / self.voxel_size)
        idx_z = int(cz / self.voxel_size)
        
        # Determine bounding box for injection
        # We use a slightly larger box to ensure we catch the sphere
        r_idx = int(radius / self.voxel_size) + 1
        
        x_min = max(0, idx_x - r_idx)
        x_max = min(self.shape[0], idx_x + r_idx + 1)
        y_min = max(0, idx_y - r_idx)
        y_max = min(self.shape[1], idx_y + r_idx + 1)
        z_min = max(0, idx_z - r_idx)
        z_max = min(self.shape[2], idx_z + r_idx + 1)
        
        # Update active bounds
        if not self.has_active_material:
            self.active_bounds = {
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'z_min': z_min, 'z_max': z_max
            }
            self.has_active_material = True
        else:
            self.active_bounds['x_min'] = min(self.active_bounds['x_min'], x_min)
            self.active_bounds['x_max'] = max(self.active_bounds['x_max'], x_max)
            self.active_bounds['y_min'] = min(self.active_bounds['y_min'], y_min)
            self.active_bounds['y_max'] = max(self.active_bounds['y_max'], y_max)
            self.active_bounds['z_min'] = min(self.active_bounds['z_min'], z_min)
            self.active_bounds['z_max'] = max(self.active_bounds['z_max'], z_max)

        # Create coordinate grids for the bounding box to check distances
        # We use local coordinates relative to the box for efficiency
        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        z_range = np.arange(z_min, z_max)
        
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # Physical coordinates of voxel centers
        pX = (X + 0.5) * self.voxel_size
        pY = (Y + 0.5) * self.voxel_size
        pZ = (Z + 0.5) * self.voxel_size
        
        # Squared distance from injection center
        dist_sq = (pX - cx)**2 + (pY - cy)**2 + (pZ - cz)**2
        
        # Select voxels within radius
        mask = dist_sq <= (radius**2)
        count = np.sum(mask)
        
        # Fallback: If radius is too small, pick the closest voxel
        if count == 0:
            # Find index of minimum distance
            min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
            # Create a mask with just that one voxel
            mask = np.zeros_like(mask, dtype=bool)
            mask[min_idx] = True
            count = 1

        # Distribute mass
        mass_per_voxel = amount / count
        
        # Extract slices
        mass_slice = self.mass[x_min:x_max, y_min:y_max, z_min:z_max]
        temp_slice = self.temperature[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Update Mass
        # We only update where mask is True
        current_mass = mass_slice[mask]
        mass_slice[mask] += mass_per_voxel
        
        # Update Temperature (Weighted average)
        # T_new = (M_old * T_old + M_added * T_added) / (M_old + M_added)
        old_temp = temp_slice[mask]
        total_mass = current_mass + mass_per_voxel # current_mass is the OLD mass before addition? No, mass_slice is a view.
        # Wait, I already added mass_per_voxel to mass_slice[mask] above.
        # So 'mass_slice[mask]' now contains (M_old + M_added).
        # 'current_mass' was a copy of M_old (because fancy indexing returns a copy).
        
        # Correct logic:
        # current_mass = mass_slice[mask] (This is a copy of data before modification if done before)
        # But I did: current_mass = mass_slice[mask]; mass_slice[mask] += ...
        # So current_mass holds the OLD mass.
        
        new_temp_values = (current_mass * old_temp + mass_per_voxel * temperature) / (current_mass + mass_per_voxel)
        temp_slice[mask] = new_temp_values

    def step(self, dt):
        """
        Evolves the physics state by one time step.
        """
        if not self.has_active_material:
            return

        # Extract active region slices (with 1 voxel padding for stencil)
        s_x = slice(max(0, self.active_bounds['x_min'] - 1), min(self.shape[0], self.active_bounds['x_max'] + 1))
        s_y = slice(max(0, self.active_bounds['y_min'] - 1), min(self.shape[1], self.active_bounds['y_max'] + 1))
        s_z = slice(max(0, self.active_bounds['z_min'] - 1), min(self.shape[2], self.active_bounds['z_max'] + 1))
        
        # Get views of the active region
        mass_view = self.mass[s_x, s_y, s_z]
        temp_view = self.temperature[s_x, s_y, s_z]
        
        # 1. Cooling
        # Newton's Law of Cooling: dT/dt = -k * (T - T_env)
        cooling_rate = self.config.cooling_rate
        temp_view -= cooling_rate * (temp_view - self.config.ambient_temperature) * dt
        
        # 2. Viscosity Calculation
        # Simple model: High viscosity if T < T_solid, Low if T > T_melt
        # Using inverse viscosity (mobility) for flux calculation
        mobility = np.zeros_like(temp_view)
        melt_mask = temp_view > self.config.solidification_temp
        mobility[melt_mask] = 1.0 / self.config.viscosity_molten
        mobility[~melt_mask] = 1.0 / self.config.viscosity_solid
        
        # 3. Mass Flow (Simplified Diffusion + Gravity)
        # Flux = -Mobility * (Gradient(Pressure) - Gravity)
        # Pressure ~ Mass (for incompressible-ish fluid)
        
        # We can use a simple stencil:
        # Flow to neighbor = Mobility_interface * (Mass_self - Mass_neighbor + Gravity_bias)
        
        # This part requires careful implementation to conserve mass and be stable.
        # For this prototype, we'll just do a simple gravity settling for molten material.
        
        # Vertical flow (Gravity)
        # Move mass from z to z-1 if z is molten and z-1 has space
        # This is a cellular automata rule
        
        # Vectorized "Sandpile" / Flow logic
        # Calculate potential flow downwards
        # Only flow if mobility is high enough
        
        # flow_mask = mobility > (1.0 / (self.config.viscosity_solid * 2)) # Threshold
        
        # Simple gravity:
        # If cell is molten and cell below is empty(er), move mass down.
        # We need to be careful with array slicing for z and z-1
        
        # Slices for vertical flow
        # upper: z=1 to end
        # lower: z=0 to end-1
        upper_mass = mass_view[:, :, 1:]
        lower_mass = mass_view[:, :, :-1]
        upper_mobility = mobility[:, :, 1:]
        
        # Flow amount proportional to mass, mobility, and gravity
        # Only flow if upper has mass
        # Flow rate factor
        flow_rate = self.config.gravity_factor * dt
        
        # Calculate potential flow
        # We want to move mass from upper to lower
        # But don't overfill lower (optional, but good for stability)
        # And don't take more than upper has
        
        # Simple flow: amount = upper_mass * mobility * flow_rate
        flow_amount = upper_mass * upper_mobility * flow_rate
        
        # Clamp flow to available mass
        flow_amount = np.minimum(flow_amount, upper_mass)
        
        # Apply flow
        # Subtract from upper
        mass_view[:, :, 1:] -= flow_amount
        # Add to lower
        mass_view[:, :, :-1] += flow_amount
        
        # Advect Temperature (Simplified)
        # Heat moves with mass.
        # T_lower_new = (M_lower * T_lower + M_flow * T_upper) / (M_lower + M_flow)
        
        # We need to be careful with division by zero
        # Calculate new mass for lower layer
        new_lower_mass = lower_mass + flow_amount
        
        # Mask where flow occurred
        flow_mask = flow_amount > 1e-6
        
        if np.any(flow_mask):
            # Only update temperature where flow happened
            # T_lower = temp_view[:, :, :-1]
            # T_upper = temp_view[:, :, 1:]
            
            # Weighted average
            # temp_view[:, :, :-1][flow_mask] = (
            #     lower_mass[flow_mask] * temp_view[:, :, :-1][flow_mask] + 
            #     flow_amount[flow_mask] * temp_view[:, :, 1:][flow_mask]
            # ) / new_lower_mass[flow_mask]
            
            # Vectorized update
            t_lower = temp_view[:, :, :-1]
            t_upper = temp_view[:, :, 1:]
            
            # Calculate heat energy
            energy_lower = lower_mass * t_lower
            energy_flow = flow_amount * t_upper
            
            # Update temperature
            # Avoid division by zero by using a safe mask
            safe_mask = new_lower_mass > 1e-6
            
            # Update only where we have mass
            t_lower[safe_mask] = (energy_lower[safe_mask] + energy_flow[safe_mask]) / new_lower_mass[safe_mask]
        
        # Update active bounds based on where mass moved
        # (If mass moved outside current bounds, expand them)
        # Since we only moved z to z-1, we might need to expand z_min downwards if we were at the edge
        # But active_bounds includes padding, so we should be okay for one step.
        # However, if we flow into z_min, we should decrement z_min for next time.
        
        # Check if mass flowed into the bottom slice of the active region
        if np.any(flow_amount[:, :, 0] > 0):
             self.active_bounds['z_min'] = max(0, self.active_bounds['z_min'] - 1)

        # 4. Horizontal Flow (Spreading)
        # Driven by Mass Difference (Pressure)
        # Flow X
        # left: x=0 to end-1, right: x=1 to end
        # Flux from left to right: (mass_left - mass_right)
        
        spread_rate = self.config.spread_factor * dt
        
        # X-Direction
        mass_left = mass_view[:-1, :, :]
        mass_right = mass_view[1:, :, :]
        mob_left = mobility[:-1, :, :]
        mob_right = mobility[1:, :, :]
        
        # Average mobility at interface
        mob_interface_x = (mob_left + mob_right) * 0.5
        
        # Flux > 0 means flow Left -> Right
        flux_x = (mass_left - mass_right) * mob_interface_x * spread_rate
        
        # Limit flux to avoid instability (don't move more than the difference/2)
        # If flux is too high, it oscillates.
        # Max flow is (diff / 2)
        max_flux_x = (mass_left - mass_right) * 0.5
        
        # We need to clamp magnitude but keep sign
        # This is a bit tricky vectorized.
        # Let's just use a small spread_rate to be safe for now.
        
        # Apply X Flux
        mass_view[:-1, :, :] -= flux_x
        mass_view[1:, :, :] += flux_x
        
        # Y-Direction
        mass_front = mass_view[:, :-1, :]
        mass_back = mass_view[:, 1:, :]
        mob_front = mobility[:, :-1, :]
        mob_back = mobility[:, 1:, :]
        
        mob_interface_y = (mob_front + mob_back) * 0.5
        
        flux_y = (mass_front - mass_back) * mob_interface_y * spread_rate
        
        # Apply Y Flux
        mass_view[:, :-1, :] -= flux_y
        mass_view[:, 1:, :] += flux_y
        
        # Note: We are not advecting temperature for horizontal flow yet to save perf.
        # It will diffuse naturally via the mass mixing if we did it, but for now
        # we assume local thermal equilibrium or just let vertical flow dominate heat transport.

        # Update views back to main array (if views were copies, but numpy slices are views)


    def get_voxel_grid(self):
        """
        Returns the binary voxel grid for compatibility with existing code.
        """
        # Threshold should be based on voxel volume.
        # self.mass stores volume of material in mm^3.
        # A voxel is considered filled if it contains > 50% of its volume.
        voxel_vol = self.voxel_size ** 3
        threshold = voxel_vol * 0.5
        
        max_mass = np.max(self.mass)
        logger.info(f"Physics Grid Stats: Max Mass/Voxel={max_mass:.6f}, Threshold={threshold:.6f}, Total Mass={np.sum(self.mass):.6f}")
        
        return (self.mass > threshold).astype(np.int8)

    def expand(self, new_shape, offset):
        """
        Expands the grid to a new shape, shifting the existing content by offset.
        offset is (x_off, y_off, z_off)
        """
        new_mass = np.zeros(new_shape, dtype=np.float32)
        new_temp = np.full(new_shape, self.config.ambient_temperature, dtype=np.float32)
        
        ox, oy, oz = offset
        sx, sy, sz = self.shape
        
        new_mass[ox:ox+sx, oy:oy+sy, oz:oz+sz] = self.mass
        new_temp[ox:ox+sx, oy:oy+sy, oz:oz+sz] = self.temperature
        
        self.mass = new_mass
        self.temperature = new_temp
        self.shape = new_shape
        
        # Update active bounds
        if self.has_active_material:
            self.active_bounds['x_min'] += ox
            self.active_bounds['x_max'] += ox
            self.active_bounds['y_min'] += oy
            self.active_bounds['y_max'] += oy
            self.active_bounds['z_min'] += oz
            self.active_bounds['z_max'] += oz

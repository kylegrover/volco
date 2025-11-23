import unittest
import numpy as np
from app.physics.solver import PhysicsSolver

class MockConfig:
    def __init__(self):
        self.ambient_temperature = 25.0
        self.solidification_temp = 200.0
        self.cooling_rate = 1.0
        self.viscosity_molten = 1.0
        self.viscosity_solid = 1000.0
        self.gravity_factor = 0.1
        self.spread_factor = 1.0

class TestPhysicsSolver(unittest.TestCase):
    def test_initialization(self):
        config = MockConfig()
        shape = (10, 10, 10)
        solver = PhysicsSolver(shape, 1.0, config)
        
        self.assertEqual(solver.mass.shape, shape)
        self.assertEqual(solver.temperature.shape, shape)
        self.assertTrue(np.all(solver.temperature == 25.0))

    def test_add_material(self):
        config = MockConfig()
        shape = (10, 10, 10)
        solver = PhysicsSolver(shape, 1.0, config)
        
        center = (5.0, 5.0, 5.0)
        radius = 1.0
        amount = 10.0
        temp = 220.0
        
        solver.add_material(center, radius, amount, temp)
        
        self.assertTrue(np.sum(solver.mass) > 0)
        self.assertTrue(np.max(solver.temperature) > 25.0)
        self.assertTrue(solver.has_active_material)

    def test_cooling(self):
        config = MockConfig()
        shape = (10, 10, 10)
        solver = PhysicsSolver(shape, 1.0, config)
        
        center = (5.0, 5.0, 5.0)
        solver.add_material(center, 1.0, 10.0, 225.0)
        
        initial_max_temp = np.max(solver.temperature)
        solver.step(0.1)
        final_max_temp = np.max(solver.temperature)
        
        self.assertTrue(final_max_temp < initial_max_temp)

    def test_expand(self):
        config = MockConfig()
        shape = (10, 10, 10)
        solver = PhysicsSolver(shape, 1.0, config)
        
        solver.add_material((5,5,5), 1.0, 10.0, 220.0)
        initial_mass = np.sum(solver.mass)
        
        new_shape = (10, 10, 20)
        solver.expand(new_shape, (0, 0, 0))
        
        self.assertEqual(solver.mass.shape, new_shape)
        self.assertAlmostEqual(float(np.sum(solver.mass)), float(initial_mass), places=5)

    def test_gravity_flow(self):
        config = MockConfig()
        config.gravity_factor = 1.0
        config.viscosity_molten = 1.0
        shape = (10, 10, 10)
        solver = PhysicsSolver(shape, 1.0, config)
        
        # Add material at z=5
        center = (5.0, 5.0, 5.0)
        solver.add_material(center, 0.5, 10.0, 250.0) # Hot!
        
        # Check mass at z=5
        z5_mass_initial = np.sum(solver.mass[:, :, 5])
        z3_mass_initial = np.sum(solver.mass[:, :, 3])
        
        # Step physics
        solver.step(0.1)
        
        # Check mass moved down
        # z5 might be constant if inflow=outflow, but z3 (below the block) should gain mass
        z3_mass_final = np.sum(solver.mass[:, :, 3])
        
        self.assertTrue(z3_mass_final > z3_mass_initial)
        
        # Check conservation
        total_mass_initial = np.sum(solver.mass)
        total_mass_final = np.sum(solver.mass)
        self.assertAlmostEqual(float(total_mass_final), float(total_mass_initial), places=5)

    def test_spreading(self):
        config = MockConfig()
        config.spread_factor = 1.0
        config.viscosity_molten = 1.0
        shape = (10, 10, 10)
        solver = PhysicsSolver(shape, 1.0, config)
        
        # Add a tall stack at x=5, y=5
        center = (5.0, 5.0, 5.0)
        solver.add_material(center, 0.5, 10.0, 250.0)
        
        # Check neighbors (further out) are empty
        # Material is in 4,5,6. Check 3 and 7.
        x3_mass_initial = np.sum(solver.mass[3, 5, :])
        x7_mass_initial = np.sum(solver.mass[7, 5, :])
        self.assertEqual(x3_mass_initial, 0)
        self.assertEqual(x7_mass_initial, 0)
        
        # Step physics
        solver.step(0.1)
        
        # Check mass spread to neighbors
        x3_mass_final = np.sum(solver.mass[3, 5, :])
        x7_mass_final = np.sum(solver.mass[7, 5, :])
        
        self.assertTrue(x3_mass_final > 0)
        self.assertTrue(x7_mass_final > 0)

if __name__ == '__main__':
    unittest.main()

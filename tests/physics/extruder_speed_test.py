import math
from types import SimpleNamespace

import pytest

from app.physics.acceleration.extruder_speed import ExtruderSpeed


class TestExtruderSpeed:
    def test_should_calculate_the_speed_profile(self):
        threshold_speed = 50
        acceleration = 100.0

        extrusion_length = 100

        extruder_speed = ExtruderSpeed(
            volume=extrusion_length * math.pi * 1.75**2 / 4,
            threshold_speed=threshold_speed,
            acceleration=acceleration,
            total_time=1.25,
            printer=SimpleNamespace(feedstock_filament_diameter=1.75),
        )
        extruder_speed.calculate_displacements()

        assert extruder_speed.target_speed == pytest.approx(100.0)

        assert abs(extruder_speed.speed_profile.displacements[-1] - 100.0) < 1e-3

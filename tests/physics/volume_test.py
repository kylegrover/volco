import pytest
from types import SimpleNamespace

from app.physics.acceleration.extruder_speed import ExtruderSpeed
from app.physics.acceleration.nozzle_speed import NozzleSpeed
from app.physics.acceleration.volume import AccelerationVolume

import math


class TestVolume:
    def test_should_calculate_the_speed_profile(self):
        target_speed = 10
        threshold_speed = 40
        acceleration = 1200.0

        filament_length = 100.0

        extrusion_length = 10.0

        nozzle_speed = NozzleSpeed(
            filament_length=filament_length,
            target_speed=target_speed,
            threshold_speed=threshold_speed,
            acceleration=acceleration,
        )
        nozzle_speed.calculate_displacements()

        extruder_speed = ExtruderSpeed(
            volume=extrusion_length * math.pi * 1.75**2 / 4,
            threshold_speed=threshold_speed,
            acceleration=acceleration,
            total_time=nozzle_speed.total_time,
            printer=SimpleNamespace(feedstock_filament_diameter=1.75),
        )
        extruder_speed.calculate_displacements()

        volumes = AccelerationVolume.calculate_volumes_with_acceleration(
            number_simulation_steps=5,
            step_size=20,
            feedstock_filament_diameter=1.75,
            nozzle_profile=nozzle_speed,
            extruder_profile=extruder_speed,
        )

        expected_volume = extrusion_length * math.pi * 1.75**2 / 4

        assert volumes[-1] == pytest.approx(expected_volume, rel=1e-3)
        assert all(a <= b for a, b in zip(volumes, volumes[1:]))

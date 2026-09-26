from pathlib import Path

from app.configs.printer import Printer
from app.instructions.gcode import Gcode


class TestGcode:
    def test_should_read_the_gcode(self):
        fixtures = Path(__file__).resolve().parents[1] / "fixtures"
        gcode = Gcode(
            gcode_path=str(fixtures / "gcode_example.gcode"),
            default_nozzle_speed=40.0,
            printer=Printer(config_path=str(fixtures / "printer_settings.json")),
        )

        gcode.read()

        assert gcode.number_printed_filaments == 3

        assert len(gcode.movements) == 8

        assert len(gcode.filaments_coordinates) == 3

        assert gcode.coordinate_limits["x"] == [10.0, 14.0]
        assert gcode.coordinate_limits["y"] == [8.0, 12.0]
        assert gcode.coordinate_limits["z"] == [0.0, 0.7]

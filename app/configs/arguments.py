import argparse


class Arguments:
    @staticmethod
    def get_options():
        parser = argparse.ArgumentParser()
        parser.add_argument("--gcode", type=str)
        parser.add_argument("--sim", type=str)
        parser.add_argument("--printer", type=str)
        parser.add_argument("--preview", action="store_true", help="Enable preview mode (fast, lightweight visualization)")

        return parser.parse_args()

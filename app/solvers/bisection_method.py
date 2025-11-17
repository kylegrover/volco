import logging


logger = logging.getLogger(__name__)


def default_increase_solver_tolerance(point_a, point_b):
    return False


class BisectionMethod:
    def execute(
        self,
        fun,
        initial_point,
        tolerance,
        increment,
        args,
        fun_increase_tolerance=default_increase_solver_tolerance,
    ):
        if initial_point == 0.0:
            point_b = increment
        else:
            point_b = initial_point

        fb, out, point_a, point_b, updated_args = self._loop_fb(fun, point_b, increment, args)

        # _loop_fc returns (fc, out) where out may be the raw output from `fun`.
        fc, out = self._loop_fc(
            fun, point_a, point_b, tolerance, fun_increase_tolerance, updated_args
        )

        # If `out` is a tuple like (volume_overshoot, voxel_space), unwrap it
        if isinstance(out, tuple) and len(out) == 2:
            return fc, out[1]

        return fc, out

    def _loop_fb(self, fun, point_b, inc, args):
        fb = -1.0
        point_a = 0.0
        updated_args = list(args)
        while fb < 0.0:
            fb, out = fun(point_b, *updated_args)
            # If the function returns an updated voxel_space as the second value,
            # `out` will be that ndarray (it will have a `shape` attribute).
            if hasattr(out, "shape"):
                updated_args[0] = out
            if fb < 0:
                point_a = point_b
                point_b += inc
        return fb, out, point_a, point_b, updated_args

    def _loop_fc(self, fun, point_a, point_b, tolerance, fun_increase_tolerance, args):
        fc = 2.0 * tolerance
        updated_args = list(args)
        while abs(fc) > tolerance:
            point_c = (point_a + point_b) * 0.5
            fc, out = fun(point_c, *updated_args)
            if hasattr(out, "shape"):
                updated_args[0] = out
            if fun_increase_tolerance(point_a, point_b):
                logger.debug(
                    "[BisectionMethod]: increasing tolerance because point_b and point_a are too close"
                )
                tolerance = tolerance * 10

            if fc < 0.0:
                point_a = point_c
            else:
                point_b = point_c

        return fc, out

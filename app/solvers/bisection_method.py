import logging
import math


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
        max_evaluations=128,
    ):
        # Kept for smooth scalar callers. Voxel deposition has its own discrete
        # closest-candidate contract. Never widen a caller's error tolerance.
        if not math.isfinite(initial_point) or initial_point < 0:
            raise ValueError('initial_point must be finite and nonnegative')
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('tolerance must be finite and positive')
        if not math.isfinite(increment) or increment <= 0:
            raise ValueError('increment must be finite and positive')
        if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int) or max_evaluations <= 0:
            raise ValueError('max_evaluations must be a positive integer')
        evaluations = 0
        original_fun = fun

        def bounded_fun(point, *values):
            nonlocal evaluations
            if evaluations >= max_evaluations:
                raise RuntimeError('Bisection evaluation budget exhausted')
            if not math.isfinite(point):
                raise RuntimeError('Bisection point overflow')
            evaluations += 1
            residual, output = original_fun(point, *values)
            if not math.isfinite(residual):
                raise ValueError('Bisection residual must be finite')
            return residual, output

        fun = bounded_fun
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
            # If the function returns an updated voxel_space, propagate it.
            # The returned `out` may be an ndarray (has `shape`) or a
            # VoxelSpace-like object with a `space` attribute. Accept both.
            if hasattr(out, "shape") or hasattr(out, "space"):
                updated_args[0] = out
            if fb < 0:
                point_a = point_b
                point_b += inc
        return fb, out, point_a, point_b, updated_args

    def _loop_fc(self, fun, point_a, point_b, tolerance, fun_increase_tolerance, args):
        fc = 2.0 * tolerance
        updated_args = list(args)
        while abs(fc) > tolerance:
            point_c = point_a + (point_b - point_a) * 0.5
            if point_c == point_a or point_c == point_b:
                raise RuntimeError('Bisection stagnated without meeting tolerance')
            fc, out = fun(point_c, *updated_args)
            if hasattr(out, "shape") or hasattr(out, "space"):
                updated_args[0] = out

            if fc < 0.0:
                point_a = point_c
            else:
                point_b = point_c

        return fc, out

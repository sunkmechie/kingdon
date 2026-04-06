import cmath
import math
import numpy as np

from sympy import Expr, Integer, Piecewise
from sympy import atanh as sympy_atanh
from sympy import atan2 as sympy_atan2
from sympy import sqrt as sympy_sqrt


def _truthy(value):
    return value.any() if hasattr(value, 'any') else value


def _is_complex_value(value):
    if isinstance(value, Expr):
        return value.is_real is False
    if isinstance(value, complex):
        return True

    return np.iscomplexobj(value)


def _is_negative_real(value):
    if isinstance(value, Expr):
        return value.is_real and value.is_negative
    if isinstance(value, complex):
        return value.imag == 0 and value.real < 0

    if np.iscomplexobj(value):
        return _truthy((value.imag == 0) & (value.real < 0))

    return _truthy(value < 0)


def log(rotor, *, arctanh2=None, sqrt=None):
    r"""
    Calculate the principal logarithm of a simple rotor with scalar and bivector parts.
    On normalized simple rotors this is the inverse of `exp` on its principal branch.
    The rotor need not be normalized.

    Works for python float, int and complex dtypes, numpy arrays, and for symbolic expressions using sympy.
    For more control, it is possible to explicitly provide `arctanh2` and `sqrt` functions.
    If you provide one, you must provide both.

    If the coefficients are array-valued, custom `sqrt` and `arctanh2` functions can optionally
    accept the keyword masks `mask_circular`, `mask_hyperbolic`, and `mask_null`.

    The argument to `sqrt` is the scalar :math:`\langle S^2 \rangle_0`, where :math:`S`
    is the bivector part of the rotor.

    **Numerical tolerance** – when the bivector square is numerically close to zero we treat it as
    zero using an absolute tolerance of ``1e-12`` (scaled by the magnitude of the scalar when
    appropriate). This avoids division‑by‑tiny‑numbers for near‑null rotors.
    """
    funcs = (arctanh2, sqrt)
    if any(func is None for func in funcs) and any(func is not None for func in funcs):
        raise TypeError('Please provide `arctanh2` and `sqrt` together.')

    rotor = validate_log_input(rotor)
    scalar_part, bivector_part, scalar, bivector_square = split_scalar_and_bivector(rotor)
    symbolic = rotor.issymbolic or isinstance(bivector_square, Expr) or isinstance(scalar, Expr)
    array_valued = len(rotor) > 0

    custom_helpers = (sqrt is not None) or (arctanh2 is not None)
    if not custom_helpers:
        sqrt, arctanh2 = get_default_log_helpers(
            square=bivector_square,
            scalar=scalar,
            symbolic=symbolic,
            array_valued=array_valued,
        )

    if not bivector_part.filter(_truthy):
        if _is_negative_real(scalar):
            raise ValueError('The principal logarithm is undefined for negative real scalars.')
        return bivector_part

    coefficient = compute_log_coefficient(
        scalar=scalar,
        bivector_square=bivector_square,
        sqrt=sqrt,
        arctanh2=arctanh2,
        symbolic=symbolic,
        array_valued=array_valued,
        custom_helpers=custom_helpers,
    )
    return bivector_part * coefficient


def validate_log_input(rotor):
    """Validate that the input multivector is supported by log()."""
    rotor = rotor.filter(_truthy)
    if not rotor:
        raise ValueError('The logarithm is undefined for the zero multivector.')
    if any(grade not in (0, 2) for grade in rotor.grades):
        raise NotImplementedError(
            'Currently only simple rotors with scalar and bivector parts can be logarithmized.'
        )
    return rotor


def split_scalar_and_bivector(rotor):
    """Split a rotor into scalar and bivector parts."""
    scalar_part = rotor.grade(0)
    bivector_part = rotor.grade(2)
    bivector_square = (bivector_part * bivector_part).filter(_truthy)
    if bivector_square.grades and bivector_square.grades != (0,):
        raise NotImplementedError(
            'Currently only rotors with a simple bivector part can be logarithmized.'
        )
    return scalar_part, bivector_part, scalar_part.e, bivector_square.e


def classify_bivector_square(square, *, symbolic, array_valued):
    """Classify the bivector square into circular, hyperbolic, or null branches."""
    if symbolic:
        return dict(
            mask_circular=square < 0,
            mask_hyperbolic=square > 0,
            mask_null=False,
        )
    if array_valued:
        mask_circular = square < 0
        mask_hyperbolic = square > 0
        return dict(
            mask_circular=mask_circular,
            mask_hyperbolic=mask_hyperbolic,
            mask_null=~(mask_circular | mask_hyperbolic),
        )
    return dict(
        mask_circular=square < 0,
        mask_hyperbolic=square > 0,
        mask_null=not (square < 0 or square > 0),
    )


def get_default_log_helpers(*, square, scalar, symbolic, array_valued):
    """Return the default sqrt and arctanh2 helpers for the active backend."""
    complex_valued = _is_complex_value(square) or _is_complex_value(scalar)

    if symbolic:
        if complex_valued:
            return sympy_sqrt, lambda y, x: sympy_atanh(y / x)

        def sqrt(x, *, mask_circular=False, mask_hyperbolic=False, mask_null=False):
            return Piecewise(
                (sympy_sqrt(-x), mask_circular),
                (sympy_sqrt(x), mask_hyperbolic),
                (Integer(0), True),
            )

        def arctanh2(y, x, *, mask_circular=False, mask_hyperbolic=False, mask_null=False):
            return Piecewise(
                (sympy_atan2(y, x), mask_circular),
                (sympy_atanh(y / x), mask_hyperbolic),
                (Integer(0), True),
            )

        return sqrt, arctanh2

    if array_valued:
        import numpy as np

        if complex_valued:
            return np.sqrt, lambda y, x: np.arctanh(y / x)

        def sqrt(x, *, mask_circular=False, mask_hyperbolic=False, mask_null=False):
            res = 0 * x
            if mask_circular.any():
                res[mask_circular] = np.sqrt(-x[mask_circular])
            if mask_hyperbolic.any():
                res[mask_hyperbolic] = np.sqrt(x[mask_hyperbolic])
            return res

        def arctanh2(y, x, *, mask_circular=False, mask_hyperbolic=False, mask_null=False):
            sample = x if hasattr(x, 'shape') and x.shape != () else y
            res = 0 * sample
            x_is_array = hasattr(x, 'shape') and x.shape != ()
            if mask_circular.any():
                x_circular = x[mask_circular] if x_is_array else x
                res[mask_circular] = np.arctan2(y[mask_circular], x_circular)
            if mask_hyperbolic.any():
                x_hyperbolic = x[mask_hyperbolic] if x_is_array else x
                res[mask_hyperbolic] = np.arctanh(y[mask_hyperbolic] / x_hyperbolic)
            return res

        return sqrt, arctanh2

    if complex_valued:
        return (lambda x: x ** 0.5), (lambda y, x: cmath.atanh(y / x))

    def sqrt(x, *, mask_circular=False, mask_hyperbolic=False, mask_null=False):
        if mask_circular:
            return math.sqrt(-x)
        if mask_hyperbolic:
            return math.sqrt(x)
        return 0.0

    def arctanh2(y, x, *, mask_circular=False, mask_hyperbolic=False, mask_null=False):
        if mask_circular:
            return math.atan2(y, x)
        if mask_hyperbolic:
            return math.atanh(y / x)
        return 0.0

    return sqrt, arctanh2


def compute_log_coefficient(
    *,
    scalar,
    bivector_square,
    sqrt,
    arctanh2,
    symbolic,
    array_valued,
    custom_helpers,
):
    """Compute the scalar coefficient multiplying the bivector part in log()."""
    complex_valued = _is_complex_value(bivector_square) or _is_complex_value(scalar)
    scalar_is_array = hasattr(scalar, 'shape') and scalar.shape != ()

    if complex_valued:
        root = sqrt(bivector_square)
        angle = arctanh2(root, scalar)

        # Tolerance for floating‑point zero detection (helps when B² is numerically tiny).
        eps = 1e-12
        if array_valued:
            sample = scalar if scalar_is_array else root
            coefficient = 0 * sample
            # Use a tolerance instead of exact equality.
            mask_zero = np.abs(root) < eps
            mask_nonzero = ~mask_zero
            if mask_nonzero.any():
                coefficient[mask_nonzero] = angle[mask_nonzero] / root[mask_nonzero]
            if mask_zero.any():
                scalar_zero = scalar[mask_zero] if scalar_is_array else scalar
                if _is_negative_real(scalar_zero):
                    raise ValueError('The principal logarithm is undefined for negative real scalars.')
                coefficient[mask_zero] = 1 / scalar_zero
            return coefficient
        # Scalar (non‑array) path – use absolute tolerance.
        if np.abs(root) < eps:
            if _is_negative_real(scalar):
                raise ValueError('The principal logarithm is undefined for negative real scalars.')
            return 1 / scalar
        return angle / root

    branch_masks = classify_bivector_square(bivector_square, symbolic=symbolic, array_valued=array_valued)
    if custom_helpers:
        root = sqrt(bivector_square)
        angle = arctanh2(root, scalar)
    else:
        root = sqrt(bivector_square, **branch_masks)
        angle = arctanh2(root, scalar, **branch_masks)

    if symbolic:
        return Piecewise(
            (angle / root, branch_masks['mask_circular']),
            (angle / root, branch_masks['mask_hyperbolic']),
            (Integer(1) / scalar, True),
        )

    if array_valued:
        sample = scalar if scalar_is_array else root
        coefficient = 0 * sample
        mask_nonnull = branch_masks['mask_circular'] | branch_masks['mask_hyperbolic']
        if mask_nonnull.any():
            coefficient[mask_nonnull] = angle[mask_nonnull] / root[mask_nonnull]
        if branch_masks['mask_null'].any():
            scalar_null = scalar[branch_masks['mask_null']] if scalar_is_array else scalar
            if _is_negative_real(scalar_null):
                raise ValueError('The principal logarithm is undefined for negative real scalars.')
            coefficient[branch_masks['mask_null']] = 1 / scalar_null
        return coefficient

    if branch_masks['mask_null']:
        if _is_negative_real(scalar):
            raise ValueError('The principal logarithm is undefined for negative real scalars.')
        return 1 / scalar
    return angle / root

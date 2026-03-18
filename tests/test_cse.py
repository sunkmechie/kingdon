"""
CSE operation count tests, ported from GAmphetamine.test.js.

These tests verify the number of multiplications and additions in the
generated code for common 3DPGA operations. The expected counts match
GAmphetamine's polynomial CSE output, and serve as targets for porting
the polynomial CSE algorithm from polynomial.js to polynomial.py.

Most tests will fail until polynomial CSE is implemented.
"""
import re
import pytest
from kingdon import Algebra, MultiVector
from kingdon.polynomial import RationalPolynomial
from kingdon.codegen import do_codegen, codegen_sw, codegen_gp, codegen_rp, codegen_ip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def one():
    return RationalPolynomial([[1]])


def rp(name):
    return RationalPolynomial.fromname(name)


def get_op_counts(func):
    """Return (muls, adds) from the function docstring, or (None, None)."""
    doc = func.__doc__ or ''
    m = re.match(r'(\d+) muls / (\d+) adds', doc.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def make_even(alg, prefix='a'):
    """Full even multivector (grades 0,2,4) with symbolic RP coefficients."""
    keys = tuple(alg.indices_for_grades((0, 2, 4)))
    vals = [rp(f'{prefix}{alg.bin2canon[k][1:]}') for k in keys]
    return MultiVector.fromkeysvalues(alg, keys, vals)


def make_normalized_point(alg, prefix='b'):
    """Trivector with symbolic x,y,z and w=1 (normalized point).

    In 3DPGA: e032=b0, e013=b1, e021=b2, e123=1.
    Binary keys using Algebra.fromname('3DPGA'): e123=7, e021=11, e013=13, e032=14.
    """
    keys = (7, 11, 13, 14)
    vals = [
        one(),                                    # e123 = 1 (normalized)
        rp(f'{prefix}{alg.bin2canon[11][1:]}'),   # e021
        rp(f'{prefix}{alg.bin2canon[13][1:]}'),   # e013
        rp(f'{prefix}{alg.bin2canon[14][1:]}'),   # e032
    ]
    return MultiVector.fromkeysvalues(alg, keys, vals)


def make_direction(alg, prefix='b'):
    """Trivector with symbolic x,y,z only (direction, no homogeneous component)."""
    keys = (11, 13, 14)
    vals = [rp(f'{prefix}{alg.bin2canon[k][1:]}') for k in keys]
    return MultiVector.fromkeysvalues(alg, keys, vals)


def make_origin(alg):
    """Trivector with only e123=1 (the origin point)."""
    return MultiVector.fromkeysvalues(alg, (7,), [one()])


def make_pure_e032(alg):
    """Single e032 blade with value 1."""
    return MultiVector.fromkeysvalues(alg, (14,), [one()])


def make_rotation(alg, prefix='a'):
    """Rotation rotor: scalar=1, e12=x, e31=y, e23=z.

    In 3DPGA: the rotation bivectors are e12(3), e31(5), e23(6).
    This matches GAmphetamine Element.even("1","a0","a1","a2","0","0","0","0").
    """
    keys = (0, 3, 5, 6)
    vals = [
        one(),
        rp(f'{prefix}{alg.bin2canon[3][1:]}'),   # e12
        rp(f'{prefix}{alg.bin2canon[5][1:]}'),   # e31
        rp(f'{prefix}{alg.bin2canon[6][1:]}'),   # e23
    ]
    return MultiVector.fromkeysvalues(alg, keys, vals)


def make_translation(alg, prefix='a'):
    """Translation motor: scalar=1, e01=x, e02=y, e03=z.

    In 3DPGA: the ideal (null) bivectors are e01(9), e02(10), e03(12).
    This matches GAmphetamine Element.even("1","0","0","0","a0","a1","a2","0").
    """
    keys = (0, 9, 10, 12)
    vals = [
        one(),
        rp(f'{prefix}{alg.bin2canon[9][1:]}'),   # e01
        rp(f'{prefix}{alg.bin2canon[10][1:]}'),  # e02
        rp(f'{prefix}{alg.bin2canon[12][1:]}'),  # e03
    ]
    return MultiVector.fromkeysvalues(alg, keys, vals)


def make_vector(alg, prefix='b'):
    """Grade-1 vector (plane in 3DPGA) with all symbolic RP coefficients."""
    keys = tuple(alg.indices_for_grade(1))
    vals = [rp(f'{prefix}{alg.bin2canon[k][1:]}') for k in keys]
    return MultiVector.fromkeysvalues(alg, keys, vals)


def make_bivector(alg, prefix='b'):
    """Grade-2 bivector (line in 3DPGA) with all symbolic RP coefficients."""
    keys = tuple(alg.indices_for_grade(2))
    vals = [rp(f'{prefix}{alg.bin2canon[k][1:]}') for k in keys]
    return MultiVector.fromkeysvalues(alg, keys, vals)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pga3d():
    return Algebra.fromname('3DPGA', cse=True)


@pytest.fixture(scope='module')
def pga3d_no_cse():
    return Algebra.fromname('3DPGA', cse=False)


# ---------------------------------------------------------------------------
# Tests 1-3: sandwich product of even element with various trivectors
# ---------------------------------------------------------------------------

def test_sw_even_normalized_point(pga3d):
    """3DPGA normalized even >>> normalized point must be 21 muls, 18 adds."""
    a = make_even(pga3d)
    b = make_normalized_point(pga3d)
    _, func = do_codegen(codegen_sw, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 21 and adds == 18


def test_sw_even_direction(pga3d):
    """3DPGA normalized even >>> direction must be 18 muls, 12 adds."""
    a = make_even(pga3d)
    b = make_direction(pga3d)
    _, func = do_codegen(codegen_sw, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 18 and adds == 12


def test_sw_even_origin(pga3d):
    """3DPGA normalized even >>> origin must be 15 muls, 9 adds."""
    a = make_even(pga3d)
    b = make_origin(pga3d)
    _, func = do_codegen(codegen_sw, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 15 and adds == 9


# ---------------------------------------------------------------------------
# Test 4: geometric product of two even elements
# ---------------------------------------------------------------------------

def test_gp_even_even(pga3d):
    """3DPGA normalized even * even must be 48 muls, 40 adds."""
    a = make_even(pga3d, 'a')
    b = make_even(pga3d, 'b')
    _, func = do_codegen(codegen_gp, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 48 and adds == 40


# ---------------------------------------------------------------------------
# Test 5: (even >>> e032) / 2, no CSE
# ---------------------------------------------------------------------------

def test_sw_even_e032_half_no_cse(pga3d_no_cse):
    """3DPGA normalized (even >>> e032) / 2 must be 6 muls, 4 adds (no CSE)."""
    from fractions import Fraction

    a = make_even(pga3d_no_cse)
    b = make_pure_e032(pga3d_no_cse)

    def codegen_sw_half(a, b):
        result = codegen_sw(a, b)
        half = RationalPolynomial([[Fraction(1, 2)]])
        keys = tuple(result.keys())
        vals = [half * v for v in result.values()]
        return MultiVector.fromkeysvalues(a.algebra, keys, vals)

    _, func = do_codegen(codegen_sw_half, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 6 and adds == 4


# ---------------------------------------------------------------------------
# Tests 6-8: geometric products of specialized even elements
# ---------------------------------------------------------------------------

def test_gp_even_translation(pga3d):
    """3DPGA compose even, translation must be 12 muls, 12 adds."""
    a = make_even(pga3d, 'a')
    b = make_translation(pga3d, 'b')
    _, func = do_codegen(codegen_gp, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 12 and adds == 12


def test_gp_translation_translation(pga3d):
    """3DPGA compose translation, translation must be 0 muls, 3 adds."""
    a = make_translation(pga3d, 'a')
    b = make_translation(pga3d, 'b')
    _, func = do_codegen(codegen_gp, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 0 and adds == 3


def test_gp_rotation_rotation(pga3d):
    """3DPGA compose rotation, rotation must be 9 muls, 12 adds."""
    a = make_rotation(pga3d, 'a')
    b = make_rotation(pga3d, 'b')
    _, func = do_codegen(codegen_gp, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 9 and adds == 12


# ---------------------------------------------------------------------------
# Tests 9-11: regressive product (join)
# ---------------------------------------------------------------------------

def test_rp_point_point(pga3d):
    """3DPGA join two points must be 6 muls, 6 adds."""
    a = make_normalized_point(pga3d, 'a')
    b = make_normalized_point(pga3d, 'b')
    _, func = do_codegen(codegen_rp, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 6 and adds == 6


def test_rp_point_line(pga3d):
    """3DPGA join point and line must be 9 muls, 9 adds."""
    a = make_normalized_point(pga3d, 'a')
    b = make_bivector(pga3d, 'b')
    _, func = do_codegen(codegen_rp, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 9 and adds == 9


def test_rp_three_points(pga3d):
    """3DPGA join three points must be 9 muls, 12 adds."""
    a = make_normalized_point(pga3d, 'a')
    b = make_normalized_point(pga3d, 'b')
    c = make_normalized_point(pga3d, 'c')

    def codegen_join3(a, b, c):
        return a.rp(b).rp(c)

    _, func = do_codegen(codegen_join3, a, b, c)
    muls, adds = get_op_counts(func)
    assert muls == 9 and adds == 12


# ---------------------------------------------------------------------------
# Tests 12-14: compound projection expressions
# These require polynomial CSE to match the expected counts.
# ---------------------------------------------------------------------------

def test_project_point_on_plane(pga3d):
    """3DPGA project point on plane must be 18 muls, 12 adds.

    Expression: (a | b) / b  where a is a normalized point and b is a plane.
    """
    a = make_normalized_point(pga3d, 'a')
    b = make_vector(pga3d, 'b')

    def codegen_proj_point_plane(a, b):
        return a.ip(b) * b.inv()

    _, func = do_codegen(codegen_proj_point_plane, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 18 and adds == 12


def test_project_point_on_normalized_plane(pga3d):
    """3DPGA project point on normalized plane must be 6 muls, 6 adds.

    Expression: (a | b) * b + (a * (1 - b * ~b)).grade(3)
    This uses the fact that b is normalized (b * ~b = 1), allowing
    polynomial CSE to simplify the second term.
    """
    a = make_normalized_point(pga3d, 'a')
    b = make_vector(pga3d, 'b')

    def codegen_proj_point_norm_plane(a, b):
        alg = a.algebra
        e = MultiVector.fromkeysvalues(alg, (0,), [one()])  # scalar 1
        ip_ab = a.ip(b)
        b_normsq = b * b.reverse()   # = scalar b*~b
        correction = (a * (e - b_normsq)).grade(3)
        return ip_ab * b + correction

    _, func = do_codegen(codegen_proj_point_norm_plane, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 6 and adds == 6


def test_project_point_on_normalized_line(pga3d):
    """3DPGA project point on normalized line must be 15 muls, 15 adds.

    Expression: (a | b) * (-b) + (a * (1 - b * ~b)).grade(3)
    """
    a = make_normalized_point(pga3d, 'a')
    b = make_bivector(pga3d, 'b')

    def codegen_proj_point_norm_line(a, b):
        alg = a.algebra
        e = MultiVector.fromkeysvalues(alg, (0,), [one()])
        ip_ab = a.ip(b)
        b_normsq = b * b.reverse()
        correction = (a * (e - b_normsq)).grade(3)
        return ip_ab * (-b) + correction

    _, func = do_codegen(codegen_proj_point_norm_line, a, b)
    muls, adds = get_op_counts(func)
    assert muls == 15 and adds == 15

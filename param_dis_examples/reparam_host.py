import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
from numpy.random import rand


#===========Check loma_code/reparam.py and define here===========
_m, _k = -3.0, 0.5
_n, _p = 0.0, 0.0
_op = ">"
#===========Check loma_code/reparam.py and define here===========

"""
3x < 0.5t -> positive; CORRECT
3x > 0.5t -> negative; CORRECT

-3x < 0.5t -> positive; (correct integral, 0, wrong expect dval)
-3x > 0.5t -> negative; (correct integral, 0, wrong expect dval)

3x < -0.5t -> negative; CORRECT
3x > -0.5t -> positive; CORRECT

-3x < -0.5t -> negative; (correct integral, 0, wrong expect dval)
-3x > -0.5t -> positive; (correct integral, 0, wrong expect dval)
"""
positive = bool("<" in _op) == bool(_k > 0.0)
interval_reversed = False

def t_should_between(low, up):
    """This time the integrand has general form
    [_m*x+_n {<, >} _k*t+_p];
    something looks like
    [(3.0 * x - 0.77) < (0.5 * t - 0.07)]
    Now the t that can produce non-zero derivative is no longer
    between low and up.

    In the compiler implementation, we check
    [R(lower) < 0 < R(upper)]
    Here, to test, we do a different manual computation.
    And t should produce non-zero derivative when
    [lhs < t < rhs]
    """
    # now, t must be between
    lhs = (_m * low + _n - _p) / _k
    rhs = (_m * up + _n - _p) / _k
    if lhs < rhs:
        global interval_reversed
        interval_reversed = True
        lhs, rhs = rhs, lhs
    return lhs, rhs

def correct_dval() -> float:
    return _k/_m * (1.0 if "<" in _op else -1.0)

def correct_val(lhs, rhs, interval_length, t) -> float:
    numerator = (t-lhs) if positive else (rhs-t)
    return numerator / (rhs - lhs) * interval_length

def correct_vals_dvals(choices, lhs, rhs, interval_length):
    vals, dvals = [], []
    for t in choices:
        if t <= lhs:
            # first and last, out of bound with zero derivative
            dvals.append(0.0)
            vals.append(interval_length * (1 - int(positive)))
        elif t >= rhs:
            dvals.append(0.0)
            vals.append(interval_length * int(positive))
        else:
            # non-zero case
            dvals.append(correct_dval())
            vals.append(correct_val(lhs, rhs, interval_length, t))

    return vals, dvals

if __name__ == '__main__':
    with open('loma_code/reparam.py') as f:
        structs, lib = compiler.compile(
            f.read(),
            target = 'c',
            output_filename = '_code/reparam'
        )

    low, up = 5.0, 7.0
    _dfloat = structs['_dfloat']
    dlow, dup = _dfloat(low, 0.0), _dfloat(up, 0.0)

    rhs, lhs = t_should_between(low, up)
    # choices = [lhs-1.0, lhs + (rhs - lhs) * float(rand()), rhs+1.0]
    choices = [lhs-1.0, 34.0 * (1 if _k * _m > 0 else -1), rhs+1.0]
    vals, dvals = correct_vals_dvals(choices, lhs, rhs, up-low)

    print(f"We take integral from low={low} to up={up}")
    print(f"integrand is indicator [{_m}*x+{_n} {_op} {_k}*t+{_p}]")
    print(f"And t should produce non-zero integral value and derivative when [{lhs} < t < {rhs}]")

    for t, val, dval in zip(choices, vals, dvals):
        df_t = _dfloat(t, 1.0)
        res = lib.fwd_IntegralEval(dlow, dup, df_t)
        print(f"With t={t:.3f}, Integral is {res.val:.3f}, expect {val:.3f} "
            f"derivative is {res.dval:.3f}, expect {dval:.3f}")

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes

if __name__ == '__main__':
    with open('loma_code/reparam.py') as f:
        structs, lib = compiler.compile(
            f.read(),
            target = 'c',
            output_filename = '_code/reparam'
        )

    """This time the integrand is
    [(3.0 * x + 0.77) < (0.5 * t + 0.07)]
    Now the t that can produce non-zero derivative is no longer
    between low and up.
    """
    low, up = 5.0, 7.0
    _m, _k = 3.0, 0.5
    _n, _p = 0.77, 0.07
    # now, t must be between
    lhs = (_m * low + _n - _p) / _k
    rhs = (_m * up + _n - _p) / _k
    choices = [lhs-1.0, (lhs+rhs)/2, rhs+1.0]


    _dfloat = structs['_dfloat']
    dlow, dup = _dfloat(low, 0.0), _dfloat(up, 0.0)
    for t in choices:
        df_t = _dfloat(t, 1.0)
        res = lib.fwd_IntegralEval(dlow, dup, df_t)
        print(f"With t={t}, IntegralEval({low}, {up}, [x<t], t) gives "
            f"result {res.val:.3f} and "
            f"derivative {res.dval:.3f}")

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
from numpy.random import rand
# Teg import
from teg import TegVar, Var, Teg, IfElse
from teg.derivs import FwdDeriv
from teg.eval.numpy_eval import evaluate


#===========Check loma_code/reparam.py and define here===========
_m, _k = 3.0, 0.5
_n, _p = 12.4, 5.6
ret_if, ret_else = -11.0, -1.0
_op = "<"
#===========Check loma_code/reparam.py and define here===========

"""General if-else: _m*x+_n {<, >} _k*t+_p ? ret_if : ret_else

This is an easy update:
Nothing needs to be changed for the correct Integral;
Simply multiply with (ret_if - ret_else) to get the correct derivative of Integral

In the previous setting (reparam_host.py), ret_if, ret_else = 1.0, 0.0
"""

def t_should_between(low, up):
    """Effective range shouldn't change, see reparam_host.py
    """
    # now, t must be between
    lhs = (_m * low + _n - _p) / _k
    rhs = (_m * up + _n - _p) / _k
    # ensure lhs < rhs
    if lhs > rhs:
        lhs, rhs = rhs, lhs
    return lhs, rhs

# Let Teg compute the correct val and dval for reference
def correct_val_dval(_t: float):
    x, t = TegVar('x'), Var('t', _t)

    # we can't introduce IfElse (discountinuity) to this function,
    # so we define it dynamically
    if _op in ("<", "<="):
        def integrand_pd():
            return (_m * x + _n) < (_k * t + _p)
    elif _op in (">", ">="):
        def integrand_pd():
            return (_m * x + _n) > (_k * t + _p)
    else:
        assert False, "TYPO"

    body = IfElse(integrand_pd(), ret_if, ret_else)
    expr = Teg(5, 7, body, x)
    deriv_expr = FwdDeriv(expr, [(t, 1)])
    return evaluate(expr), evaluate(deriv_expr)

if __name__ == '__main__':
    with open('loma_code/more_reparam.py') as f:
        structs, lib = compiler.compile(
            f.read(),
            target = 'c',
            output_filename = '_code/more_reparam'
        )

    low, up = 5.0, 7.0  # they will change Integral but not derivative of Integral
    _dfloat = structs['_dfloat']
    dlow, dup = _dfloat(low, 0.0), _dfloat(up, 0.0)

    lhs, rhs = t_should_between(low, up)
    choices = [lhs-1.0, lhs + (rhs - lhs) * float(rand()), rhs+1.0]
    # choices = [lhs-1.0, (lhs+rhs)/2, rhs+1.0]  # debug use: set _n, _p = 0

    print(f"*****DOUBLE CHECK integrand_pd in loma_code***** " 
        f"{_m}*x+{_n} {_op} {_k}*t+{_p} ? {ret_if} : {ret_else}")
    print(f"We take integral from low={low} to up={up}", \
        f" and expect non-zero derivative when [{lhs:.3f} < t < {rhs:.3f}]")

    for _t in choices:
        val, dval = correct_val_dval(_t)
        df_t = _dfloat(_t, 1.0)
        res = lib.fwd_IntegralEval(dlow, dup, df_t)
        print(f"With t={_t:.3f}, Integral is {res.val:.3f}, expect {val:.3f}; "
            f"derivative is {res.dval:.3f}, expect {dval:.3f}")

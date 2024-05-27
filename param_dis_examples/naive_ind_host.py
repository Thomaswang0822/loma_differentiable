import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes

if __name__ == '__main__':
    with open('loma_code/naive_ind.py') as f:
        structs, lib = compiler.compile(
            f.read(),
            target = 'c',
            output_filename = '_code/naive_ind'
        )

    """ 
    IntegralEval(low, up, f(x,t), dx), where f(x,t) is indicator function [x < t]
    So integral itself should eval to
    * 0 if t<low 
    * up-low if t>up
    * t-low if low <= t < up
    """

    low, up = 5, 7

    # Check on naive "discretize before differentiate"
    _dfloat = structs['_dfloat']
    dlow, dup = _dfloat(low, 0.0), _dfloat(up, 0.0)
    for t in (low-1.0, low+0.5, up+1.0):
        df_t = _dfloat(t, 1.0)
        res = lib.fwd_IntegralEval(dlow, dup, df_t)
        print(f"With t={t}, IntegralEval({low}, {up}, [x<t], t) gives "
            f"result {res.val:.3f} and "
            f"derivative {res.dval:.3f}")

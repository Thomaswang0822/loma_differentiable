import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes

if __name__ == '__main__':
    with open('loma_code/simple_d_integral.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/simple_integral')

    _dfloat = structs['_dfloat']
    # we don't take derivative wrt upper and lower limits
    a = _dfloat(5.0, 0.0)
    b = _dfloat(7.0, 0.0)

    res = lib.fwd_simple_integral(a, b)
    print(f"Integral eval to (val, dval): {res.val}, {res.dval}")
    # We compute d/dx Integral(5 to 7, 2x+5, x), which should be
    # (2x+5).eval(5, 7)
    expected = lib.simple_integral(a.val, b.val)
    print(f"Should be (val, dval): {expected}, {0.0}")

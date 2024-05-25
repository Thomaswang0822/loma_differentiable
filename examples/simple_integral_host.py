import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes

if __name__ == '__main__':
    with open('loma_code/simple_integral.py') as f:
        _, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/simple_integral')

    # integral is 2x*dx from 0 to 1, which should eval to around 1
    # assert abs(lib.simple_integral() - 1.0) < 1e-6
    res = lib.simple_integral()
    print(f"Integral eval to {res}")

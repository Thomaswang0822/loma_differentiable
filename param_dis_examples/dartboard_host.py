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


if __name__ == '__main__':
    with open('loma_code/dartboard.py') as f:
        structs, lib = compiler.compile(
            f.read(),
            target = 'c',
            output_filename = '_code/dartboard'
        )
    print("********COMPILED SUCCESSFULLY********")
import os
import sys
import math
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


"""Application idea:
On a diskboard of radius RADIUS, there is a parameter t with 0<t<RADIUS.
If the dart is further than t (from the center), you get a good score;
otherwise, you get a bad score.
We don't care about the case where the dart doesn't land on the board.
Our goal is to find (optimize with gradient descent) t 
such that the score-weighted areas of good-score region (the inner circle)
and bad-score region (the outer ring) are equal.
"""

# global parameters
RADIUS = 6.0
EPSILON = 1e-4
FIXED_STEP_SIZE = 1e-5

good_score = 8.0
bad_score = 1.0

# reference (correct answer of t)
def compute_correct_t():
    """Essentially, t is the solution to

    pi * t^2 * good_score == pi * (R^2 - t^2) * bad_score
    <=>
    t^2 == R^2 * (bad_score / (good_score + bad_score))
    """
    return math.sqrt(RADIUS**2 * bad_score / (good_score + bad_score))

# DEBUG helper
def find_curr_areas(curr_t: float) -> tuple[float, float]:
    good_area = math.pi * curr_t**2 * good_score
    bad_area = math.pi * (RADIUS**2 - curr_t**2) * bad_score
    return good_area, bad_area

def print_correct_info():
    t = compute_correct_t()
    g, b = find_curr_areas(t)
    assert abs(g-b) < EPSILON
    print(f"Correct t should be: {t:.4f}, and we reach equal score-weighted area {g:.4f}")

if __name__ == '__main__':
    with open('loma_code/dartboard.py') as f:
        structs, lib = compiler.compile(
            f.read(),
            target = 'c',
            output_filename = '_code/dartboard'
        )
    _dfloat = structs['_dfloat']
    print("********COMPILED SUCCESSFULLY********")

    # begin with "the middle" guess
    curr_t = _dfloat(RADIUS/2, 1.0)
    R = _dfloat(RADIUS, 0.0)
    curr_diff = 1.0  # target difference
    i = 0  # debug counter

    print_correct_info()
    # adaptive GD
    while (i <= 300 and abs(curr_diff) > EPSILON):
        # DEBUG print
        g, b = find_curr_areas(curr_t.val)
        if (i % 10 == 0):
            print(f"At iter {i}, t={curr_t.val:.4f}, good and bad areas are {g:.4f} and {b:.4f}")
            # print(curr_diff)

        out_t = lib.fwd_good_bad_diff(R, curr_t)
        # update t and target
        step_size = FIXED_STEP_SIZE * (g-b)
        curr_t.val -= step_size * out_t.dval
        # BUG: integral itself (out_t.val) is very inaccurate,
        # so we use reference computation good-bad instead.
        curr_diff = g-b

        i += 1
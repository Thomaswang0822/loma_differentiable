def divide(x : In[float], y : In[float]) -> float:
    return x / y

d_divide = rev_diff(divide)

# def _d_divide():
#     _dx += _dreturn * (1/y)
#     _dy += _dreturn * (-x/y^2)
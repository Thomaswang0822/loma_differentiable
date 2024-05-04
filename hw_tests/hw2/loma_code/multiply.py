def multiply(x : In[float], y : In[float]) -> float:
    return x * y

d_multiply = rev_diff(multiply)

# def _d_multiply(
#     x: In[float], _dx: Out[float],
#     y: In[float], _dy: Out[float],
#     _dreturn: In[float]
# ):
#     """r = x * y, so ∂r/∂x = y, ∂r/∂y = x
#     dx = ∂F/∂x = ∂F/∂r * ∂r/∂x = dr * y
#     similarly,
#     dy = dr * x
#     """
#     _dx += _dreturn * y
#     _dy += _dreturn * x
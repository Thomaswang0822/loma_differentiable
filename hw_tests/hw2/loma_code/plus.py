def plus(x : In[float], y : In[float]) -> float:
    return x + y

d_plus = rev_diff(plus)

# def d_plus(
#     x: In[float], _dx: Out[float],
#     y: In[float], _dy: Out[float],
#     _dreturn: In[float]
# ):
#     _dx += _dreturn
#     _dy += _dreturn

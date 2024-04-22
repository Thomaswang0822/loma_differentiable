def identity(x : In[float]) -> float:
    return x

d_identity = rev_diff(identity)

# def d_identity(x: In[float], _dx: Out[float], _dreturn: Out[float]):
#     _dx = _dx + _dreturn
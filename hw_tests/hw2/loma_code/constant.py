def constant(x : In[float]) -> float:
    return 2.0

d_constant = rev_diff(constant)

# def d_constant(x: In[float], _dx: Out[float], _dreturn: In[float]):
#     pass
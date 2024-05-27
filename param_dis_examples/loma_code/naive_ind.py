# simply (x<t)? 1:0, float version
def integrand_f(x: In[float], t: In[float]) -> float:
    if x < t:
        return 1.0
    else:
        return 0.0

# Above function should integrated to (t-a)
def IntegralEval(lower: In[float], upper: In[float], t: In[float]) -> float:
    curr_x: float = lower
    n: int = (upper - lower) / 0.01 + 1
    i: int = 0
    res: float = 0.0
    while (i < n, max_iter := 100):
        res = res + integrand_f(curr_x, t)
        i = i + 1
        curr_x = curr_x + 0.01
    res = res * (upper - lower) / n
    return res

# Do a wrong "discretize before differentiate"
fwd_IntegralEval = fwd_diff(IntegralEval)

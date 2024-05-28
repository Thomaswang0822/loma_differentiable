# simply (x<t)? 1:0, float version
def integrand_pd(x: In[float], t: In[float]) -> float:
    if (3.0 * x + 0.77) < (0.5 * t + 0.07):
        return 1.0
    else:
        return 0.0

# Above function should integrated to (t-a)
def IntegralEval(lower: In[float], upper: In[float], t: In[float]) -> float:
    curr_x: float = lower
    n: int = (upper - lower) / 0.01 + 1
    i: int = 0
    res: float = 0.0
    while (i < n, max_iter := 10000):
        res = res + integrand_pd(curr_x, t)
        i = i + 1
        curr_x = curr_x + 0.01
    res = res * (upper - lower) / n
    return res

# Do a wrong "discretize before differentiate"
fwd_IntegralEval = fwd_diff(IntegralEval)

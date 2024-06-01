# indicator [ _m*x+_n {<, >} _k*t+_p ]
def integrand_pd(x: In[float], t: In[float]) -> float:
    if (3.0 * x + 12.4) < (0.5 * t + 5.6):
        return 1.0
    else:
        return 0.0


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

# Do a correct "differentiate before discretize"
fwd_IntegralEval = fwd_diff(IntegralEval)

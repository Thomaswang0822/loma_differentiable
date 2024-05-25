def integrand_f(x: In[float]) -> float:
    return 4*x + 5

# "fake" MC integral eval
# samples are not random, but 0.1 apart
def IntegralEval(lower: In[float], upper: In[float]) -> float:
    curr_x: float = lower
    n: int = (upper - lower) / 0.1 + 1
    i: int = 0
    res: float = 0.0
    while (i < n, max_iter := 100):
        res = res + integrand_f(curr_x)
        i = i + 1
        curr_x = curr_x + 0.1
    res = res * (upper - lower) / n
    return res

def simple_integral() -> float:
    res: float
    res = IntegralEval(0, 1)
    return res
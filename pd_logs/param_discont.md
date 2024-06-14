# System Design: Handle Parametric Discontinuity

This document provide system design ideas of handling parametric discontinuity.

At this step, we only consider the simplest indicator integrand, namely
**f(x, t) = [x < t]**

## Notations

Each concept could have more than one name/notation.

- lower integration limit: a, lower
- upper integration limit: b, upper
- interval: [lower, upper], [a, b]
- integration variable: x, dx
- discontinuous parameter: t
- integrand: f(x, t), [x < t]
- primal value: val
- derivative: dval
- parametric discontinuity: pd, PD

## Goal

Related files are

- driver code: *param_dis_examples\ind_host.py*
- loma code: *param_dis_examples\loma_code\ind.py*

We want to compute **d/dt * Integral(lower, upper, f(x, t), dx)**, which
should be evaluated to **[lower < t < upper]**. Reference: the paper.

Using the setup in the driver code, we expect to have

```shell
With t=4.0, IntegralEval(5.0, 7.0, [x<t], t) gives result 0.000 and derivative 0.000
With t=5.5, IntegralEval(5.0, 7.0, [x<t], t) gives result 0.498 and derivative 1.000
With t=8.0, IntegralEval(5.0, 7.0, [x<t], t) gives result 2.000 and derivative 0.000
```

## Assumptions

Here are some assumptions we made at the current stage. We will relax some of them later, after we have reparameterization to handle more general PD.

- Everything is in float
- Single variable (1D) integral
- Derivatives won't be taken wrt. lower and upper. In practice, loma will compute them, but we don't guarantee the correctness.
- There will only be "simple" If-Else condition. This means we only consider **if x < t** but not **if (x-5) > (7-t)**. General conditions will be resolved by reparameterization, the next step of the project.
- Both `then_stmts` and `else_stmts` will only have a single Return statement. This maybe relaxed depending on the difficulty and time.
- `else_stmts` will return 0.0, otherwise the integrand is no longer a indicator function. This will also be dealt after we have reparameterization.

## Design

### integral eval caller

First, the **IntegralEval** function will be moved to the *compiler.py* because it's a
general-purpose caller that evaluates the integral. It has nothing special, just a "fake" Monte-Carlo that places samples evenly (0.01 apart) in [a, b].

```python
def IntegralEval(lower: In[float], upper: In[float], t: In[float]) -> float:
    curr_x: float = lower
    n: int = (upper - lower) / 0.01 + 1
    i: int = 0
    res: float = 0.0
    while (i < n, max_iter := 1000):
        res = res + integrand_pd(curr_x, t)
        i = i + 1
        curr_x = curr_x + 0.01
    res = res * (upper - lower) / n
    return res
```

After being prepended to the actual user code, its foward-diff version will be taken care by the auto-diff pipeline.

### integrands

With our integral representation design, users will define the integrands themselves (could be many). **f(x, t) = [x < t]** will look like

```python
def integrand_pd(x: In[float], t: In[float]) -> float:
    if x < t:
        return 1.0
    else:
        return 0.0
```

When the integrand doens't have a discountinuous parameter (no If-Else), there isn't any special treatment. The integrand is just a custom function call. See *param_dis_examples\loma_code\simple_d_integral.py*

When there is a discontinuity, however, we need 2 things:

1. In the fwd-diff version of caller **IntegralEval**, **integrand_pd(curr_x, t)** will be turned to **_d_fwd_integrand_pd(curr_x,t,lower,upper)**
2. When differentiating the integrand, we manually construct the code such that it computes both val and dval correctly. This will be explained in more details later.

We use a hack to determine if there is a discontinuity: check whether the integrand has both "integrand" and "pd" in its name.

## Manual Construction of the Integrand

This is the fwd-diff version of the above **integrand_pd** function, which should be 90% self-explanatory:

```python
def _d_fwd_integrand_pd(x : In[_dfloat], t : In[_dfloat], lower : In[_dfloat], upper : In[_dfloat]) -> _dfloat:
        correct_val : float
        if ((x).val) < ((t).val):
                correct_val = (float)(1.0)
        else:
                correct_val = (float)(0.0)
        correct_dval : float
        if (((lower).val) < ((t).val)) && (((t).val) < ((upper).val)):
                correct_dval = ((float)(1.0)) / (((upper).val) - ((lower).val))
        else:
                correct_dval = (float)(0.0)
        return make__dfloat(correct_val,correct_dval)
```

The only thing worth notice is the `correct_dval = 1.0 / (upper.val - lower.val)` instead of `correct_dval = 1.0`.

The high-level idea is, the derivative of a indicator function is a Dirac Delta, which is non-trivial to represent or handle
in the existing loma infrastructure. Fortunately, Dirac Delta is only a by-product under our "differentiate then discretize/integrate" framework. We integrate this Dirac Delta signal over **t** and get the **[lower < t < upper]**.

Thus, we only aim to produce the correct final result **[lower < t < upper]** only.
Geometrically, a 1D Dirac Delta signal is a rectangle infinitesimally thin and infinitely tall, but has area-under-curve 1. Under the current "simple" condition without reparameterization, it's equivalent to **IntegralEval** of a rectangle with width=(upper-lower) and height=1/(upper-lower).

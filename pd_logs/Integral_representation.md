# Choices of Representing Integral in Loma

*Created after I made Integral work.*

The first step outlined in the project proposal is

> Setup loma infrastructure to handle an integral. This means that something like y =
integrate(x, 0 to pi/2, cos(x)+5k) should be evaluated to a numerically correct value

I explored the following options and chose the custom function representation.

## Intermediate Rrepresentation (IR)

This means adding something like

```txt
IntegralEval(expr a, expr b, expr integrand, expr wrt)
```

to that super long string in **ir.py**. This sounds very promising at first - actually this is what I planned originally - but very difficult, if not impossible, to implement.

The whole reason is not easy to explain. I will explain more in the project report (hopefully). In short, letting Python ADT to recognize *Integral*, this novel type, definition, whatever we call it, is hard.

This part in homework0 writeup could be more explanatory:
> ... loma is embedded in Python. That is, we borrow the Python syntax (but not the semantics), so that we can reuse Python’s parser, and we can exploit on people’s familiarity of Python’s syntax.

Python doesn't have its built-in syntax designed for Integral. Unless I did some very fundamental change, inevitably Python parser will treat the Integral as a Class or Function.

Besides feasibility, adopting this also incurrs more work. This means many loma code needs to be modified, like **irvisitor, type_inference, irmutator**, before we can get Integral to work.

## Custom Struct

This means something like:

```python
class Integral:
    integrand : float
    wrt : float

def eval_integral(a: In[float], b: In[float], integral: In[Integral]) -> float:
    # implementation
```

This has several issues:

- No matter how restricted our *Integral* is (1d, float only, etc.), the integrand must be of **loma_ir.expr** instead of **loma_ir.Var**. In other words, we shouldn't only support **IntegralEval(a=1, b=8, x, x)**. Something as simple as **IntegralEval(a=1, b=8, sin(x)+8x, x)** must be supported at least. And this posts serious questions on how to represent **integrand** or what type it should be. Using **float** is wrong, as just explained. Mathmatically, the integrand (for 1d at least) should actually be a function *f(x)*
- A loma Struct (**Class Integral** in the above python syntax) can only has data members but not member functions. This is a language-specific limitation, and it limits us to give "function property" to the **integrand** member.
- Think one step ahead, after the Integral infrastucture has been set, we take derivative. But what is the meaning of a differential struct, class **_dIntegral**? What about its "diff type" members? Doesn't really make sense.

## Custom Function

To be fair, in terms of conciseness, custom function is better, but not by too much, than custom struct. We still need something like:

```python
def integrand_f(x: In[float]) -> float:
    return 4*sin(x) + 5

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
```

Note that even there is only one **IntegralEval()** needed (probably two after we add in discontinuity), a loma user needs to define N copies of **integrand_f()** if they
need evaluate integral on N different functions.

The biggest advantage is the ease of implementation. Everything is built by existing loma features. Or we can say difficulties in compiler design are turned into user overhead:

- Define a function for each integrand.
- Copy the **IntegralEval()** definition once for every program they write.

which is arguably not too bad.

Another thing to notice is, if lambda function (anonymous function) is supported, the overhead can be greatly reduced.

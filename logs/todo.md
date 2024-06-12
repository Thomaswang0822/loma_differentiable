# TODOs to finish up the project

1. Move IntegralEval() from user code to compiler auto-generated code.
   - current attempt: let compiler/parser note the existence of a `integrand_xxx(...)` fucntion, and automatically inject the "definition" of `Eval_integrand_xxx(lower, upper, ...)`
2. An application
   - something 1D (we only support 1D integral)
   - look at the paper for inspiration
   - better have an visualization (look at how Tzu-Mao uses Matplotlib)
3. Combine those logs and do partial rewrite to make it a project report.
   - write about IntegralEval() internal representation, also briefly explain why some other options are impossible or much harder to implement
   - write about application
   - turn some informal formula in the first 2 logs into formal LaTex
   - make sure LaTex get rendered correctly in the pdf

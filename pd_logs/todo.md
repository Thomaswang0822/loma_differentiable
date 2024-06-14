# TODOs to finish up the project

1. Move IntegralEval() from user code to compiler auto-generated code.
   - [x] current attempt: let compiler/parser note the existence of a `integrand_xxx(...)` fucntion, and automatically inject the "definition" of `eval_xxx(lower, upper, ...)`
   - [x] UPDATE: attempt is successful, both compiling and evaluation are correct.
2. An application
   - [x] something 1D (we only support 1D integral)
   - [x] look at the paper for inspiration
   - [x] UPDATE: apps in Teg look doable, but not compilable. Code in teg_applications are not updated.
   - [x] application + driver code; check GD correct convergence
   - [x] better have an visualization (look at how Tzu-Mao uses Matplotlib)
   - [x] correct the mis-usage of "disk"; it should be "dartboard".
3. Combine those logs and do partial rewrite to make it a project report.
   - [x] write about IntegralEval() internal representation, also briefly explain why some other options are impossible or much harder to implement
   - [x] write about application
   - [x] turn some informal formula in the first 2 logs into formal LaTex
   - [ ] make sure LaTex get rendered correctly on Github

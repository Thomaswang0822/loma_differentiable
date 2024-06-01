import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
# get function name: print(f"{inspect.stack()[0][3]} Check node {node}")
import inspect

def forward_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_fwd : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply forward differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', forward_diff() should return
        def d_square(x : In[_dfloat]) -> _dfloat:
            return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
        where the class _dfloat is
        class _dfloat:
            val : float
            dval : float
        and the function make__dfloat is
        def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
            ret : _dfloat
            ret.val = val
            ret.dval = dval
            return ret

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # data
    ZERO = loma_ir.ConstFloat(0.0)
    ONE = loma_ir.ConstFloat(1.0)

    # static helper functions
    def negate_float(node: loma_ir.expr):
        return loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), node)

    # Class functions: Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node: loma_ir.FunctionDef) -> loma_ir.FunctionDef:
            if "integrand" in node.id and "pd" in node.id:
                # if user defines integrand_pd(x, t), it needs special
                # handle of parametric discontinuity
                # function-name check is a hack.
                return self.mutate_integrand_pd_def(node)
            
            # change all the args to their diff type (float to _dfloat)
            d_args = [
                loma_ir.Arg(arg.id, 
                autodiff.type_to_diff_type(diff_structs, arg.t),
                arg.i)  # same name, same In/Out, float -> _dfloat
                for arg in node.args
            ]
            # rewrite body, need to flatten the list
            new_body = irmutator.flatten([self.mutate_stmt(stmt) for stmt in node.body])
            # change return to its diff type
            d_ret_type = autodiff.type_to_diff_type(diff_structs, node.ret_type)
            return loma_ir.FunctionDef(\
                diff_func_id, d_args, new_body, node.is_simd, d_ret_type, lineno = node.lineno)

        def mutate_return(self, node):
            val, dval = self.mutate_expr(node.val)
            ret_expr = None
            if isinstance(node.val.t, loma_ir.Float):
                ret_expr = loma_ir.Call(
                    'make__dfloat', 
                    [val, dval]
                )
            elif isinstance(node.val.t, (loma_ir.Int, loma_ir.Array, loma_ir.Struct,)):
                ret_expr = val
            else:
                raise TypeError("return expression has wrong type")
            
            return loma_ir.Return(ret_expr, lineno = node.lineno)

        def mutate_declare(self, node):
            d_type = autodiff.type_to_diff_type(diff_structs, node.t)
            # Note that array/struct can, but array/struct-access can't be on LHS, 
            # so no need to mutate target (unlike mutate_assign)
            if node.val is None:
                return loma_ir.Declare(node.target, d_type, None, lineno = node.lineno)
            # optional expression
            opt_expr = None
            rhs_val, rhs_dval = self.mutate_expr(node.val)
            if isinstance(node.val.t, loma_ir.Float):
                opt_expr = loma_ir.Call(
                    'make__dfloat', 
                    [rhs_val, rhs_dval]
                )
            elif isinstance(node.val.t, (loma_ir.Int, loma_ir.Array, loma_ir.Struct,)):
                opt_expr = rhs_val
            else:
                raise TypeError("declared value has wrong type")
            
            return loma_ir.Declare(node.target, d_type, opt_expr, lineno = node.lineno)

        def mutate_assign(self, node):
            assert (node.val is not None)
            val, dval = self.mutate_expr(node.val)
            # lhs need special handle for array access:
            #   y[float2int(j)] -> y[float2int((j).val)], but not y[float2int((j).val)].val
            target = node.target
            if isinstance(node.target, loma_ir.ArrayAccess):
                # mutate index
                idx, _ = self.mutate_expr(node.target.index)
                target = loma_ir.ArrayAccess(
                    node.target.array,
                    idx,
                    lineno = node.lineno,
                    t = node.target.t)
            # right-hand-side expression
            rhs_expr = None
            rhs_val, rhs_dval = self.mutate_expr(node.val)
            if isinstance(node.val.t, loma_ir.Float):
                rhs_expr = loma_ir.Call(
                    'make__dfloat', 
                    [val, dval]
                )
            elif isinstance(node.val.t, (loma_ir.Int, loma_ir.Array, loma_ir.Struct,)):
                rhs_expr = rhs_val
            else:
                raise TypeError("assigned value has wrong type")
            return loma_ir.Assign(target, rhs_expr, lineno = node.lineno)

        def mutate_ifelse(self, node: loma_ir.IfElse) -> loma_ir.IfElse:
            return super().mutate_ifelse(node)

        def mutate_while(self, node):
            return super().mutate_while(node)

        """2.0 -> (2.0, 0.0)
        """
        def mutate_const_float(self, node):
            return node, loma_ir.ConstFloat(0.0)

        def mutate_const_int(self, node):
            return node, loma_ir.ConstFloat(0.0)

        def mutate_var(self, node) -> tuple[loma_ir.expr]:
            if isinstance(node.t, loma_ir.Float):
                return loma_ir.StructAccess(node, 'val'), loma_ir.StructAccess(node, 'dval')
            elif isinstance(node.t, (loma_ir.Int, loma_ir.Array, loma_ir.Struct,)):
                return node, loma_ir.ConstFloat(0.0)
            else:
                raise TypeError("mutate_var() node.t has wrong type")

        def mutate_array_access(self, node):
            """arr[i+j] -> (arr[i+j].val, arr[i+j].val)
            """
            # index may not be trival: arr[x + 3 - 2*5], where
            #   x is a float
            # Thus, we need sth like arr[x.dval + 3 - 2*5] in the diff code
            idx, _ = self.mutate_expr(node.index)
            accessed_element = loma_ir.ArrayAccess(
                node.array,
                idx,
                lineno = node.lineno,
                t = node.t)
            return self.mutate_var(accessed_element)

        def mutate_struct_access(self, node):
            """
            Key: struct access operator (.) is followed by struct member name, which won't
            be changed in diff code.
            We only need to treat it like a var, e.g.
            - school.location.x -> (school.location.x.val, school.location.x.dval)
            - school.location (assume it has struct tupe Position) -> school.location
            """
            return self.mutate_var(node)

        """x+y -> make__dfloat(x.val + y.val, x.dval + y.dval)
        """
        def mutate_add(self, node):
            lval, rval, ldval, rdval = self.extract_binary_val_dval(node)
            # construct "x.val + y.val" and "x.dval + y.dval"
            val = loma_ir.BinaryOp(loma_ir.Add(), lval, rval)
            dval = loma_ir.BinaryOp(loma_ir.Add(), ldval, rdval)
            return val, dval

        """x-y -> make__dfloat(x.val - y.val, x.dval - y.dval)
        """
        def mutate_sub(self, node):
            lval, rval, ldval, rdval = self.extract_binary_val_dval(node)
            # construct "x.val - y.val" and "x.dval - y.dval"
            val = loma_ir.BinaryOp(loma_ir.Sub(), lval, rval)
            dval = loma_ir.BinaryOp(loma_ir.Sub(), ldval, rdval)
            return val, dval

        """x*y -> make__dfloat(x.val * y.val, x.dval * y.val + x.val * y.dval)
        """
        def mutate_mul(self, node):
            lval, rval, ldval, rdval = self.extract_binary_val_dval(node)
            # construct "x.val + y.val" and "x.dval * y.val + x.val * y.dval"
            val = loma_ir.BinaryOp(loma_ir.Mul(), lval, rval)
            dval = loma_ir.BinaryOp(
                loma_ir.Add(), 
                loma_ir.BinaryOp(loma_ir.Mul(), ldval, rval), 
                loma_ir.BinaryOp(loma_ir.Mul(), lval, rdval)
            )
            return val, dval

        """x/y -> make__dfloat(x.val / y.val, (x.dval * y.val - x.val * y.dval)/(y.val * y.val) )
        """
        def mutate_div(self, node):
            lval, rval, ldval, rdval = self.extract_binary_val_dval(node)
            # construct "x.val / y.val" 
            val = loma_ir.BinaryOp(loma_ir.Div(), lval, rval)
            # and "(x.dval * y.val - x.val * y.dval)/(y.val * y.val)"
            numerator = loma_ir.BinaryOp(
                loma_ir.Sub(), 
                loma_ir.BinaryOp(loma_ir.Mul(), ldval, rval), 
                loma_ir.BinaryOp(loma_ir.Mul(), lval, rdval)
            )
            denom = loma_ir.BinaryOp(loma_ir.Mul(), rval, rval)
            dval = loma_ir.BinaryOp(
                loma_ir.Div(), 
                numerator, 
                denom
            )
            return val, dval

        def mutate_less(self, node):
            # don't care about dval during compare
            lval, rval, _, _ = self.extract_binary_val_dval(node)
            return loma_ir.BinaryOp(
                loma_ir.Less(), lval, rval,
                lineno=node.lineno, t=node.t
            )

        def mutate_less_equal(self, node):
            # don't care about dval during compare
            lval, rval, _, _ = self.extract_binary_val_dval(node)
            return loma_ir.BinaryOp(
                loma_ir.LessEqual(), lval, rval,
                lineno=node.lineno, t=node.t
            )

        def mutate_greater(self, node):
            # don't care about dval during compare
            lval, rval, _, _ = self.extract_binary_val_dval(node)
            return loma_ir.BinaryOp(
                loma_ir.Greater(), lval, rval,
                lineno=node.lineno, t=node.t
            )

        def mutate_greater_equal(self, node):
            # don't care about dval during compare
            lval, rval, _, _ = self.extract_binary_val_dval(node)
            return loma_ir.BinaryOp(
                loma_ir.GreaterEqual(), lval, rval,
                lineno=node.lineno, t=node.t
            )

        def mutate_equal(self, node):
            # don't care about dval during compare
            lval, rval, _, _ = self.extract_binary_val_dval(node)
            return loma_ir.BinaryOp(
                loma_ir.Equal(), lval, rval,
                lineno=node.lineno, t=node.t
            )

        """handle intrinsic function calls (sin, exp, etc.)
        and let others pass by.
        """
        def mutate_call(self, node: loma_ir.Call):
            # these intrinsic functions have at least 1 arg, 
            # (pow has 2) we call it x (and y)
            if len(node.args) == 0:
                return self.mutate_custom_call_fwd(node)
            x_val, x_dval = self.mutate_expr(node.args[0])
            # for returned d_float
            val = loma_ir.Call(node.id, [x_val]) if node.id != "pow" else None
            dval = None  # to be filled in swtich
            match node.id:
                case "sin":
                    dval = loma_ir.BinaryOp(
                        loma_ir.Mul(), 
                        loma_ir.Call("cos", [x_val]), 
                        x_dval
                    )  #  cos(x.val) * x.dval
                case "cos":
                    sinx_dx = loma_ir.BinaryOp(
                        loma_ir.Mul(), 
                        loma_ir.Call("sin", [x_val]), 
                        x_dval
                    )  #  sin(x.val) * x.dval
                    #  0 - sin(x.val) * x.dval
                    dval = loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), sinx_dx)
                case "sqrt":
                    dval = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.BinaryOp(
                            loma_ir.Div(),
                            loma_ir.ConstFloat(0.5), 
                            loma_ir.Call("sqrt", [x_val])
                        ),
                        x_dval
                    )  # (0.5 / sqrt(x)) * x_dval
                case "pow":
                    assert len(node.args) == 2, "pow() should have 2 args"
                    y_val, y_dval = self.mutate_expr(node.args[1])
                    # d/dx(x^y) = y * x^(y-1)
                    # d/dy(x^y) = x^y * log(x)
                    partial_x = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        y_val,
                        loma_ir.Call(
                            "pow", [
                                x_val,
                                loma_ir.BinaryOp(
                                    loma_ir.Sub(),
                                    y_val,
                                    loma_ir.ConstFloat(1.0), 
                                )  # y-1
                            ]
                        )
                    )  # y * x^(y-1)
                    partial_y = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.Call(
                            "log",
                            [x_val]
                        ),  # log(x)
                        loma_ir.Call(
                            "pow", [
                                x_val,
                                y_val
                            ]
                        ) # x^y
                    )  # log(x) * x^y
                    dval = loma_ir.BinaryOp(
                        loma_ir.Add(),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            x_dval, 
                            partial_x
                        ),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            y_dval, 
                            partial_y
                        ),
                    )  # partial_x * dx + partial_y * dy
                    # val is also different since we need one more arg
                    val = loma_ir.Call(node.id, [x_val, y_val])
                case "exp":
                    dval = loma_ir.BinaryOp(
                        loma_ir.Mul(), 
                        loma_ir.Call("exp", [x_val]), 
                        x_dval
                    )  #  exp(x.val) * x.dval
                case "log":
                    dval = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.BinaryOp(
                            loma_ir.Div(),
                            loma_ir.ConstFloat(1.0), 
                            x_val
                        ),
                        x_dval
                    )  # (1 / x) * x_dval
                case "int2float":
                    dval = loma_ir.ConstFloat(0.0)
                case "float2int":
                    dval = loma_ir.ConstFloat(0.0)
                case _:
                    # non-intrinsic function with >=0 args
                    return self.mutate_custom_call_fwd(node)
            # return for the intrinsic
            return val, dval

        """helper function that extract the val and dval for
        the 2 operands in a binary op
        """
        def extract_binary_val_dval(self, node):
            # get x and y
            lval, ldval = self.mutate_expr(node.left)
            rval, rdval = self.mutate_expr(node.right)
            return (lval, rval, ldval, rdval)

        def extract_ifelse_condition(self, node: loma_ir.BinaryOp,
                arg_x: loma_ir.Arg, arg_t: loma_ir.Arg):
            """Given a general condition in the form
            (mx + n) > (kt + p), we extract (also make declaration) of these
            m, n, k, p coefficients
            See report for derivation

            In addition, we return the "ratio" term, which is -k/m
            NOTE:
                m, n, k, p can hold negative values, and we want to fit to the exact form
                above. If the operator is <, we negate all these 4 coefficients.

            Returns:
                1. declare_stmts: list[loma_ir.stmt], declaration of m,n,k,p, and -k/m
                2. cond_tmp_Vars: list[loma_ir.expr], their respective expresion
                3. gt: bool, whether the operator is >, >= or <, <=
            """
            WRONG_FORMAT = f"IFElse condition in the integrand must have format (mx + n) [>, <] (kt + p)\n" +\
                f"Note that x should be on the left and t on the right"
            PLUS_MINUS = (loma_ir.Add, loma_ir.Sub)
            COMPARISONS = (loma_ir.Less, loma_ir.LessEqual, loma_ir.Greater, loma_ir.GreaterEqual)
            assert isinstance(node, loma_ir.BinaryOp) and isinstance(node.op, COMPARISONS), WRONG_FORMAT
            # if less, we negate all 4 coefficients
            less: bool = isinstance(node.op, (loma_ir.Less, loma_ir.LessEqual))

            declare_stmts = []

            # Decode LHS: mx+n, note the case m=1 or n=0
            LHS = node.left
            _m, _n = loma_ir.Var("_m", t=loma_ir.Float()), loma_ir.Var("_n", t=loma_ir.Float())
            _m_expr: loma_ir.expr = None
            _n_expr: loma_ir.expr = None
            _minus_n = False
            if isinstance(LHS, loma_ir.Var):
                # x
                assert LHS.id == arg_x.id, WRONG_FORMAT
                _m_expr = ONE
                _n_expr = ZERO
            elif isinstance(LHS, loma_ir.BinaryOp):
                if isinstance(LHS.op, loma_ir.Mul):
                    # m*x
                    assert isinstance(LHS.right, loma_ir.Var) \
                        and LHS.right.id == arg_x.id, WRONG_FORMAT
                    _m_expr, _ = self.mutate_expr(LHS.left)
                    _n_expr = ZERO
                elif isinstance(LHS.left, loma_ir.Var) and LHS.left.id == arg_x.id:
                    # x +/- n
                    _m_expr = ONE
                    assert isinstance(LHS.op, PLUS_MINUS), WRONG_FORMAT
                    _minus_n = isinstance(LHS.op, loma_ir.Sub)
                    _n_expr, _ = self.mutate_expr(LHS.right)
                elif isinstance(LHS.left.op, loma_ir.Mul):
                    _mx = LHS.left
                    # m*x +/- n
                    assert isinstance(_mx.right, loma_ir.Var) \
                        and _mx.right.id == arg_x.id, WRONG_FORMAT
                    _m_expr, _ = self.mutate_expr(_mx.left)
                    assert isinstance(LHS.op, PLUS_MINUS), WRONG_FORMAT
                    _minus_n = isinstance(LHS.op, loma_ir.Sub)
                    _n_expr, _ = self.mutate_expr(LHS.right)
            else:
                # something we couldn't or shouldn't handle, like 3.5 > kt+p
                assert False, WRONG_FORMAT
            # negate n and m if necessary
            if less:
                _m_expr = negate_float(_m_expr)
            if less ^ _minus_n:
                _n_expr = negate_float(_n_expr)
            # Store Declare()
            declare_stmts += [
                loma_ir.Declare(_m.id, t=loma_ir.Float(), val=_m_expr),
                loma_ir.Declare(_n.id, t=loma_ir.Float(), val=_n_expr)
            ]

            # Decode RHS: kt + p, note the case k=1 or p=0
            RHS = node.right
            _k, _p = loma_ir.Var("_k", t=loma_ir.Float()), loma_ir.Var("_p", t=loma_ir.Float())
            _k_expr: loma_ir.expr = None
            _p_expr: loma_ir.expr = None
            _minus_p = False
            if isinstance(RHS, loma_ir.Var):
                # t
                assert RHS.id == arg_t.id, WRONG_FORMAT
                _k_expr = ONE
                _p_expr = ZERO
            elif isinstance(RHS, loma_ir.BinaryOp):
                if isinstance(RHS.op, loma_ir.Mul):
                    # k*t
                    assert isinstance(RHS.right, loma_ir.Var) \
                        and RHS.right.id == arg_t.id, WRONG_FORMAT
                    _k_expr, _ = self.mutate_expr(RHS.left)
                    _p_expr = ZERO
                elif isinstance(RHS.left, loma_ir.Var) and RHS.left.id == arg_t.id:
                    # t +/- p
                    _k_expr = ONE
                    assert isinstance(RHS.op, PLUS_MINUS), WRONG_FORMAT
                    _minus_p = isinstance(RHS.op, loma_ir.Sub)
                    _p_expr, _ = self.mutate_expr(RHS.right)
                elif isinstance(RHS.left.op, loma_ir.Mul):
                    _kt = RHS.left
                    # k*t +/- p
                    assert isinstance(_kt.right, loma_ir.Var) \
                        and _kt.right.id == arg_t.id, WRONG_FORMAT
                    _k_expr, _ = self.mutate_expr(_kt.left)
                    assert isinstance(RHS.op, PLUS_MINUS), WRONG_FORMAT
                    _minus_p = isinstance(RHS.op, loma_ir.Sub)
                    _p_expr, _ = self.mutate_expr(RHS.right)
            else:
                # something we couldn't or shouldn't handle, like 3.5 > kt+p
                assert False, WRONG_FORMAT
            # negate k and p if necessary
            if less:
                _k_expr = negate_float(_k_expr)
            if less ^ _minus_p:
                _p_expr = negate_float(_p_expr)
            # Store Declare()
            declare_stmts += [
                loma_ir.Declare(_k.id, t=loma_ir.Float(), val=_k_expr),
                loma_ir.Declare(_p.id, t=loma_ir.Float(), val=_p_expr)
            ]

            # post process, construct the ratio -k/m
            _ratio = loma_ir.Var("_ratio", t=loma_ir.Float())
            _ratio_expr = negate_float(loma_ir.BinaryOp(
                loma_ir.Div(), 
                _k, 
                loma_ir.Call("abs", [_m])
            ))  # -k/abs(m)
            declare_stmts.append(loma_ir.Declare(_ratio.id, t=loma_ir.Float(), val=_ratio_expr))

            return declare_stmts, [_m, _n, _k, _p, _ratio]

        def reparam_lower_upper(
                self, bd: loma_ir.expr, 
                cond_tmp_Vars: list[loma_ir.Var],
                param_t: loma_ir.expr
            ):
            """Before reparam, lower=a and upper=b;
            After reparam, lower=R(a) and upper=R(b).

            R(x) = mx - kt + n - p,
            so here we compute R(bd)

            Args:
                bd (loma_ir.expr): original bound in float, lower.val or upper.val
                cond_tmp_Vars (list[loma_ir.Var]): [_m, _n, _k, _p, _ratio]
                param_t: (loma_ir.expr): t.val
            """
            _m, _n, _k, _p, _ratio = cond_tmp_Vars
            mx_kt = loma_ir.BinaryOp(
                loma_ir.Sub(),
                loma_ir.BinaryOp(loma_ir.Mul(), _m, bd),
                loma_ir.BinaryOp(loma_ir.Mul(), _k, param_t),
                t=loma_ir.Float()
            )  # mx - kt
            return loma_ir.BinaryOp(
                loma_ir.Add(),
                left=mx_kt,
                right=loma_ir.BinaryOp(loma_ir.Sub(), _n, _p),
                t=loma_ir.Float()
            )


        def mutate_integrand_pd_def(self, node: loma_ir.FunctionDef) -> loma_ir.FunctionDef:
            # Starting from very restrictive
            assert len(node.args) == 2, "an integrand func with parametric discontinuity must have 2 args"
            arg_x, arg_t = node.args
            assert isinstance(arg_x.t, loma_ir.Float), "integration var x must be float"
            assert isinstance(arg_t.t, loma_ir.Float), "parameter t must be float"
            assert isinstance(node.ret_type, loma_ir.Float), "integrand must return float"
            assert isinstance(node.body[0], loma_ir.IfElse), "For now, nothing before if-else"

            # add 2 more args to hold lower and upper
            new_args = list(node.args) + [
                loma_ir.Arg("lower", t=loma_ir.Float(), i=loma_ir.In()),
                loma_ir.Arg("upper", t=loma_ir.Float(), i=loma_ir.In()),
            ]
            d_args = [
                loma_ir.Arg(arg.id, 
                autodiff.type_to_diff_type(diff_structs, arg.t),
                arg.i)  # same name, same In/Out, float -> _dfloat
                for arg in new_args
            ]
            # change return to its diff type, just a _dfloat
            d_ret_type = autodiff.type_to_diff_type(diff_structs, node.ret_type)

            # These Var are all _dfloat, so need to access .val
            lower = loma_ir.StructAccess(loma_ir.Var("lower"), "val")
            upper = loma_ir.StructAccess(loma_ir.Var("upper"), "val")
            param_t = loma_ir.StructAccess(loma_ir.Var(arg_t.id), "val")
            new_body = []
            for stmt in node.body:
                if isinstance(stmt, loma_ir.IfElse):
                    # primal value pass (compute .val, leave .dval incorrectly)
                    new_body += self.mutate_discont_val(stmt)
                    # diff value pass (reparametrize, compute .dval correctly)
                    new_body += self.mutate_discont_dval(stmt, arg_x, arg_t, param_t, lower, upper)
                else:
                    new_body += [self.mutate_stmt(stmt)]
            new_body = irmutator.flatten(new_body)
            # At last, make the defloat and return
            correct_val = loma_ir.Var("correct_val", t=loma_ir.Float())
            correct_dval = loma_ir.Var("correct_dval", t=loma_ir.Float())
            new_body.append(loma_ir.Return(
                loma_ir.Call(
                    "make__dfloat", args=[correct_val, correct_dval])
            ))

            return loma_ir.FunctionDef(\
                diff_func_id, d_args, new_body, node.is_simd, d_ret_type, lineno = node.lineno)        

        def mutate_discont_val(self, node: loma_ir.IfElse) -> list[loma_ir.stmt]:
            # Prepare: assume a single Return under each branch
            assert len(node.then_stmts) == 1 and isinstance(node.then_stmts[0], loma_ir.Return)
            assert len(node.else_stmts) == 1 and isinstance(node.else_stmts[0], loma_ir.Return)
            # These 2 are expr, likely Var
            ret_then_expr = node.then_stmts[0].val
            ret_else_expr = node.else_stmts[0].val
            # Mutated expr (Var), leave dval alone
            correct_then, _ = self.mutate_expr(ret_then_expr)
            correct_else, _ = self.mutate_expr(ret_else_expr)
            dfloat_type = diff_structs['float']
            stmts = []  # returned

            # Declare the new Var to store .val of return value
            correct_val = loma_ir.Var("correct_val", t=loma_ir.Float())
            stmts.append(loma_ir.Declare("correct_val", t=loma_ir.Float()))

            # create a new IfElse. Instead of return, store to correct_val
            newIE = loma_ir.IfElse(
                self.mutate_expr(node.cond),
                then_stmts=[loma_ir.Assign(correct_val, correct_then)],
                else_stmts=[loma_ir.Assign(correct_val, correct_else)]
            )
            stmts.append(newIE)

            return stmts

        def mutate_discont_dval(
            self, node: loma_ir.IfElse, arg_x: loma_ir.Arg, arg_t: loma_ir.Arg,
            param_t: loma_ir.Var, lower: loma_ir.Var, upper: loma_ir.Var
        ) -> list[loma_ir.stmt]:
            
            # Prepare: assume a single Return under each branch
            assert len(node.then_stmts) == 1 and isinstance(node.then_stmts[0], loma_ir.Return)
            assert len(node.else_stmts) == 1 and isinstance(node.else_stmts[0], loma_ir.Return)

            dfloat_type = diff_structs['float']
            stmts = []  # returned

            # Declare the new Var to store .dval of return value
            correct_dval = loma_ir.Var("correct_dval", t=loma_ir.Float())
            stmts.append(loma_ir.Declare("correct_dval", t=loma_ir.Float()))

            # UPDATE: process condition using reparam
            cond_stmts, cond_tmp_Vars = self.extract_ifelse_condition(node.cond, arg_x, arg_t)
            _m = cond_tmp_Vars[0]
            stmts += cond_stmts
            # and declare R(lower) and R(upper)
            _R_lower = loma_ir.Var("_R_lower", t=loma_ir.Float())
            _R_upper = loma_ir.Var("_R_upper", t=loma_ir.Float())
            stmts += [
                loma_ir.Declare(
                    _R_lower.id, 
                    t=loma_ir.Float(), 
                    val=self.reparam_lower_upper(lower, cond_tmp_Vars, param_t)
                ),
                loma_ir.Declare(
                    _R_upper.id, 
                    t=loma_ir.Float(), 
                    val=self.reparam_lower_upper(upper, cond_tmp_Vars, param_t)
                )
            ]
            # and construct new condition (for derivative computation only)
            
            # if _m is negative (can't be deterimined during compile), 
            # the final interval becomes [R(a) > 0 > R(b)]
            # Thus, we can check equivalent condition [m*R(a) < 0 < m*R(b)]
            lower_reparam = loma_ir.BinaryOp(loma_ir.Mul(), _m, _R_lower)
            upper_reparam = loma_ir.BinaryOp(loma_ir.Mul(), _m, _R_upper)
            new_cond = loma_ir.BinaryOp(
                loma_ir.And(),
                left=loma_ir.BinaryOp(
                    loma_ir.Less(), left=lower_reparam, right=ZERO
                ),
                right=loma_ir.BinaryOp(
                    loma_ir.Less(), left=ZERO, right=upper_reparam
                )
            )  # [m*R(lower) < 0 < m*R(upper)]
            
            
            """ # change condition to: t > lower and t < upper
            new_cond = loma_ir.BinaryOp(
                loma_ir.And(),
                left=loma_ir.BinaryOp(
                    loma_ir.Less(), left=lower, right=param_t
                ),  # lower < t
                right=loma_ir.BinaryOp(
                    loma_ir.Less(), left=param_t, right=upper
                )  # t < upper
            ) """

            # compute correct dval value
            # UPDATE: after reparam, need to multiply with the _ratio
            _ratio = cond_tmp_Vars[-1]
            correct_then: loma_ir.expr = loma_ir.BinaryOp(
                loma_ir.Div(),
                left=_ratio,
                right=loma_ir.BinaryOp(loma_ir.Sub(), upper, lower),
                t=loma_ir.Float()
            )  # mimic dirac delta, see doc
            correct_else = loma_ir.ConstFloat(0.0)

            # reconstruct if-else
            newIE = loma_ir.IfElse(
                new_cond,
                then_stmts=[loma_ir.Assign(correct_dval, correct_then)],
                else_stmts=[loma_ir.Assign(correct_dval, correct_else)]
            )
            stmts.append(newIE)

            return stmts
            
        def mutate_integrand_pd_call(self, node: loma_ir.Call) -> tuple[loma_ir.expr]:
            """Handle Call to parametric discontinuous integrand func, which happens in IntegralEval()
            This means call to integrand_pd(curr_x ,t) in primal code will be turned to 
            fwd_integrand_f(curr_x, t, lower, upper)

            Args:
                node (loma_ir.Call): _description_

            Returns:
                tuple[loma_ir.expr]: _description_
            """
            # some check first
            assert len(node.args) == 2, "integrand should have 2 args"

            # add 2 more args, which have fixed name 'lower' and 'upper'
            dfloat_type = diff_structs['float']
            lower = loma_ir.Var("lower", t=dfloat_type)
            upper = loma_ir.Var("upper", t=dfloat_type)
            d_args = list(node.args) + [lower, upper]

            # function name
            d_func_name = func_to_fwd[node.id]

            # return
            d_call: loma_ir.Call = loma_ir.Call(
                d_func_name, d_args,
                lineno=node.lineno, t=dfloat_type
            )
            return loma_ir.StructAccess(d_call, 'val'), loma_ir.StructAccess(d_call, 'dval')
        
        def mutate_custom_call_fwd(self, node: loma_ir.Call) -> tuple[loma_ir.expr]:
            if "integrand" in node.id and "pd" in node.id:
                # FROM integrand_pd(curr_x, t)
                # TO _d_fwd_integrand_pd(curr_x, t, lower, upper)
                return self.mutate_integrand_pd_call(node)
       
            d_func_name = func_to_fwd[node.id]
            d_ret_type = autodiff.type_to_diff_type(diff_structs, node.t)

            # grab In/Out of the arguments
            actual_func: loma_ir.FunctionDef = funcs[node.id]
            inouts: list[loma_ir.inout] = [arg.i for arg in actual_func.args]
            assert len(inouts) == len(node.args)
            
            # process args:
            # (x, y) -> make__dfloat(x.val, x.dval), make__dfloat(y.val, y.dval)
            d_args: list[loma_ir.expr] = []
            for arg_expr, io in zip(node.args, inouts):
                arg_expr: loma_ir.expr
                # y shouldn't be make__dfloat(y.val, y.dval) if it's an Out
                if io == loma_ir.Out():
                    d_args.append(arg_expr)
                    continue
                
                val, dval = self.mutate_expr(arg_expr)
                if isinstance(arg_expr.t, loma_ir.Float):
                    d_args.append(loma_ir.Call(
                        'make__dfloat', 
                        [val, dval]
                    ))
                elif isinstance(arg_expr.t, (loma_ir.Int, loma_ir.Array, loma_ir.Struct,)):
                    d_args.append(val)
                else:
                    raise TypeError("mutate_var() node.t has wrong type")
            # create the Call expr:
            # _d_fwd_foo(make__dfloat(x.val, x.dval), make__dfloat(y.val, y.dval))
            d_call: loma_ir.Call = loma_ir.Call(
                d_func_name, d_args,
                lineno=node.lineno, t=d_ret_type
            )
            # if foo(x, y) returns a float, _d_fwd_foo() will return a _dfloat
            if isinstance(node.t, loma_ir.Float):
                return loma_ir.StructAccess(d_call, 'val'), loma_ir.StructAccess(d_call, 'dval')
            elif isinstance(node.t, (loma_ir.Int, loma_ir.Array, loma_ir.Struct,)):
                return d_call, None
            # but for foo(x : In[float], y : Out[float]), node.t is None
            else:
                # assert False, f"node.t has wrong type: {node.t}"
                return d_call

    return FwdDiffMutator().mutate_function_def(func)

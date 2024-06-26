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

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
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

        def mutate_custom_call_fwd(self, node: loma_ir.Call) -> tuple[loma_ir.expr]:
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

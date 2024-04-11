import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import inspect  # inspect.stack()[0][3] get function name

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
            val, dval = super().mutate_expr(node.val)
            return loma_ir.Return(
                loma_ir.Call(
                    'make__dfloat', 
                    [val, dval]
                ),
                lineno = node.lineno)

        def mutate_declare(self, node):
            # optional expression
            opt_expr = loma_ir.Call(
                'make__dfloat', 
                self.mutate_expr(node.val)
            ) if (node.val is not None) else None
            return loma_ir.Declare(
                node.target,
                autodiff.type_to_diff_type(diff_structs, node.t),
                opt_expr,
                lineno = node.lineno)

        def mutate_assign(self, node):
            assert (node.val is not None)
            # right-hand-side expression
            val, dval = self.mutate_expr(node.val)
            rhs_expr = loma_ir.Call('make__dfloat', [val, dval])
            return loma_ir.Assign(
                node.target,
                rhs_expr,
                lineno = node.lineno)

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        """2.0 -> make__dfloat(2.0, 0.0)
        """
        def mutate_const_float(self, node):
            # return loma_ir.Call(
            #     'make__dfloat', 
            #     [node, loma_ir.ConstFloat(0.0)]
            # )
            return node, loma_ir.ConstFloat(0.0)

        def mutate_const_int(self, node):
            # HW1: TODO
            return super().mutate_const_int(node)

        def mutate_var(self, node):
            return loma_ir.StructAccess(node, 'val'), loma_ir.StructAccess(node, 'dval')

        def mutate_array_access(self, node):
            # HW1: TODO
            return super().mutate_array_access(node)

        def mutate_struct_access(self, node):
            # HW1: TODO
            return super().mutate_struct_access(node)

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

        """handle intrinsic function calls (sin, exp, etc.)
        and let others pass by.
        """
        def mutate_call(self, node):
            # these intrinsic functions have at least 1 arg, 
            # (pow has 2) we call it x (and y)
            if len(node.args) == 0:
                return super().mutate_call(node)
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
                case _:
                    # non-intrinsic function with >=0 args
                    return super().mutate_call(node)
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

    return FwdDiffMutator().mutate_function_def(func)

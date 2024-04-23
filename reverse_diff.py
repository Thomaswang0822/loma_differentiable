import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random

# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_id_generator(size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def reverse_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_rev : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply reverse differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', reverse_diff() should return
        def d_square(x : In[float], _dx : Out[float], _dreturn : float):
            _dx = _dx + _dreturn * x + _dreturn * x

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
        func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return 'int'
            case loma_ir.Float():
                return 'float'
            case loma_ir.Array():
                return 'array_' + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False

    def accum_deriv(target: loma_ir.expr, deriv: loma_ir.expr, overwrite: bool) -> list[loma_ir.stmt]:
        """E.g. r = x * y ->
        accum_deriv(dx, dr*y, False) gives: dx += dr * y
        accum_deriv(dy, dr*x, False) gives: dy += dr * x

        Args:
            target (loma_ir.expr): lhs derivative to be accumulated
            deriv (loma_ir.expr): rhs value
            overwrite (bool): whether dx = dr * y or dx += dr * y

        Returns:
            list[loma_ir.stmt]: list of statements depending on how many var in the primal code
        """
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    return [loma_ir.Assign(target,
                        loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    deriv_m = loma_ir.StructAccess(
                        deriv, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False

    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or 
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(\
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno = node.lineno)

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(\
                val,
                lineno = node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Declare(\
                node.target,
                node.t,
                val,
                lineno = node.lineno)]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Assign(\
                target,
                val,
                lineno = node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(\
                call,
                lineno = node.lineno)]

        def mutate_call(self, node):
            new_args = []
            for arg in node.args:
                if not isinstance(arg, loma_ir.Var) and \
                        not isinstance(arg, loma_ir.ArrayAccess) and \
                        not isinstance(arg, loma_ir.StructAccess):
                    arg = self.mutate_expr(arg)
                    tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(\
                        tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(\
                        tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            return loma_ir.Call(node.id, new_args, t = node.t)

    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.

    """It needs to
    Add declare of _dx after (existing) declare of x
    """
    class PrimalCodeMutator(irmutator.IRMutator):
        def mutate_stmt(self, node):
            match node:
                case loma_ir.Return():
                    # hide original return
                    return []
                case loma_ir.Declare():
                    return self.mutate_declare(node)
                case loma_ir.Assign():
                    []
                case loma_ir.IfElse():
                    []
                case loma_ir.While():
                    []
                case loma_ir.CallStmt():
                    []
                case _:
                    assert False, f'Visitor error: unhandled statement {node}'

        def mutate_declare(self, node):
            # automatically initialized to zero if no val
            diff_declare = loma_ir.Declare('_d' + node.target, t=node.t)  
            
            return [node, diff_declare]

    # Apply the differentiation.
    class RevDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW2: TODO
            # Signature (args)
            new_args = []
            for arg in node.args:
                if isinstance(arg.i, loma_ir.In):
                    new_args.append(arg)
                    # also _dx of x
                    darg = loma_ir.Arg(
                        '_d' + arg.id, arg.t, loma_ir.Out()
                    )
                    new_args.append(darg)
                else:
                    assert False, "NOT IMPLEMENTED"
            
            # _dreturn as the start point of rev diff
            if node.ret_type is not None:
                new_args.append(loma_ir.Arg('_dreturn', node.ret_type, loma_ir.In()))

            # copy paste forward code
            fwd_new_body = irmutator.flatten( [PrimalCodeMutator().mutate_stmt(stmt) for stmt in node.body] )
            

            # backward diff
            rev_new_body = irmutator.flatten( [self.mutate_stmt(stmt) for stmt in reversed(node.body)] )

            body = fwd_new_body + rev_new_body

            return loma_ir.FunctionDef(diff_func_id, new_args, body, node.is_simd, ret_type=None)



        def mutate_return(self, node):
            # HW2: TODO
            # in bwd part, mutate_return should be the first to execute,
            # set global adjoint s.t. callee can use
            self.adjoint = loma_ir.Var('_dreturn')
            stmts = self.mutate_expr(node.val)  # is a list
            self.adjoint = None

            return stmts

        def mutate_declare(self, node):
            self.adjoint = loma_ir.Var('_d' + node.target, t=node.t)
            stmts = self.mutate_expr(node.val)  # is a list
            self.adjoint = None
            return stmts

        def mutate_assign(self, node):
            # HW2: TODO
            return super().mutate_assign(node)

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_call_stmt(self, node):
            # HW3: TODO
            return super().mutate_call_stmt(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            return []

        def mutate_const_int(self, node):
            # HW2: TODO
            return super().mutate_const_int(node)

        def mutate_var(self, node: loma_ir.Var) -> list[loma_ir.stmt]:
            """ x -> _dx = _dx + adjoint

            Args:
                node (loma_ir.Var): x

            Returns:
                list[loma_ir.stmt]: _dx = _dx + adjoint
            """
            dx = loma_ir.Var('_d' + node.id, lineno=node.lineno, t=node.t)
            return accum_deriv(dx, self.adjoint, overwrite=False)

        def mutate_array_access(self, node):
            # HW2: TODO
            return super().mutate_array_access(node)

        def mutate_struct_access(self, node):
            # HW2: TODO
            return super().mutate_struct_access(node)

        def mutate_add(self, node: loma_ir.Add) -> list[loma_ir.stmt]:
            """f(x, y) = x + y ->
            _dx += _dreturn
            _dy += _dreturn

            Args:
                node (loma_ir.Add)

            Returns:
                list[loma_ir.stmt]
            """
            left_stmt = self.mutate_expr(node.left)
            right_stmt = self.mutate_expr(node.right)
            return left_stmt + right_stmt

        def mutate_sub(self, node: loma_ir.Sub) -> list[loma_ir.stmt]:
            """f(x, y) = x - y ->
            _dx += _dreturn
            _dy -= _dreturn

            Args:
                node (loma_ir.Sub)

            Returns:
                list[loma_ir.stmt]
            """
            left_stmt = self.mutate_expr(node.left)
            # - _dreturn <=> + (0.0 - _dreturn)
            orig_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(
                loma_ir.Sub(), loma_ir.ConstFloat(0.0), self.adjoint )
            right_stmt = self.mutate_expr(node.right)
            self.adjoint = orig_adjoint
            return left_stmt + right_stmt

        def mutate_mul(self, node: loma_ir.Mul) -> list[loma_ir.stmt]:
            """f(x, y) = x * y ->
            _dx += _dreturn * y
            _dy += _dreturn * x
            Args:
                node (loma_ir.Mul)

            Returns:
                list[loma_ir.stmt]
            """
            # store adjoint
            orig_adjoint = self.adjoint
            # deal with left, x
            self.adjoint = loma_ir.BinaryOp(
                loma_ir.Mul(), self.adjoint, node.right 
            )  # _dreturn * y
            left_stmt = self.mutate_expr(node.left)
            self.adjoint = orig_adjoint
            # deal with right, y
            self.adjoint = loma_ir.BinaryOp(
                loma_ir.Mul(), self.adjoint, node.left 
            )  # _dreturn * x
            right_stmt = self.mutate_expr(node.right)
            self.adjoint = orig_adjoint

            return left_stmt + right_stmt

        def mutate_div(self, node: loma_ir.Div) -> list[loma_ir.stmt]:
            """f(x, y) = x * y ->
            _dx += _dreturn * (1/y)
            _dy += _dreturn * (-x/y^2)

            Args:
                node (loma_ir.Div)

            Returns:
                list[loma_ir.stmt]
            """
            # store adjoint
            orig_adjoint = self.adjoint
            # deal with left, x
            self.adjoint = loma_ir.BinaryOp(
                loma_ir.Div(), self.adjoint, node.right 
            )  # _dreturn * (1/y)
            left_stmt = self.mutate_expr(node.left)
            self.adjoint = orig_adjoint
            # deal with right, y
            x_y2 = loma_ir.BinaryOp(
                loma_ir.Div(),
                node.left,  # x
                loma_ir.BinaryOp(loma_ir.Mul(), node.right, node.right)
            )  # x/y^2
            self.adjoint = loma_ir.BinaryOp(
                loma_ir.Mul(), 
                self.adjoint, 
                loma_ir.BinaryOp(
                    loma_ir.Sub(),
                    loma_ir.ConstFloat(0.0),
                    x_y2
                )
            )  # _dreturn * (-x/y^2)
            right_stmt = self.mutate_expr(node.right)
            self.adjoint = orig_adjoint

            return left_stmt + right_stmt

        def mutate_call(self, node):
            # HW2: TODO
            return super().mutate_call(node)

    return RevDiffMutator().mutate_function_def(func)

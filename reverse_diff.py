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

    """Data structures accessed by all sub-classes"""
    # dicts: str -> stmt
    increment_ptr, decrement_ptr = {}, {}
    # dict: str -> expr
    cache_access = {}
    # 'y', 'arr', etc if they are Out in the function args
    output_args: set[str] = set()

    def setup_cache_stmts() -> None:
        """Store stmts and exprs to be used many times in global memory. 
        e.g.
        _stack_ptr_float = _stack_ptr_float + 1;
        _t_float[_stack_ptr_float]
        """
        var_int_ptr = loma_ir.Var('_stack_ptr_int')
        var_float_ptr = loma_ir.Var('_stack_ptr_float')
        INT_ONE = loma_ir.ConstInt(1)
        
        increment_ptr['int'] = loma_ir.Assign(var_int_ptr, loma_ir.BinaryOp(loma_ir.Add(), var_int_ptr, INT_ONE))
        increment_ptr['float'] = loma_ir.Assign(var_float_ptr, loma_ir.BinaryOp(loma_ir.Add(), var_float_ptr, INT_ONE))
        decrement_ptr['int'] = loma_ir.Assign(var_int_ptr, loma_ir.BinaryOp(loma_ir.Sub(), var_int_ptr, INT_ONE))
        decrement_ptr['float'] = loma_ir.Assign(var_float_ptr, loma_ir.BinaryOp(loma_ir.Sub(), var_float_ptr, INT_ONE))
        
        
        cache_access['int'] = loma_ir.ArrayAccess(
            loma_ir.Var("_t_int"),
            var_int_ptr,
            t=loma_ir.Int() )
        cache_access['float'] = loma_ir.ArrayAccess(
            loma_ir.Var("_t_float"),
            var_float_ptr,
            t=loma_ir.Float() )
        return

    # Some utility functions
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
                assert False, f"Must specify target.t, got: {target.t}"

    def check_lhs_is_output_arg(lhs: loma_ir.expr) -> bool:
        """If we have y : Out[float],
        Assign to y should be skipped during the forward pass

        Args:
            lhs (loma_ir.expr): y, arr[2], student.name
        """
        match lhs:
            case loma_ir.Var():
                return lhs.id in output_args
            case loma_ir.StructAccess():
                return check_lhs_is_output_arg(lhs.struct, output_args)
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_arg(lhs.array, output_args)
            case _:
                assert False
    
    def advance_stack_ptr(target: str, isIncr: bool) -> loma_ir.Assign:
        """return an Assign stmt
        _stack_ptr_float = _stack_ptr_float +/- 1;
        """
        return loma_ir.Assign(
            target, 
            loma_ir.BinaryOp(
                loma_ir.Add() if isIncr else loma_ir.Sub(),
                loma_ir.Var(target),
                loma_ir.ConstInt(1)
            )
        )

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
                    return self.mutate_assign(node)
                case loma_ir.IfElse():
                    return []
                case loma_ir.While():
                    return []
                case loma_ir.CallStmt():
                    return []
                case _:
                    assert False, f'Visitor error: unhandled statement {node}'

        def mutate_declare(self, node: loma_ir.Declare):
            # automatically initialized to zero if no val
            diff_declare = loma_ir.Declare('_d' + node.target, t=node.t)  
            
            return [node, diff_declare]

        def mutate_assign(self, node: loma_ir.Assign) -> list[loma_ir.stmt]:
            """z = 2.5 * x - 3.0 * y ->
            _t_float[_stack_ptr_float] = z
            _stack_ptr_float = _stack_ptr_float + 1
            z = 2.5 * x - 3.0 * y

            Note:
                if z is Out nothing should happen (return [])
                because only _dz will appear in the diff args

            Args:
                node (loma_ir.Assign)

            Returns:
                list[loma_ir.stmt]
            """
            # node.target is expr
            if check_lhs_is_output_arg(node.target):
                return []
            type_str = type_to_string(node.target.t)
            store_cache = loma_ir.Assign(cache_access[type_str], node.target)
            return [store_cache, increment_ptr[type_str], node]


    # Apply the differentiation.
    class RevDiffMutator(irmutator.IRMutator):
        """ Global class attributes to pass data around"""
        adjoint: loma_ir.Var = None
        # 'x' -> type of x
        id_vars: dict[str, loma_ir.type] = {}
        # 'x' -> Var(_adj_x)
        tmp_adj_Vars: dict[str, loma_ir.Var] = {}

        def mutate_function_def(self, node):
            # HW2: TODO
            # Signature (args)
            new_args = self.process_args(node)
            
            # _dreturn as the start point of rev diff
            if node.ret_type is not None:
                new_args.append(loma_ir.Arg('_dreturn', node.ret_type, loma_ir.In()))

            # maually create first few lines to handle stack
            # TODO: choose optimal size later
            n = len(node.body)
            stack_body = [
                loma_ir.Declare(
                    "_t_float", 
                    t=loma_ir.Array(t=loma_ir.Float(), static_size=n)
                ),
                loma_ir.Declare("_stack_ptr_float", t=loma_ir.Int()),
                loma_ir.Declare(
                    "_t_int", 
                    t=loma_ir.Array(t=loma_ir.Int(), static_size=n)
                ),
                loma_ir.Declare("_stack_ptr_int", t=loma_ir.Int())
            ]
            # set up reusable things
            setup_cache_stmts()

            # copy paste forward code
            fwd_new_body = irmutator.flatten( [PrimalCodeMutator().mutate_stmt(stmt) for stmt in node.body] )
            
            # populate tmp_adj_Vars and create tmp adjoint variables
            tmp_adj_body = self.create_tmp_adjoints(node)

            # backward diff
            rev_new_body = irmutator.flatten( [self.mutate_stmt(stmt) for stmt in reversed(node.body)] )

            # put together everything
            body = stack_body + fwd_new_body + tmp_adj_body + rev_new_body
            return loma_ir.FunctionDef(diff_func_id, new_args, body, node.is_simd, ret_type=None)



        def mutate_return(self, node: loma_ir.Return):
            # HW2: TODO
            # in bwd part, mutate_return should be the first to execute,
            # set global adjoint s.t. callee can use
            self.adjoint = loma_ir.Var('_dreturn')
            stmts = self.mutate_expr(node.val)  # is a list
            self.adjoint = None

            # 5.
            for x, x_type in self.id_vars.items():
                dx_Var = loma_ir.Var('_d' + x, t=x_type)
                stmts += accum_deriv(dx_Var, self.tmp_adj_Vars[x], overwrite=False)
                stmts.append(loma_ir.Assign(self.tmp_adj_Vars[x], loma_ir.ConstFloat(0.0)))
            return stmts

        def mutate_declare(self, node):
            if node.val is None:
                return []
            self.adjoint = loma_ir.Var('_d' + node.target, t=node.t)
            stmts = self.mutate_expr(node.val)  # is a list
            self.adjoint = None

            for x, x_type in self.id_vars.items():
                dx_Var = loma_ir.Var('_d' + x, t=x_type)
                stmts += accum_deriv(dx_Var, self.tmp_adj_Vars[x], overwrite=False)
                stmts.append(loma_ir.Assign(self.tmp_adj_Vars[x], loma_ir.ConstFloat(0.0)))
            
            return stmts

        def mutate_assign(self, node: loma_ir.Assign) -> list[loma_ir.stmt]:
            """ e.g. z = 2.5 * x - 3.0 * y
            1. ptr--
            2. load from cache
            3. store to tmp adjoints (mutate_var does the job)
            4. zero out _dz
            5. accumulate real dval with tmp adjoints

            Note:
                only do step 3. and 5. if target is an Out

            Args:
                node (loma_ir.Assign)

            Returns:
                list[loma_ir.stmt]
            """
            stmts = []
            type_str = type_to_string(node.target.t)
            id_str = node.target.id
            dz_Var = loma_ir.Var('_d' + id_str, t=node.target.t)
            isOut = check_lhs_is_output_arg(node.target)

            if not isOut:
                stmts.append(decrement_ptr[type_str])  # 1.
                stmts.append(loma_ir.Assign(node.target, cache_access[type_str]))  # 2.
            # 3.
            self.adjoint = dz_Var
            stmts += self.mutate_expr(node.val)
            self.adjoint = None
            
            if not isOut:
                stmts.append(loma_ir.Assign(dz_Var, loma_ir.ConstFloat(0.0)))  # 4.
            # 5.
            for x, x_type in self.id_vars.items():
                dx_Var = loma_ir.Var('_d' + x, t=x_type)
                stmts += accum_deriv(dx_Var, self.tmp_adj_Vars[x], overwrite=False)
                stmts.append(loma_ir.Assign(self.tmp_adj_Vars[x], loma_ir.ConstFloat(0.0))) 

            return stmts

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
            return []

        def mutate_var(self, node: loma_ir.Var) -> list[loma_ir.stmt]:
            """ x -> _dx = _dx + adjoint
            BUT, with back propagation, it should be
            x -> _adj_x = _adj_x + adjoint

            Args:
                node (loma_ir.Var): x

            Returns:
                list[loma_ir.stmt]: _dx = _dx + adjoint
            """
            # dx = loma_ir.Var('_d' + node.id, lineno=node.lineno, t=node.t)
            # return accum_deriv(dx, self.adjoint, overwrite=False)

            # backprop version
            assert node.id in self.tmp_adj_Vars, \
                f"tmp_adj_Vars KeyError: {node.id}, keys :{self.tmp_adj_Vars.keys()}"
            lhs = self.tmp_adj_Vars[node.id]
            return accum_deriv(lhs, self.adjoint, overwrite=False)

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

        def mutate_call(self, node: loma_ir.Call) -> list[loma_ir.stmt]:
            """Deal with the following intrinsic functions
            sin(x) -> _dx += adjoint * cos(x)
            cos(x) -> _dx += adjoint * (0 - sin(x))
            sqrt(x) -> _dx += adjoint * (0.5 / sqrt(x))
            pow(x, k) -> _dx += adjoint * (k * pow(x, k-1))
                _dk += adjoint * (log(x) * pow(x, k))
            exp(x) -> _dx += adjoint * exp(x)
            log(x) -> _dx += adjoint * 1/x

            Chain Rule:
                log(x1 * x2) -> _dx1 += adjoint * 1/(x1*x2) * x2
                We should pass df/d_expr to current adjoint

            Note:
                mutate_call() is called by mutate_expr(), who has setup
                self.adjoint already.

            Args:
                node (loma_ir.Call)

            Returns:
                list[loma_ir.stmt]
            """
            if len(node.args) == 0:
                assert False, "function with no arg shouldn't be here"
            stmts = []
            # df/d_expr (e.g. f is log(x*y), expr is x*y)
            df_dexpr: loma_ir.expr = None
            match node.id:
                case "sin":
                    df_dexpr = loma_ir.Call("cos", [node.args[0]])   
                case "cos":
                    df_dexpr = loma_ir.BinaryOp(
                        loma_ir.Sub(), 
                        loma_ir.ConstFloat(0.0), 
                        loma_ir.Call("sin", [node.args[0]])
                    )  # -sin(x)
                case "sqrt":
                    df_dexpr = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        loma_ir.ConstFloat(0.5), 
                        loma_ir.Call("sqrt", [node.args[0]])
                    )  # (0.5 / sqrt(x))
                case "pow":
                    # extra work: need to mutate_expr() twice
                    # will do this for k here
                    assert len(node.args) == 2, "pow() should have 2 args"
                    df_dk = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.Call(
                            "log",
                            [node.args[0]]
                        ),  # log(x)
                        loma_ir.Call(
                            "pow", [
                                node.args[0],
                                node.args[1]
                            ]
                        ) # x^k
                    ) # log(x) * pow(x, k)
                    orig_adjoint = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), self.adjoint, df_dk)
                    stmts += self.mutate_expr(node.args[1])  # on k
                    self.adjoint = orig_adjoint
                    df_dexpr = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        node.args[1],
                        loma_ir.Call(
                            "pow", [
                                node.args[0],
                                loma_ir.BinaryOp(
                                    loma_ir.Sub(),
                                    node.args[1],
                                    loma_ir.ConstFloat(1.0), 
                                )  # k-1
                            ]
                        ) # pow(x, k-1)
                    )  # k * pow(x, k-1)
                case "exp":
                    # just exp(x) itself
                    df_dexpr = node
                case "log":
                    df_dexpr = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        loma_ir.ConstFloat(1.0), 
                        node.args[0]
                    ) # 1/x
                case "int2float":
                    return []
                case "float2int":
                    return []
                case _:
                    # non-intrinsic function with >=0 args
                    assert False, "non-intrinsic function with >=0 args"
            
            # multiply df_dexpr to adjoint and mutate_expr on x
            curr_adjoint = self.adjoint
            assert isinstance(df_dexpr, loma_ir.expr), f"CHECK df_dexpr: {df_dexpr}"
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), self.adjoint, df_dexpr)
            stmts += self.mutate_expr(node.args[0])  # on x
            self.adjoint = curr_adjoint

            return stmts


        def create_tmp_adjoints(self, node: loma_ir.FunctionDef) -> list[loma_ir.Declare]:
            """populate self.tmp_adj_Vars and create tmp adjoint variables

            Args:
                node (loma_ir.FunctionDef)

            Returns:
                list[loma_ir.Declare]
            """
            tmp_adj_body = []
            ## 1. Who need tmp adjoints?
            ## Besides for Inputs, also for local vars
            for stmt in node.body:
                if isinstance(stmt, loma_ir.Declare):
                    self.id_vars[stmt.target] = stmt.t
            ## 2. Build dict[x, Var(_adj_x)]
            for x, x_type in self.id_vars.items():
                self.tmp_adj_Vars[x] = loma_ir.Var(f"_adj_{x}", t=loma_ir.Float())
                tmp_adj_body.append(
                    loma_ir.Declare(f"_adj_{x}", t=loma_ir.Float()) )
            # print(f"CHECK self.id_vars: {self.id_vars}")

            return tmp_adj_body

        def process_args(self, node: loma_ir.FunctionDef) -> list[loma_ir.Arg]:
            new_args = []
            for arg in node.args:
                if isinstance(arg.i, loma_ir.In):
                    self.id_vars[arg.id] = arg.t
                    new_args.append(arg)
                    # also _dx of x
                    darg = loma_ir.Arg(
                        '_d' + arg.id, arg.t, loma_ir.Out()
                    )
                    new_args.append(darg)
                elif isinstance(arg.i, loma_ir.Out):
                    # original var y shouldn't be there
                    # and we don't need _adj_y
                    darg = loma_ir.Arg(
                        '_d' + arg.id, arg.t, loma_ir.In()
                    )
                    new_args.append(darg)
                    # for primal code to skip
                    output_args.add(arg.id)
                else:
                    assert False, "MUST BE IN OR OUT"

            return new_args

    return RevDiffMutator().mutate_function_def(func)

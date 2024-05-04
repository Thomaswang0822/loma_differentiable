import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random
from collections import defaultdict

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
    # dicts: str(type) -> stmt
    increment_ptr, decrement_ptr = {}, {}
    # dict: str -> expr
    cache_access = {}
    # 'y', 'arr', etc if they are Out in the function args
    output_args: set[str] = set()
    # we will scan thru statments, look for Assign,
    # add 'float', 'Foo', 'array_int', etc. as keys
    # and their number of appearance, s.t. we can allocate cache stack
    assignted_types_str: dict[str, int] = defaultdict(int)
    # map a str back to its loma_ir representation
    map_str2type: dict[str, loma_ir.type] = {}

    def setup_cache_stmts() -> list[loma_ir.stmt]:
        """Store stmts and exprs to be used many times in global memory. 
        e.g.
        _stack_ptr_float = _stack_ptr_float + 1;
        _t_float[_stack_ptr_float]

        NOTE: UPDATE
        Declare statments of those stacks and stack ptrs are also done
        and returned here.

        Returns:
            list[loma_ir.stmt]: [declare stack, declare stack ptr] for each type
        """
        INT_ONE = loma_ir.ConstInt(1)
        res = []
        # take 'float' as example
        for t_str, ct in assignted_types_str.items():
            var_ptr = loma_ir.Var(f'_stack_ptr_{t_str}')
            # stmt: _stack_ptr_float = _stack_ptr_float +/- 1
            increment_ptr[t_str] = loma_ir.Assign(var_ptr, loma_ir.BinaryOp(loma_ir.Add(), var_ptr, INT_ONE))
            decrement_ptr[t_str] = loma_ir.Assign(var_ptr, loma_ir.BinaryOp(loma_ir.Sub(), var_ptr, INT_ONE))
            # expr: _t_float[_stack_ptr_float]
            cache_access[t_str] = loma_ir.ArrayAccess(
                loma_ir.Var(f"_t_{t_str}"),
                var_ptr,
                t=map_str2type[t_str])
            res += [
                loma_ir.Declare(
                    f"_t_{t_str}", 
                    t=loma_ir.Array(
                        t=map_str2type[t_str], 
                        static_size=ct)
                ),
                loma_ir.Declare(f"_stack_ptr_{t_str}", t=loma_ir.Int())
            ]

        return res

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
                return check_lhs_is_output_arg(lhs.struct)
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_arg(lhs.array)
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

            NOTE:
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
        """ Global class attributes to pass data around
        adjoint:
            to account for Chain Rules in mutate_var(), e.g. when
            mutate_var(x):
            z = x -> _dx += dz
            z = x*y -> _dx += dz * y

        i_new & i_restore:
            will declare and use tmp adjoints on the fly
            _adj_1 : float; _adj_1 = y * _dz_; _dz_ = 0.0; 9 _dx += _adj_1;
            they are global counters

        tmp_adj_Vars:
            and whenever we create one, we need to remember the actual _dz it belongs to
            i -> (Var(_adj_i), expr(_dz))
        """
        adjoint: loma_ir.Var = None
        i_new: int = 0
        i_restore: int = 0
        tmp_adj_Vars: dict[int, tuple] = {}

        """ mutator functions """
        def mutate_function_def(self, node: loma_ir.FunctionDef) -> loma_ir.FunctionDef:
            """caller of all functions below.
            Turn a full function definition to its bwd_diff version

            Args:
                node (loma_ir.FunctionDef)

            Returns:
                loma_ir.FunctionDef
            """
            # Signature (args)
            new_args = self.process_args(node)
            
            # Cache
            """ preprocess body to know what types need their cache stack """
            self.preprocess_statements(node.body)
            """Use the info in dict assignted_types_str to
            declare stacks and their ptrs, and setup reusable
            statments (ptr++, ptr--, push, pop) """
            stack_body = setup_cache_stmts()

            # copy paste forward code
            fwd_new_body = irmutator.flatten( [PrimalCodeMutator().mutate_stmt(stmt) for stmt in node.body] )
            
            # populate tmp_adj_Vars and create tmp adjoint variables
            # UPDATE: will create tmp adjoints on the fly
            tmp_adj_body = []

            # backward diff
            rev_new_body = irmutator.flatten( [self.mutate_stmt(stmt) for stmt in reversed(node.body)] )

            # put together everything
            body = stack_body + fwd_new_body + tmp_adj_body + rev_new_body
            return loma_ir.FunctionDef(diff_func_id, new_args, body, node.is_simd, ret_type=None)


        def mutate_return(self, node: loma_ir.Return) -> list[loma_ir.stmt]:
            """similar to mutate_assign. Check its docstring

            Args:
                node (loma_ir.Return)

            Returns:
                list[loma_ir.stmt]
            """
            # special handle for Struct, e.g. return foo
            if isinstance(node.val.t, loma_ir.Struct):
                dval = loma_ir.Var('_d' + node.val.id, t=node.val.t)
                dret = loma_ir.Var('_dreturn')
                return accum_deriv(dval, dret, overwrite=True)
            
            # in bwd part, mutate_return should be the first to execute,
            # set global adjoint s.t. callee can use
            # 3.
            self.adjoint = loma_ir.Var('_dreturn')
            stmts = self.mutate_expr(node.val)  # is a list
            self.adjoint = None

            # 5.
            while self.i_restore < self.i_new:
                adj, dx = self.tmp_adj_Vars[self.i_restore]
                stmts += accum_deriv(dx, adj, overwrite=False)
                self.i_restore += 1
            return stmts

        def mutate_declare(self, node: loma_ir.Declare) -> list[loma_ir.stmt]:
            """similar to mutate_assign. Check its docstring

            Args:
                node (loma_ir.Declare)

            Returns:
                list[loma_ir.stmt]
            """
            if node.val is None:
                return []
            # special handle for Struct, e.g. foo : Foo = f
            elif isinstance(node.val.t, loma_ir.Struct):
                drhs = loma_ir.Var('_d' + node.val.id, t=node.val.t)
                dlhs = loma_ir.Var('_d' + node.target, t=node.t)
                return accum_deriv(drhs, dlhs, overwrite=False)
            
            # 3.
            self.adjoint = loma_ir.Var('_d' + node.target, t=node.t)
            stmts = self.mutate_expr(node.val)  # is a list
            self.adjoint = None

            # 5.
            while self.i_restore < self.i_new:
                adj, dx = self.tmp_adj_Vars[self.i_restore]
                stmts += accum_deriv(dx, adj, overwrite=False)
                self.i_restore += 1
            
            return stmts

        def mutate_assign(self, node: loma_ir.Assign) -> list[loma_ir.stmt]:
            """ e.g. z = 2.5 * x - 3.0 * y
            1. ptr--
            2. load from cache
            3. store to tmp adjoints (mutate_var does the job)
            4. zero out _dz
            5. accumulate real dval with tmp adjoints

            NOTE:
                only do step 3. and 5. if target is an Out

            Args:
                node (loma_ir.Assign)

            Returns:
                list[loma_ir.stmt]
            """
            # # need special handle for Struct, e.g. foo = f
            # if isinstance(node.val.t, loma_ir.Struct):
            #     # print(f"CHECK node.val: {node.val}")
            #     drhs = loma_ir.Var('_d' + node.val.id, t=node.val.t)
            #     dlhs = loma_ir.Var('_d' + node.target.id, t=node.target.t)
            #     return accum_deriv(drhs, dlhs, overwrite=False)
            
            stmts = []
            type_str = type_to_string(node.target.t)
            # lhs of assign can only be Var, ArrayAccess, or StructAccess
            if isinstance(node.target, loma_ir.Var):
                id_str = node.target.id
                d_lhs = loma_ir.Var('_d' + id_str, t=node.target.t)
            elif isinstance(node.target, loma_ir.ArrayAccess):
                d_lhs = self.diff_array_access(node.target)
            elif isinstance(node.target, loma_ir.StructAccess):
                d_lhs = self.diff_struct_access(node.target)
            else:
                assert False, "lhs of assign can only be Var, ArrayAccess, or StructAccess"
            isOut = check_lhs_is_output_arg(node.target)

            # 1. & 2.
            if not isOut:
                stmts.append(decrement_ptr[type_str])
                stmts.append(loma_ir.Assign(node.target, cache_access[type_str]))
            # 3.
            self.adjoint = d_lhs
            stmts += self.mutate_expr(node.val)
            self.adjoint = None
            
            # 4.
            if not isOut:
                # stmts.append(loma_ir.Assign(d_lhs, loma_ir.ConstFloat(0.0)))
                stmts += assign_zero(d_lhs)
            # 5.
            while self.i_restore < self.i_new:
                adj, dx = self.tmp_adj_Vars[self.i_restore]
                stmts += accum_deriv(dx, adj, overwrite=False)
                self.i_restore += 1

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
            stmts = []
            # create tmp adjoints
            adj, dx = self.new_tmp_adjoint(node)
            # declare adj
            stmts += [loma_ir.Declare(adj.id, adj.t)]
            # accumulate diff
            stmts += accum_deriv(adj, self.adjoint, overwrite=True)

            return stmts

        def mutate_array_access(self, node: loma_ir.ArrayAccess) -> list[loma_ir.stmt]:
            """see mutate_var()
            An ArrayAccess should be treated the same

            Args:
                node (loma_ir.ArrayAccess)

            Returns:
                list[loma_ir.stmt]
            """
            return self.mutate_var(node)

        def mutate_struct_access(self, node: loma_ir.StructAccess) -> list[loma_ir.stmt]:
            """see mutate_var()
            An StructAccess should be treated the same

            Args:
                node (loma_ir.StructAccess)

            Returns:
                list[loma_ir.stmt]
            """
            return self.mutate_var(node)

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

            NOTE:
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

        def process_args(self, node: loma_ir.FunctionDef) -> list[loma_ir.Arg]:
            """Rule:
            x: In -> x: In, dx: Out
            x: Out -> dx: In
            Append a _dreturn: In at the end.

            Args:
                node (loma_ir.FunctionDef): _description_

            Returns:
                list[loma_ir.Arg]: _description_
            """
            new_args = []
            for arg in node.args:
                if isinstance(arg.i, loma_ir.In):
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

            # _dreturn as the start point of rev diff
            if node.ret_type is not None:
                new_args.append(loma_ir.Arg('_dreturn', node.ret_type, loma_ir.In()))

            return new_args

        def preprocess_statements(self, stmts: list[loma_ir.stmt]) -> None:
            """The only task:
            Add type string (other than 'float' and 'int') of the LHS of an Assign statment,
            which may cause side-effect (assign overwritten), to the assignted_types_str dict.

            Args:
                stmts (list[loma_ir.stmt]): primal code list of statements
            """
            for node in stmts:
                if not isinstance(node, loma_ir.Assign):
                    continue
                # look at LHS type
                lhs_type_str = type_to_string(node.target.t)
                # defaultdict saves the check empty
                assignted_types_str[lhs_type_str] += 1
                map_str2type[lhs_type_str] = node.target.t
            
            # print(f"CHECK assignted_types_str: {assignted_types_str}")
            return

        def new_tmp_adjoint(self, node:loma_ir.expr) -> tuple[loma_ir.expr, ...]:
            """create tmp adjoint (_adj_{i}) for an expr (x, arr[2], foo.a, or foo)
            record the corresponding derivative (dx, darr[2], dfoo.a, or dfoo) it belongs to

            Args:
                node (loma_ir.expr)

            Returns:
                tmp adjoint, original derivative
            """
            # create tmp adjoints
            adj = loma_ir.Var(f"_adj_{self.i_new}", t=node.t)
            # original derivative can be
            if isinstance(node, loma_ir.Var):
                dx = loma_ir.Var('_d' + node.id, lineno=node.lineno, t=node.t)
            elif isinstance(node, loma_ir.ArrayAccess):
                # recursively trace to array variable name
                dx = self.diff_array_access(node)
            elif isinstance(node, loma_ir.StructAccess):
                dx = self.diff_struct_access(node)
            else:
                assert False, "Other type of expr shouldn't call"
            # record their link and increment counter
            self.tmp_adj_Vars[self.i_new] = (adj, dx,)
            self.i_new += 1

            return adj, dx

        def diff_array_access(self, node: loma_ir.ArrayAccess) -> loma_ir.ArrayAccess:
            """arr[1][2][3] -> _darr[1][2][3]
            by recursively traverse the array attribute until we get str

            Args:
                node (loma_ir.ArrayAccess)

            Returns:
                loma_ir.ArrayAccess
            """
            if isinstance(node.array, loma_ir.Var):
                d_array = loma_ir.Var('_d' + node.array.id)
            elif isinstance(node.array, loma_ir.StructAccess):
                # may have array as Struct member, e.g. foo.member_arr[3]
                d_array = self.diff_struct_access(node.array)
            else:
                d_array = self.diff_array_access(node.array)
            d_node = loma_ir.ArrayAccess(
                array=d_array,
                index=node.index,
                t=loma_ir.Float()  # shouldn't ever accumulate derivative to int Array
            )
            return d_node

        def diff_struct_access(self, node:loma_ir.StructAccess) -> loma_ir.StructAccess:
            """foo.bar.x -> _dfoo.bar.x

            Args:
                node (loma_ir.StructAccess)

            Returns:
                loma_ir.StructAccess
            """
            if isinstance(node.struct, loma_ir.Var):
                d_s = loma_ir.Var('_d' + node.struct.id)
            elif isinstance(node.struct, loma_ir.ArrayAccess):
                # may also have array of Struct, e.g. arr[0].x
                d_s = self.diff_array_access(node.struct)
            else:
                d_s = self.diff_array_access(node.struct)
            d_node = loma_ir.StructAccess(
                struct=d_s,
                member_id=node.member_id,
                t=node.t
            )
            return d_node
            

    return RevDiffMutator().mutate_function_def(func)

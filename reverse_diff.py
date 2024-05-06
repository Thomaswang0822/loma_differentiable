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
    """For loop"""
    curr_loop_i: int = 0
    # outmost loop will have value 1, useful for checking whether outmost
    parent_iter_sizes: dict[int, int] = defaultdict(int)
    curr_parent_iter_size: int = 1
    # storage of tuple(_loop_counter_1_arr, _loop_counter_1_ptr, _loop_counter_1_tmp)
    # First 2 will be None for outermost loop, need to check when use
    loop_counter_vars_map: dict[int, tuple[loma_ir.Var, ...]] = defaultdict(tuple)

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
                    return [loma_ir.CallStmt(call=loma_ir.Call(
                        'atomic_add',
                        args=[target, deriv],
                        t=target.t
                    ))]
                    # return [loma_ir.Assign(target,
                    #     loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
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

    def find_CallStmt_Out_args(node: loma_ir.CallStmt) -> list[loma_ir.expr]:
        """Given a primal-code CallStmt, find those args (expr) that act as
        Out of the callee but are not of the self/caller.
        These args can potentially cause side effect.

        Args:
            node (loma_ir.CallStmt): e.g. foo(x, y)

        Returns:
            list[loma_ir.expr]: _description_
        """
        out_args_expr: list[loma_ir.expr] = []
        func_name: str = node.call.id
        actual_func: loma_ir.FunctionDef = funcs[func_name]
        inouts: list[loma_ir.inout] = [arg.i for arg in actual_func.args]
        for arg_expr, io in zip(node.call.args, inouts):
            # arg_expr takes the place of an Out arg of the callee
            if io == loma_ir.Out():
                # but skip if arg_expr is an Out arg of the caller
                if not check_lhs_is_output_arg(arg_expr):
                    out_args_expr.append(arg_expr)
        # print(f"CHECK out_args_expr: {out_args_expr}")
        return out_args_expr

    def store_loop_counter_vars(is_inner: bool):
        """store to loop_counter_vars_map
        """
        arr = loma_ir.Var(
            f"_loop_counter_{curr_loop_i}",
            t=loma_ir.Array(t=loma_ir.Int(), static_size=curr_parent_iter_size)
        ) if is_inner else None
        ptr = loma_ir.Var(
            f"_loop_counter_{curr_loop_i}_ptr", 
            t=loma_ir.Int()
        ) if is_inner else None
        tmp = loma_ir.Var(f"_loop_counter_{curr_loop_i}_tmp", t=loma_ir.Int())

        loop_counter_vars_map[curr_loop_i] = (arr, ptr, tmp)
    
    def declare_loop_counter_vars() -> list[loma_ir.stmt]:
        """Generate stmts like these:
_loop_counter_0_tmp : int

_loop_counter_1 : Array[int, 50]
_loop_counter_1_ptr : int = 0
_loop_counter_1_tmp : int

        Returns:
            list[loma_ir.stmt]
        """
        stmts = []
        for i in range(curr_loop_i):
            is_inner = bool(parent_iter_sizes[i] > 1)
            # Grab LHS var
            arr, ptr, tmp = loop_counter_vars_map[i]
            if is_inner:
                assert bool(arr is not None) and bool(ptr is not None)
                stmts += [
                    loma_ir.Declare(arr.id, t=arr.t),
                    loma_ir.Declare(ptr.id, t=loma_ir.Int())
                ]
            stmts.append(loma_ir.Declare(tmp.id, t=loma_ir.Int()))

        return stmts
    
    def inner_loop_stmt_fwd(is_inner: bool):
        """Generate 4 (or 2) stmts for while() in fwd pass
    _loop_counter_1_tmp = 0
    # while():
        _loop_counter_1_tmp = _loop_counter_1_tmp + 1
    _loop_counter_1[_loop_counter_1_ptr] = _loop_counter_1_tmp
    _loop_counter_1_ptr = _loop_counter_1_ptr + 1

        NOTE:
            Last 2 only for is_inner
        """
        assert curr_loop_i in loop_counter_vars_map, "Haven't store loop_counter_vars"
        arr, ptr, tmp = loop_counter_vars_map[curr_loop_i]

        # _loop_counter_1_tmp = 0 is always regardless of is_inner
        zero_tmp = [ loma_ir.Assign(target=tmp, val=loma_ir.ConstInt(0)) ]

        inc_tmp = [loma_ir.Assign(
            target=tmp, 
            val=loma_ir.BinaryOp(loma_ir.Add(), tmp, loma_ir.ConstInt(1))
        )]
        
        end2 = [
            loma_ir.Assign(
                target=loma_ir.ArrayAccess(
                    array=arr,
                    index=ptr,
                    t=loma_ir.Int()
                ),
                val=tmp
            ),
            loma_ir.Assign(
                target=ptr,
                val=loma_ir.BinaryOp(loma_ir.Add(), ptr, loma_ir.ConstInt(1))
            )
        ] if is_inner else []
        return zero_tmp, inc_tmp, end2

    def inner_loop_stmt_bwd(is_inner: bool):
        """Generate
    _loop_counter_1_ptr = _loop_counter_1_ptr - 1
    _loop_counter_1_tmp = _loop_counter_1[_loop_counter_1_ptr]
        _loop_counter_1_tmp > 0  # An expr
        _loop_counter_1_tmp = _loop_counter_1_tmp - 1
        
        NOTE:
            First 2 only for is_inner
        """
        # print(f"CHECK curr_loop_i: {curr_loop_i}")
        # print(f"CHECK loop_counter_vars_map: {loop_counter_vars_map}")
        assert curr_loop_i in loop_counter_vars_map, "Haven't store loop_counter_vars"
        arr, ptr, tmp = loop_counter_vars_map[curr_loop_i]

        start2 = [
            loma_ir.Assign(
                target=ptr,
                val=loma_ir.BinaryOp(loma_ir.Sub(), ptr, loma_ir.ConstInt(1))
            ),
            loma_ir.Assign(
                target=tmp,
                val=loma_ir.ArrayAccess(
                    array=arr,
                    index=ptr,
                    t=loma_ir.Int()
                )
            )          
        ] if is_inner else []

        cond_expr = loma_ir.BinaryOp(loma_ir.Greater(), tmp, loma_ir.ConstInt(0))

        dec_tmp = [loma_ir.Assign(
            target=tmp, 
            val=loma_ir.BinaryOp(loma_ir.Sub(), tmp, loma_ir.ConstInt(1))
        )]

        # list[2 stmts], expr, list[1 stmt]
        return start2, cond_expr, dec_tmp


    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or 
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    class CallNormalizeMutator(irmutator.IRMutator):
        """Helper class that factor out the expressions inside the function arguments, 
        until they can be used as a left hand side in an assign statement.

        Example: f(x*y, 5.0+z) will be turned into
        _call_t_0: float
        _call_t_1: float
        _call_t_0 = x*y
        _call_t_1 = 5.0+z
        f(_call_t_0, _call_t_1)
        """
        def mutate_function_def(self, node: loma_ir.FunctionDef):
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
                    # tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    tmp_name = f'_call_t_{self.tmp_count}'
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
        def mutate_return(self, node: loma_ir.Return):
            # hide original return
            return []

        def mutate_declare(self, node: loma_ir.Declare):
            if isinstance(node.t, loma_ir.Int):
                return [node]
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

        def mutate_call_stmt(self, node: loma_ir.CallStmt) -> list[loma_ir.stmt]:
            """Analogous to mutate_assign() above

            Args:
                node (loma_ir.CallStmt): _description_

            Returns:
                list[loma_ir.stmt]: _description_
            """
            # If the primal code calls foo(x, y), but y is an Out in the arg list,
            # we need to ignore this CallStmt
            for arg_expr in node.call.args:
                if check_lhs_is_output_arg(arg_expr):
                    return []
            

            pre = []
            call_lines = [super().mutate_call_stmt(node)]

            # 0. find all Out arg (expr)
            out_args_expr: list[loma_ir.expr] = find_CallStmt_Out_args(node)

            # Example: y is Out in foo(x, y)
            for arg_expr in out_args_expr:
                # arg_expr is y
                type_str: str = type_to_string(arg_expr.t)
                # store cache
                pre.append(loma_ir.Assign(cache_access[type_str], arg_expr))
                # increment ptr
                pre.append(increment_ptr[type_str])

            return pre + call_lines

        def mutate_while(self, node: loma_ir.While) -> list[loma_ir.stmt]:
            """Forward Pass should look like:
_loop_counter_0_tmp = 0
while (cond0, max_iter := 50):
    # ...
    _loop_counter_1_tmp = 0
    while (cond1, max_iter := 60):
        # ...
        _loop_counter_2_tmp = 0
        while (cond2, max_iter := 70):
            # ...
            _loop_counter_2_tmp = _loop_counter_2_tmp + 1
        # push 2
        _loop_counter_2[_loop_counter_2_ptr] = _loop_counter_2_tmp
        _loop_counter_2_ptr = _loop_counter_2_ptr + 1
        # increment 1
        _loop_counter_1_tmp = _loop_counter_1_tmp + 1
    # push 1
    _loop_counter_1[_loop_counter_1_ptr] = _loop_counter_1_tmp
    _loop_counter_1_ptr = _loop_counter_1_ptr + 1
    # increment 0
    _loop_counter_0_tmp = _loop_counter_0_tmp + 1
            """
            nonlocal curr_loop_i, curr_parent_iter_size

            # record parent itersize
            parent_iter_sizes[curr_loop_i] = curr_parent_iter_size
            is_inner: bool = bool(curr_parent_iter_size > 1)
            # And with this size, we can create the 3 (or 1) Var
            store_loop_counter_vars(is_inner)

            # Create those 3 stmts that sandwich while(), if inner loop
            zero_tmp, inc_tmp, end2 = inner_loop_stmt_fwd(is_inner)
            # Must create stmts for "push to counter array" after while() here,
            # since after mutating body, curr_loop_i is no longer 1

            # before mutate while body, 
            # propagate max_iter and i++
            curr_loop_i += 1
            curr_parent_iter_size *= node.max_iter

            # mutate body
            # new_cond = self.mutate_expr(node.cond)
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)
            new_body += inc_tmp
            big_while: loma_ir.While = loma_ir.While(
                cond=node.cond,
                max_iter=node.max_iter,
                body=new_body
            )

            # restore curr_parent_iter_size, in case of a sibling while()
            curr_parent_iter_size //= node.max_iter

            return zero_tmp + [big_while] + end2

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
        def mutate_function_def(self, primal_node: loma_ir.FunctionDef) -> loma_ir.FunctionDef:
            """caller of all functions below.
            Turn a full function definition to its bwd_diff version

            Args:
                primal_node (loma_ir.FunctionDef)

            Returns:
                loma_ir.FunctionDef
            """
            # BEFORE ALL, normalize call such as f(x+y, 5*z), see
            node: loma_ir.FunctionDef = CallNormalizeMutator().mutate_function_def(primal_node)
            
            # Signature (args)
            new_args = self.process_args(node)
            
            # Cache
            """ preprocess body to know what types need their cache stack """
            self.preprocess_statements(node.body)
            """Use the info in dict assignted_types_str to
            declare stacks and their ptrs, and setup reusable
            statments (ptr++, ptr--, push, pop) """
            stack_body = setup_cache_stmts()

            # non-trivially "copy-paste" primal code, see PrimalCodeMutator
            fwd_new_body = irmutator.flatten( [PrimalCodeMutator().mutate_stmt(stmt) for stmt in node.body] )

            # loop UPDATE: after forward pass, we have full info about loop counters
            loop_counter_body = declare_loop_counter_vars()

            # If there are 4 loops (in the tree), i=5 after fwd pass
            # want to begin with 4 and end with -1
            nonlocal curr_loop_i
            curr_loop_i -= 1

            # backward pass
            rev_new_body = irmutator.flatten( [self.mutate_stmt(stmt) for stmt in reversed(node.body)] )

            # UPDATE: tmp adjoint declaration lines at the beginning of rev code,
            # but only can be created after we mutate the body
            tmp_adj_body = self.declare_tmp_adjoints()

            # put together everything in the correct order
            body = stack_body + loop_counter_body + fwd_new_body + tmp_adj_body + rev_new_body
            return loma_ir.FunctionDef(diff_func_id, new_args, body, node.is_simd, ret_type=None)


        def mutate_return(self, node: loma_ir.Return) -> list[loma_ir.stmt]:
            """similar to mutate_assign. Check its docstring

            Args:
                node (loma_ir.Return)

            Returns:
                list[loma_ir.stmt]
            """            
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
            stmts = []
            type_str = type_to_string(node.target.t)
            isOut = check_lhs_is_output_arg(node.target)

            # 0. find _d{node.target}, which is also an expr
            # lhs of assign can only be Var, ArrayAccess, or StructAccess
            d_lhs = self.to_d_expr(node.target)    

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

        def mutate_ifelse(self, node: loma_ir.IfElse) -> loma_ir.IfElse:
            # in rev mode, y and _dy are separate
            new_cond = node.cond
            new_then_stmts = [self.mutate_stmt(stmt) for stmt in reversed(node.then_stmts)]
            new_else_stmts = [self.mutate_stmt(stmt) for stmt in reversed(node.else_stmts)]
            # Important: mutate_stmt can return a list of statements. We need to flatten the lists.
            new_then_stmts = irmutator.flatten(new_then_stmts)
            new_else_stmts = irmutator.flatten(new_else_stmts)
            return loma_ir.IfElse(
                new_cond,
                new_then_stmts,
                new_else_stmts,
                lineno = node.lineno)

        def mutate_call_stmt(self, node: loma_ir.CallStmt) -> list[loma_ir.stmt]:
            """Similar to mutate_assign(), use tmp adjoint and cache stack
            to deal with side effect.
            Side effect is dealt with here instead of in mutate_call() because
            a Call can be a RHS expr, like y = foo(x,y), which will be dealt
            by mutate_assign().

            NOTE:
                This time, the 12345 steps are done on every Out of primal function.
            
            NOTE:
                Need special handling for atomic_add() here, since it's a void function.
            """
            # maunally deal with atomic_add(y, x) <=> y = y + x
            # rev code should be _dx = _dx + _dy <=> atomic_add(_dx, _dy)
            if isinstance(node.call, loma_ir.Call) and node.call.id == "atomic_add":
                assert len(node.call.args) == 2
                y, x = node.call.args
                dy, dx = self.to_d_expr(y), self.to_d_expr(x)
                aa_call = loma_ir.Call(
                    "atomic_add",
                    [dx, dy],
                    t=node.call.t
                )
                return [loma_ir.CallStmt(aa_call)]

            pre, post = [], []
            call_lines = self.mutate_custom_call_bwd(node.call)

            # 0. find all Out arg (expr)
            out_args_expr: list[loma_ir.expr] = find_CallStmt_Out_args(node)

            # Example: y is Out in foo(x, y)
            for arg_expr in out_args_expr:
                # arg_expr is y
                type_str: str = type_to_string(arg_expr.t)
                # 1. & 2.
                pre.append(decrement_ptr[type_str])
                pre.append(loma_ir.Assign(arg_expr, cache_access[type_str]))
                # 3. mutate_custom_call_bwd() should be outside
                # 4. zero out _dy
                post += assign_zero(self.to_d_expr(arg_expr))
                # 5.
                while self.i_restore < self.i_new:
                    adj, dx = self.tmp_adj_Vars[self.i_restore]
                    post += accum_deriv(dx, adj, overwrite=False)
                    self.i_restore += 1

            """Deal with:
            _adj_0 : float
            _d_rev_foo(x,_adj_0,_dy)
            _dx = (_dx) + (_adj_0)
            """
            while self.i_restore < self.i_new:
                adj, dx = self.tmp_adj_Vars[self.i_restore]
                post += accum_deriv(dx, adj, overwrite=False)
                self.i_restore += 1

            return pre + call_lines + post

        def mutate_while(self, node: loma_ir.While) -> list[loma_ir.stmt]:
            """Reverse pass should look like:
while (_loop_counter_0_tmp > 0, max_iter := 50):
    # body 0

    # pop 1
    _loop_counter_1_ptr = _loop_counter_1_ptr - 1
    _loop_counter_1_tmp = _loop_counter_1[_loop_counter_1_ptr]
    while (_loop_counter_1_tmp > 0, max_iter := 60):
        # body 1

        # pop 2
        _loop_counter_2_ptr = _loop_counter_2_ptr - 1
        _loop_counter_2_tmp = _loop_counter_2[_loop_counter_2_ptr]
        while (_loop_counter_2_tmp > 0, max_iter := 70):
            # body 2

            # decrement 2
            _loop_counter_2_tmp = _loop_counter_2_tmp - 1
        
        # decrement 1
        _loop_counter_1_tmp = _loop_counter_1_tmp - 1
    
    # decrement 0
    _loop_counter_0_tmp = _loop_counter_0_tmp - 1
            """
            # NOTE: if we think of nested while as a tree, counter index is
            # incremented in "pre-order, left-child-first" traveral order
            # To perfect invert this traversal (such that we can index--),
            # we need to traverse in "post-order, right-child-first"
            
            nonlocal curr_loop_i, curr_parent_iter_size

            # post-order means we mutate body first
            new_body = [self.mutate_stmt(stmt) for stmt in reversed(node.body)]
            new_body = irmutator.flatten(new_body)

            # After mutate body (children traversal), current while gets back its index
            is_inner = bool(parent_iter_sizes[curr_loop_i] > 1)
            # and we can get those stmts
            start2, cond_expr, dec_tmp = inner_loop_stmt_bwd(is_inner)

            # reconstruct while
            new_body += dec_tmp
            big_while = loma_ir.While(
                cond=cond_expr,
                max_iter=node.max_iter,
                body=new_body
            )
            # Last step, i--
            curr_loop_i -= 1

            return start2 + [big_while]

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
            # UPDATE: CANNOT declare adj ON-THE-FLY
            # stmts += [loma_ir.Declare(adj.id, adj.t)]
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
            # if len(node.args) == 0:
            #     assert False, "function with no arg shouldn't be here"
            stmts = []
            # df/d_expr (e.g. f is log(x*y), expr is x*y)
            df_dexpr: loma_ir.expr = None
            match node.id:
                case "thread_id":
                    return []
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
                    # custom function
                    return self.mutate_custom_call_bwd(node)
            
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
                # In/Out
                if isinstance(arg.i, loma_ir.In):
                    # # Save type str to map
                    # type_str: str = type_to_string(arg.t)
                    # assignted_types_str[type_str] += 1
                    # map_str2type[type_str] = arg.t
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

        def preprocess_statements(self, stmts: list[loma_ir.stmt], num_iter: int = 1) -> None:
            """The only task:
            Add type string (other than 'float' and 'int') of the LHS of an Assign statment,
            which may cause side-effect (assign overwritten), to the assignted_types_str dict.

            NOTE: update for while loop
            For each occurance in the while loop body, it needs num_iter slots in the cache.

            Args:
                stmts (list[loma_ir.stmt]): primal code list of statements
            """
            for node in stmts:
                if isinstance(node, loma_ir.Assign):
                    # look at LHS type
                    lhs_type_str = type_to_string(node.target.t)
                    # defaultdict saves the check empty
                    assignted_types_str[lhs_type_str] += 1 * num_iter
                    map_str2type[lhs_type_str] = node.target.t
                elif isinstance(node, loma_ir.Declare):
                    # look at LHS type
                    lhs_type_str = type_to_string(node.t)
                    # defaultdict saves the check empty
                    assignted_types_str[lhs_type_str] += 1
                    map_str2type[lhs_type_str] = node.t
                elif isinstance(node, loma_ir.While):
                    # Assign may happen in loop
                    self.preprocess_statements(stmts=node.body, num_iter=node.max_iter)
            
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

        def declare_tmp_adjoints(self) -> list[loma_ir.Declare]:
            """Use tmp_adj_Vars, which is populated after mutate body,
            to create tmp adjoints declaration lines

            Returns:
                list[loma_ir.Declare]
            """
            res = []
            for i in self.tmp_adj_Vars:
                # grab tmp adjoint: loma_ir.Var
                adj, _ = self.tmp_adj_Vars[i]
                res.append(loma_ir.Declare(adj.id, adj.t))
            return res

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

        def to_d_expr(self, node: loma_ir.expr) -> loma_ir.expr:
            """Add "_d" in front of a LHS value; struct and array access
            make it non-trivial

            NOTE:
                LHS value can only be Var, ArrayAccess, or StructAccess;
                and Var include Array and Struct
            """
            d_lhs: loma_ir.expr = None
            if isinstance(node, loma_ir.Var):
                id_str = node.id
                d_lhs = loma_ir.Var('_d' + id_str, t=node.t)
            elif isinstance(node, loma_ir.ArrayAccess):
                d_lhs = self.diff_array_access(node)
            elif isinstance(node, loma_ir.StructAccess):
                d_lhs = self.diff_struct_access(node)
            else:
                assert False, f"lhs value can only be Var, ArrayAccess, or StructAccess, got {node}"
            return d_lhs

        def mutate_custom_call_bwd(self, node: loma_ir.Call) -> list[loma_ir.stmt]:
            stmts = []
            d_func_name = func_to_rev[node.id]

            # grab In/Out of the arguments, this time for a different reason
            actual_func: loma_ir.FunctionDef = funcs[node.id]
            inouts: list[loma_ir.inout] = [arg.i for arg in actual_func.args]
            assert len(inouts) == len(node.args)

            # process args, example foo(x : In[float], y : Out[float])
            d_args: list[loma_ir.expr] = []
            for arg_expr, io in zip(node.args, inouts):
                arg_expr: loma_ir.expr
                # if In, keep x as In and _dx as out
                if io == loma_ir.In():
                    d_args.append(arg_expr)
                    # find out "_dx"
                    if isinstance(arg_expr.t, loma_ir.Array):
                        d_args.append(self.to_d_expr(arg_expr))
                    else:
                        adj, _ = self.new_tmp_adjoint(arg_expr)
                        d_args.append(adj)
                # if Out, keep only _dy as In
                elif io == loma_ir.Out():
                    d_args.append(self.to_d_expr(arg_expr))
                else:
                    assert False, f"custom call on {d_func_name} has arg not In/Out"
            # IMPORTANT: if primal function returns instead of storing to Out
            if actual_func.ret_type is not None:
                # e.g. foo(x : In[float], y : In[float]) -> float:
                # This Call should appear in an Assign or Declare or Return,
                # where the storage Var has been written to self.adjoint
                d_args.append(self.adjoint)

            stmts.append(loma_ir.CallStmt(
                call=loma_ir.Call(id=d_func_name, args=d_args, lineno=node.lineno, t=node.t)
            ))

            return stmts
                  
            

    return RevDiffMutator().mutate_function_def(func)

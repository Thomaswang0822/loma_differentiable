import inspect
import ast
import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import attrs
import error

def annotation_to_inout(arg) -> loma_ir.inout:
    """ Determine whether the function argument
        is input or output.
        In[float] -> input
        Out[int] -> output
    """
    annotation = arg.annotation
    if type(annotation) != ast.Subscript:
        raise error.FuncArgNotAnnotated(arg)
    # TODO: error message
    assert type(annotation.value) == ast.Name
    if annotation.value.id == 'In':
        return loma_ir.In()
    elif annotation.value.id == 'Out':
        return loma_ir.Out()
    else:
        # TODO: error message
        assert False

def annotation_to_type(node) -> loma_ir.type:
    """ Given a Python AST node, returns the corresponding
        loma type
    """

    match node:
        case ast.Name():
            if node.id == 'int':
                return loma_ir.Int()
            elif node.id == 'float':
                return loma_ir.Float()
            else:
                # Struct members to be filled later
                return loma_ir.Struct(node.id, [])
        case ast.Subscript():
            assert isinstance(node.value, ast.Name)
            if node.value.id == 'In' or node.value.id == 'Out':
                # Ignore input/output qualifiers
                return annotation_to_type(node.slice)
            elif node.value.id == 'Array':
                array_type = node.slice
                static_size = None
                if isinstance(array_type, ast.Tuple):
                    assert len(array_type.elts) == 2 # TODO: error message
                    assert isinstance(array_type.elts[1], ast.Constant)
                    static_size = int(array_type.elts[1].value)
                    array_type = array_type.elts[0]
                return loma_ir.Array(annotation_to_type(array_type), static_size)
            elif node.value.id == 'Diff':
                # This is a "differential type" -- we'll resolve this in autodiff
                return loma_ir.Diff(annotation_to_type(node.slice))
            else:
                # TODO: error message
                assert False
        case _:
            assert False

def ast_cmp_op_convert(node) -> loma_ir.bin_op:
    """ Given a Python AST node representing
        a comparison operator,
        convert to the corresponding loma
        comparison operator.
    """
    match node:
        case ast.Lt():
            return loma_ir.Less()
        case ast.LtE():
            return loma_ir.LessEqual()
        case ast.Gt():
            return loma_ir.Greater()
        case ast.GtE():
            return loma_ir.GreaterEqual()
        case ast.Eq():
            return loma_ir.Equal()
        case ast.And():
            return loma_ir.And()
        case ast.Or():
            return loma_ir.Or()
        case _:
            # TODO: error message
            assert False

def parse_ref(node) -> loma_ir.expr:
    """ Given a Python AST node representing
        a LHS reference,
        convert to the corresponding loma expression.
    """

    match node:
        case ast.Name():
            return loma_ir.Var(node.id)
        case ast.Subscript():
            return loma_ir.ArrayAccess(parse_ref(node.value),
                                       visit_expr(node.slice))
        case ast.Attribute():
            return loma_ir.StructAccess(parse_ref(node.value),
                                        node.attr)
        case _:
            # TODO: error message
            assert False

def visit_FunctionDef(node) -> loma_ir.FunctionDef:
    """ Given a Python AST node representing
        a function definition,
        convert to the corresponding loma
        FunctionDef.
    """

    node_args = node.args
    assert node_args.vararg is None
    assert node_args.kwarg is None
    args = [loma_ir.Arg(arg.arg,
                        annotation_to_type(arg.annotation),
                        annotation_to_inout(arg)) for arg in node_args.args]
    body = [visit_stmt(b) for b in node.body]
    ret_type = None
    if node.returns:
        ret_type = annotation_to_type(node.returns)

    is_simd = False
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            if decorator.id == 'simd':
                is_simd = True

    return loma_ir.FunctionDef(node.name,
                               args,
                               body,
                               is_simd,
                               ret_type = ret_type,
                               lineno = node.lineno)

def visit_Differentiate(node) -> loma_ir.func:
    """ Given a Python AST node representing
        a global assignment,
        convert to the corresponding loma
        derivative function declaration.

        For example, the following Python code
        d_foo = fwd_diff(foo)
        converts to
        loma_ir.ForwardDiff('d_foo', 'foo')
    """

    assert isinstance(node, ast.Assign)
    assert len(node.targets) == 1
    func_id = node.targets[0].id
    assert isinstance(node.value, ast.Call)
    call_name = node.value.func
    assert isinstance(call_name, ast.Name)
    call_name = call_name.id
    assert len(node.value.args) == 1
    primal_func_id = node.value.args[0]
    assert isinstance(primal_func_id, ast.Name)
    primal_func_id = primal_func_id.id
    if call_name == 'fwd_diff':
        return loma_ir.ForwardDiff(func_id, primal_func_id, lineno = node.lineno)
    elif call_name == 'rev_diff':
        return loma_ir.ReverseDiff(func_id, primal_func_id, lineno = node.lineno)
    else:
        assert False, f'Unknown function transform operation {call_name}'

def visit_ClassDef(node) -> loma_ir.Struct:
    """ Given a Python AST node representing a class definition,
        convert to a loma Struct.

        e.g.,
        class Foo:
            x : int
            y : float
    """

    members = []
    for member in node.body:
        match member:
            case ast.AnnAssign():
                assert isinstance(member.target, ast.Name)
                t = annotation_to_type(member.annotation)
                members.append(loma_ir.MemberDef(member.target.id, t))
            case _:
                assert False, f'Unknown class member statement {type(member).__name__}'
    return loma_ir.Struct(node.name, members, lineno = node.lineno)

def visit_stmt(node) -> loma_ir.stmt:
    """ Given a Python AST node representing a statement,
        converts to a loma IR statement.
    """
    match node:
        case ast.Return():
            return loma_ir.Return(visit_expr(node.value), lineno = node.lineno)
        case ast.AnnAssign():
            t = annotation_to_type(node.annotation)
            return loma_ir.Declare(node.target.id,
                                   t,
                                   visit_expr(node.value),
                                   lineno = node.lineno)
        case ast.Assign():
            assert len(node.targets) == 1
            target = node.targets[0]
            return loma_ir.Assign(parse_ref(target),
                                  visit_expr(node.value),
                                  lineno = node.lineno)
        case ast.If():
            cond = visit_expr(node.test)
            then_stmts = [visit_stmt(s) for s in node.body]
            else_stmts = [visit_stmt(s) for s in node.orelse]
            return loma_ir.IfElse(cond,
                                  then_stmts,
                                  else_stmts,
                                  lineno = node.lineno)
        case ast.While():
            # node.test needs to be a two items tuple 
            # (condition, max_iter := ...)
            # where condition is an expression, and max_iter is an assignemt
            # of a compile-time static integer
            # TODO: error messages
            assert isinstance(node.test, ast.Tuple) 
            assert len(node.test.elts) == 2 
            cond = visit_expr(node.test.elts[0])
            max_iter_assign = node.test.elts[1]
            assert isinstance(max_iter_assign, ast.NamedExpr)
            assert max_iter_assign.target.id == 'max_iter'
            assert isinstance(max_iter_assign.value, ast.Constant)
            max_iter = int(max_iter_assign.value.value)
            body = [visit_stmt(s) for s in node.body]
            return loma_ir.While(cond, max_iter, body, lineno = node.lineno)
        case ast.Expr():
            # TODO: error messages
            assert isinstance(node.value, ast.Call)
            call_expr = visit_expr(node.value)
            return loma_ir.CallStmt(call_expr, lineno = node.lineno)
        case _:
            assert False, f'Unknown statement {type(node).__name__}'

def visit_expr(node) -> loma_ir.expr:
    """ Given a Python AST node representing an expression,
        converts to a loma IR expression.
    """

    match node:
        case ast.Name():
            return loma_ir.Var(node.id, lineno = node.lineno)
        case ast.Constant():
            if type(node.value) == int:
                return loma_ir.ConstInt(node.value, lineno = node.lineno)  
            elif type(node.value) == float:
                return loma_ir.ConstFloat(node.value, lineno = node.lineno)
            else:
                assert False, f'Unknown constant type'
        case ast.UnaryOp():
            if isinstance(node.op, ast.USub):
                return loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstInt(0), visit_expr(node.operand), lineno = node.lineno)
            else:
                assert False, f'Unknown UnaryOp {type(node.op).__name__}'
        case ast.BinOp():
            match node.op:
                case ast.Add():
                    return loma_ir.BinaryOp(loma_ir.Add(),
                                            visit_expr(node.left),
                                            visit_expr(node.right),
                                            lineno = node.lineno)
                case ast.Sub():
                    return loma_ir.BinaryOp(loma_ir.Sub(),
                                            visit_expr(node.left),
                                            visit_expr(node.right),
                                            lineno = node.lineno)
                case ast.Mult():
                    return loma_ir.BinaryOp(loma_ir.Mul(),
                                            visit_expr(node.left),
                                            visit_expr(node.right),
                                            lineno = node.lineno)
                case ast.Div():
                    return loma_ir.BinaryOp(loma_ir.Div(),
                                            visit_expr(node.left),
                                            visit_expr(node.right),
                                            lineno = node.lineno)
                case _:
                    assert False, f'Unknown BinOp {type(node.op).__name__}'
        case ast.Subscript():
            return loma_ir.ArrayAccess(visit_expr(node.value),
                                       visit_expr(node.slice),
                                       lineno = node.lineno)
        case ast.Attribute():
            return loma_ir.StructAccess(visit_expr(node.value),
                                        node.attr,
                                        lineno = node.lineno)
        case ast.Compare():
            assert len(node.ops) == 1
            assert len(node.comparators) == 1
            op = ast_cmp_op_convert(node.ops[0])
            left = visit_expr(node.left)
            right = visit_expr(node.comparators[0])
            return loma_ir.BinaryOp(op, left, right, lineno = node.lineno)
        case ast.BoolOp():
            op = ast_cmp_op_convert(node.op)
            assert len(node.values) == 2
            left = visit_expr(node.values[0])
            right = visit_expr(node.values[1])
            return loma_ir.BinaryOp(op, left, right, lineno = node.lineno)
        case ast.Call():
            assert type(node.func) == ast.Name
            return loma_ir.Call(node.func.id,
                                [visit_expr(arg) for arg in node.args],
                                lineno = node.lineno)
        case None:
            return None
        case _:
            assert False, f'Unknown expr {type(node).__name__}'

def check_struct(t : loma_ir.type):
    # check if the struct has zero length member, recursively
    if isinstance(t, loma_ir.Int) or isinstance(t, loma_ir.Float):
        return False
    if isinstance(t, loma_ir.Array):
        return check_struct(t.t)
    if len(t.members) == 0:
        return True
    for m in t.members:
        if isinstance(m.t, loma_ir.Struct) or \
           isinstance(m.t, loma_ir.Array):
            if check_struct(m.t):
                return True
    return False

def check_structs(structs : dict[str, loma_ir.Struct]):
    for s in structs.values():
        if check_struct(s):
            return True
    return False

def fill_structs(s : loma_ir.Struct,
                 structs : dict[str, loma_ir.Struct]):
    assert isinstance(s, loma_ir.Struct)
    new_members = []
    for m in s.members:
        new_m = m
        if isinstance(m.t, loma_ir.Struct):
            new_m = attrs.evolve(m, t=structs[m.t.id])
        elif isinstance(m.t, loma_ir.Array):
            new_m = attrs.evolve(m,
                t=loma_ir.Array(structs[m.t.t.id], m.t.static_size))
        new_members.append(new_m)
    return attrs.evolve(s, members=new_members)

def parse(code : str) -> tuple[dict[str, loma_ir.Struct], dict[str, loma_ir.func]]:
    """ Given a loma frontend code represented as a string,
        convert the code to loma IR.
        Returns both the parsed loma Structs and functions.
    """
    module = ast.parse(code)

    structs = {}
    for d in module.body:
        if isinstance(d, ast.ClassDef):
            s = visit_ClassDef(d)
            structs[s.id] = s

    # Fill in struct information, run until converge
    while check_structs(structs):
        for k, s in structs.items():
            structs[k] = fill_structs(s, structs)

    funcs = {}
    for d in module.body:
        if isinstance(d, ast.FunctionDef):
            f = visit_FunctionDef(d)
            funcs[f.id] = f
            if "integrand" in f.id:
                eval_id = f.id.replace('integrand', 'eval')
                print(f"Found a integrand function: {f.id},", 
                    f"will auto create its eval function {eval_id}")
                funcs[eval_id] = integrand_eval_converter(f)
            # print(f"parser.parse line 375, CHECK f.id: {f.id}")
        elif isinstance(d, ast.Assign):
            f = visit_Differentiate(d)
            funcs[f.id] = f

    return structs, funcs

def integrand_eval_converter(f: loma_ir.FunctionDef) -> loma_ir.FunctionDef:
    """Given a integrand function (user defined), "manually" construct its eval function
    e.g. integrand_pd1(...) -> eval_pd1(lower, upper, ...)

    The actual definition should look like:
    
    def IntegralEval(lower: In[float], upper: In[float], t: In[float]) -> float:
        curr_x: float = lower
        n: int = (upper - lower) / 0.01 + 1
        i: int = 0
        res: float = 0.0
        while (i < n, max_iter := 10000):
            res = res + integrand_pd(curr_x, t)
            i = i + 1
            curr_x = curr_x + 0.01
        res = res * (upper - lower) / n
        return res

    Args:
        f (loma_ir.FunctionDef): integrand_pd1(x, t)

    Returns:
        loma_ir.FunctionDef: eval_pd1(lower, upper, t)
    """
    # function name
    _id = f.id.replace('integrand', 'eval')

    # args: we assume the first arg to be the integration variable
    _args = [
        loma_ir.Arg("lower", t=loma_ir.Float(), i=loma_ir.In()),
        loma_ir.Arg("upper", t=loma_ir.Float(), i=loma_ir.In())
    ]
    _args += f.args[1:]
    
    # prepare some variables to construct body
    lower = loma_ir.Var("lower", t=loma_ir.Float())
    upper = loma_ir.Var("upper", t=loma_ir.Float())
    upper_minus_lower = loma_ir.BinaryOp(loma_ir.Sub(), upper, lower)
    curr_x = loma_ir.Var("curr_x", t=loma_ir.Float())
    n = loma_ir.Var("n", t=loma_ir.Int())
    i = loma_ir.Var("i", t=loma_ir.Int())
    res = loma_ir.Var("res", t=loma_ir.Float())

    # body
    _body = [
        loma_ir.Declare("curr_x", t=loma_ir.Float(), val=lower),
        loma_ir.Declare("n", t=loma_ir.Int(), val=loma_ir.BinaryOp(
            loma_ir.Add(), 
            left=loma_ir.BinaryOp(
                loma_ir.Div(),
                left=upper_minus_lower,
                right=loma_ir.ConstFloat(0.01),
                t=loma_ir.Float()
            ),
            right=loma_ir.ConstInt(1),
            t=loma_ir.Int()
        )),  # n: int = (upper - lower) / 0.01 + 1
        loma_ir.Declare("i", t=loma_ir.Int(), val=loma_ir.ConstInt(0)),
        loma_ir.Declare("res", t=loma_ir.Float(), val=loma_ir.ConstFloat(0.0))
    ]  # the first 4 declarations
    # construct while loop
    _call_args: list[loma_ir.expr] = [curr_x] + [loma_ir.Var(arg.id) for arg in f.args[1:]]
    _while_stmt = [
        loma_ir.Assign(
            res,
            loma_ir.BinaryOp(
                loma_ir.Add(),
                left=res,
                right=loma_ir.Call(f.id, _call_args, t=loma_ir.Float())
            )
        ),  # res = res + integrand_pd(curr_x, t)
        loma_ir.Assign(i, loma_ir.BinaryOp(loma_ir.Add(), left=i, right=loma_ir.ConstInt(1))),
        loma_ir.Assign(curr_x, loma_ir.BinaryOp(loma_ir.Add(), left=curr_x, right=loma_ir.ConstFloat(0.01))),
    ]
    _body.append(loma_ir.While(
        cond=loma_ir.BinaryOp(loma_ir.Less(), left=i, right=n),
        max_iter=10000,
        body=_while_stmt
    ))
    _body += [
        loma_ir.Assign(
            target=res,
            val=loma_ir.BinaryOp(
                loma_ir.Div(),
                left=loma_ir.BinaryOp(loma_ir.Mul(), left=res, right=upper_minus_lower),
                right=n,
                t=loma_ir.Float()
            )
        ),  # res = res * (upper - lower) / n
        loma_ir.Return(val=res)
    ]

    return loma_ir.FunctionDef(
        id=_id,
        args=_args,
        body=_body,
        is_simd=f.is_simd,
        ret_type=loma_ir.Float()
    )

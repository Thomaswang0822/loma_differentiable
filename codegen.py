import attrs
import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import visitor

def type_to_string(node):
    match node:
        case loma_ir.Int():
            return 'int'
        case loma_ir.Float():
            return 'float'
        case loma_ir.Array():
            return type_to_string(node.t) + '*'
        case loma_ir.Struct():
            return node.id
        case None:
            return 'void'
        case _:
            assert False

def codegen(structs, funcs):
    @attrs.define()
    class CGVisitor(visitor.IRVisitor):
        code = ''
        tab_count = 0

        def emit_tabs(self):
            self.code += '\t' * self.tab_count

        def visit_function(self, node):
            self.code += f'extern \"C\" {type_to_string(node.ret_type)} {node.id}('
            for i, arg in enumerate(node.args):
                if i > 0:
                    self.code += ', '
                self.code += f'{type_to_string(arg.t)} {arg.id}'
            self.code += ') {\n'
            self.tab_count += 1
            for stmt in node.body:
                self.visit_stmt(stmt)
            self.tab_count -= 1
            self.code += '}\n'        

        def visit_return(self, ret):
            self.emit_tabs()
            self.code += f'return {self.visit_expr(ret.val)};\n'

        def visit_declare(self, dec):
            self.emit_tabs()
            self.code += f'{type_to_string(dec.t)} {dec.target}'
            expr_str = self.visit_expr(dec.val)
            if expr_str != '':
                self.code += f' = {expr_str}'
            self.code += ';\n'

        def visit_assign(self, ass):
            self.emit_tabs()
            self.code += self.visit_lhs(ass.target)
            expr_str = self.visit_expr(ass.val)
            if expr_str != '':
                self.code += f' = {expr_str}'
            self.code += ';\n'

        def visit_ifelse(self, ifelse):
            self.emit_tabs()
            self.code += f'if ({self.visit_expr(ifelse.cond)}) {{\n'
            self.tab_count += 1
            for stmt in ifelse.then_stmts:
                self.visit_stmt(stmt)
            self.tab_count -= 1
            self.emit_tabs()
            self.code += f'}} else {{\n'
            self.tab_count += 1
            for stmt in ifelse.else_stmts:
                self.visit_stmt(stmt)
            self.tab_count -= 1
            self.emit_tabs()
            self.code += '}\n'

        def visit_while(self, loop):
            self.emit_tabs()
            self.code += f'while ({self.visit_expr(loop.cond)}) {{\n'
            self.tab_count += 1
            for stmt in loop.body:
                self.visit_stmt(stmt)
            self.tab_count -= 1
            self.emit_tabs()
            self.code += '}\n'

        def visit_expr(self, expr):
            match expr:
                case loma_ir.Var():
                    return expr.id
                case loma_ir.ArrayAccess():
                    return f'{expr.id}[{self.visit_expr(expr.index)}]'
                case loma_ir.StructAccess():
                    return f'{expr.struct_id}.{expr.member_id}'
                case loma_ir.ConstFloat():
                    return f'(float)({expr.val})'
                case loma_ir.ConstInt():
                    return f'(int)({expr.val})'
                case loma_ir.Add():
                    return f'({self.visit_expr(expr.left)}) + ({self.visit_expr(expr.right)})'
                case loma_ir.Sub():
                    return f'({self.visit_expr(expr.left)}) - ({self.visit_expr(expr.right)})'
                case loma_ir.Mul():
                    return f'({self.visit_expr(expr.left)}) * ({self.visit_expr(expr.right)})'
                case loma_ir.Div():
                    return f'({self.visit_expr(expr.left)}) / ({self.visit_expr(expr.right)})'
                case loma_ir.Compare():
                    match expr.op:
                        case loma_ir.Less():
                            return f'({self.visit_expr(expr.left)}) < ({self.visit_expr(expr.right)})'
                        case loma_ir.LessEqual():
                            return f'({self.visit_expr(expr.left)}) <= ({self.visit_expr(expr.right)})'
                        case loma_ir.Greater():
                            return f'({self.visit_expr(expr.left)}) > ({self.visit_expr(expr.right)})'
                        case loma_ir.GreaterEqual():
                            return f'({self.visit_expr(expr.left)}) >= ({self.visit_expr(expr.right)})'
                        case loma_ir.Equal():
                            return f'({self.visit_expr(expr.left)}) == ({self.visit_expr(expr.right)})'
                        case loma_ir.And():
                            return f'({self.visit_expr(expr.left)}) && ({self.visit_expr(expr.right)})'
                        case loma_ir.Or():
                            return f'({self.visit_expr(expr.left)}) || ({self.visit_expr(expr.right)})'
                        case _:
                            assert False
                case loma_ir.Call():
                    ret = f'{expr.id}('
                    for arg in expr.args:
                        ret += self.visit_expr(arg)
                    ret += ')'
                    return ret
                case None:
                    return ''
                case _:
                    assert False, f'Visitor error: unhandled expression {expr}'

        def visit_lhs(self, lhs):
            match lhs:
                case loma_ir.LHSName():
                    return lhs.id
                case loma_ir.LHSArray():
                    return self.visit_lhs(lhs.array) + f'[{self.visit_expr(lhs.index)}]'
                case loma_ir.LHSStruct():
                    return self.visit_lhs(lhs.struct) + f'.{lhs.member}'
                case _:
                    assert False, f'Visitor error: unhandled lhs {lhs}'

    # Forward declaration of structs
    code = ''
    for s in structs.values():
        code += f'struct {s.id};\n'
    # Forward declaration of functions
    for f in funcs.values():
        code += f'extern \"C\" {type_to_string(f.ret_type)} {f.id}('
        for i, arg in enumerate(f.args):
            if i > 0:
                code += ', '
            code += f'{type_to_string(arg.t)} {arg.id}'
        code += ');\n'
    
    # Actual definition of structs
    for s in structs.values():
        code += f'struct {s.id} {{\n'
        for m in s.members:
            code += f'\t{type_to_string(m.t)} {m.id};\n'
        code += f'}};\n'
    
    for f in funcs.values():
        cg = CGVisitor()
        cg.visit_function(f)
        code += cg.code
    return code

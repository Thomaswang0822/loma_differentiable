from asdl_gen import ADT

def generate_asdl_file():
    ADT("""
    module loma {
      stmt = Assign     ( lhs target, expr val )
           | Declare    ( string target, type t, expr? val )
           | Return     ( expr val )
           | IfElse     ( expr cond, stmt* then_stmts, stmt* else_stmts )
           | While      ( expr cond, stmt* body )
           attributes   ( int? lineno )

      expr = Var          ( string id )
           | ArrayAccess  ( string id, expr index )
           | StructAccess ( string struct_id, string member_id )
           | ConstFloat   ( float val )
           | ConstInt     ( int val )
           | Add          ( expr left, expr right )
           | Sub          ( expr left, expr right )
           | Mul          ( expr left, expr right )
           | Div          ( expr left, expr right )
           | Compare      ( cmp_op op, expr left, expr right )
           | Call         ( string id, expr* args )
           attributes     ( int? lineno, type? t )

      lhs = LHSName   ( string id )
          | LHSArray  ( lhs array, expr index )
          | LHSStruct ( lhs struct, string member )

      func = FunctionDef ( string id, arg* args, stmt* body, type? ret_type)
             attributes  ( int? lineno )

      arg  = Arg ( string id, type t, inout i )

      type = Int    ( )
           | Float  ( )
           | Array  ( type t )
           | Struct ( string id, struct_member* members, int? lineno )

      struct_member = MemberDef ( string id, type t )

      cmp_op = Less()
             | LessEqual()
             | Greater()
             | GreaterEqual()
             | Equal()
             | And()
             | Or()

      inout = In() | Out()
    }
    """,
    header= '',
    ext_types = {},
    memoize = ['Var'])

import contextlib
import dataclasses
import enum
import re
import sys
import traceback
from typing import Callable, NoReturn


def die(message: str, line: int | None = None) -> NoReturn:
    location = f" on line {line + 1}" if line is not None else ""
    print("\n" + "-" * 30 + "\n", file=sys.stderr)
    traceback.print_stack()
    print("\n" + "-" * 30 + "\n", file=sys.stderr)
    print(f"error{location}: {message}", file=sys.stderr)
    sys.exit(1)


_emit_disabled = False


def emit(code: str) -> None:
    if not _emit_disabled:
        print(code)


@contextlib.contextmanager
def no_emit():
    global _emit_disabled
    try:
        _emit_disabled = True
        yield
    finally:
        _emit_disabled = False


class TokenKind(enum.Enum):
    Invalid = "Invalid"
    Eof = "Eof"
    Type = "Type"
    Name = "Name"
    IntConst = "IntConst"
    If = "if"
    Else = "else"
    While = "while"
    Do = "do"
    For = "for"
    Return = "return"
    OpenParen = "("
    CloseParen = ")"
    OpenCurly = "{"
    CloseCurly = "}"
    Semicolon = ";"
    Equals = "="
    Plus = "+"
    Minus = "-"
    Star = "*"
    Slash = "/"
    Percent = "%"
    Ampersand = "&"
    Pipe = "|"
    Caret = "^"
    Shl = "<<"
    Shr = ">>"
    Comma = ","


@dataclasses.dataclass
class Token:
    kind: TokenKind
    content: str
    line: int


class Lexer:
    def __init__(self, src: str, loc=0, line=0, types: list[str] = ["int"]) -> None:
        self.src = src
        self.loc = loc
        self.line = line
        self.types = types  # TODO: lexer hack

    def clone(self) -> "Lexer":
        return Lexer(self.src, self.loc, self.line, self.types)

    def skip_ws(self) -> None:
        while self.loc < len(self.src) and self.src[self.loc] in " \t\n":
            if self.src[self.loc] == "\n":
                self.line += 1
            self.loc += 1

    def peek(self) -> Token:
        self.skip_ws()
        if self.loc >= len(self.src):
            return Token(
                kind=TokenKind.Eof,
                content="",
                line=self.line,
            )
        # basic literal tokens
        for token_kind in TokenKind:
            if token_kind in (
                TokenKind.Invalid,
                TokenKind.Eof,
                TokenKind.Type,
                TokenKind.Name,
                TokenKind.IntConst,
            ):
                continue  # non-literals
            if self.src[self.loc :].startswith(token_kind.value):
                return Token(
                    kind=token_kind,
                    content=token_kind.value,
                    line=self.line,
                )
        # complex tokens
        m = re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*", self.src[self.loc :])
        if m is not None:
            return Token(
                kind=TokenKind.Type if m.group(0) in self.types else TokenKind.Name,
                content=m.group(0),
                line=self.line,
            )
        m = re.match(r"^[0-9]+", self.src[self.loc :])
        if m is not None:
            return Token(
                kind=TokenKind.IntConst,
                content=m.group(0),
                line=self.line,
            )
        return Token(
            kind=TokenKind.Invalid,
            content=self.src[self.loc : self.loc + 10],
            line=self.line,
        )

    def next(self, kind: TokenKind | None = None) -> Token:
        token = self.peek()
        if kind is not None and token.kind != kind:
            die(f"expected {kind.value}, got {token.content!r}", self.line)
        if token.kind != TokenKind.Invalid:
            self.loc += len(token.content)
        return token

    def try_next(self, kind: TokenKind) -> Token | None:
        if self.peek().kind != kind:
            return None
        return self.next()


@dataclasses.dataclass
class CType:
    token: Token
    # 0 = not a pointer, 1 = int *x, 2 = int **x, etc.
    pointer_level: int

    @property
    def typename(self) -> str:
        return self.token.content


def parse_pointer_level(lexer: Lexer) -> int:
    pointer_level = 0
    while lexer.try_next(TokenKind.Star):
        pointer_level += 1
    return pointer_level


def parse_type(lexer: Lexer) -> CType:
    return CType(lexer.next(TokenKind.Type), parse_pointer_level(lexer))


def ctype_to_wasmtype(c_type: CType) -> str:
    if c_type.pointer_level > 0 or c_type.typename == "int":
        return "i32"
    else:
        die(f"unknown type: {c_type.typename}", c_type.token.line)


def sizeof_wasmtype(wasmtype: str) -> int:
    if wasmtype in ("i32", "f32"):
        return 4
    elif wasmtype in ("i64", "f64"):
        return 8
    else:
        die(f"unrecognized wasmtype: {wasmtype}")


@dataclasses.dataclass
class Variable:
    name: str
    type: CType


@dataclasses.dataclass
class FrameSlot:
    variable: Variable
    local_offset: int


class StackFrame:
    def __init__(self, parent: "StackFrame | None" = None):
        self.parent = parent
        self.variables: dict[str, FrameSlot] = {}
        self.frame_size = 0
        self.frame_offset = 0
        if parent is not None:
            self.frame_offset = parent.frame_offset + parent.frame_size

    def add_var(self, name: str, type: CType) -> None:
        self.variables[name] = FrameSlot(Variable(name, type), self.frame_size)
        self.frame_size += sizeof_wasmtype(ctype_to_wasmtype(type))

    def lookup_var_and_offset(self, name: str) -> tuple[FrameSlot, int] | None:
        if name in self.variables:
            slot = self.variables[name]
            return slot, self.frame_offset + slot.local_offset
        elif self.parent is not None:
            return self.parent.lookup_var_and_offset(name)
        else:
            return None

    def get_offset(self, name: Token) -> int:
        slot_and_offset = self.lookup_var_and_offset(name.content)
        if slot_and_offset is None:
            die(f"unknown variable {name.content}", name.line)
        return slot_and_offset[1]


def emit_return(frame: StackFrame) -> None:
    emit(f"    ;; return--adjust stack pointer")
    emit(f"    global.get $__stack_pointer")
    emit(f"    i32.const {frame.frame_size}")
    emit(f"    i32.add")
    emit(f"    global.set $__stack_pointer")
    emit(f"    return")


class ExprResultKind(enum.Enum):
    Value = "Value"
    # a place (corresponding to an address) that can be loaded from / stored to
    # not all addresses are places, e.g. &x is a value (&x = 1 is meaningless)
    Place = "Place"


def load_result(erk: ExprResultKind) -> None:
    if erk == ExprResultKind.Place:
        emit("    i32.load")


def expression(lexer: Lexer, frame: StackFrame) -> ExprResultKind:
    def value() -> ExprResultKind:
        if const := lexer.try_next(TokenKind.IntConst):
            emit(f"    i32.const {const.content}")
            return ExprResultKind.Value
        elif varname := lexer.try_next(TokenKind.Name):
            emit(f"    ;; load {varname.content}")
            emit(f"    global.get $__stack_pointer")
            emit(f"    i32.const {frame.get_offset(varname)}")
            emit(f"    i32.add")
            return ExprResultKind.Place
        elif lexer.try_next(TokenKind.Minus):
            if lexer.peek().kind == TokenKind.Minus:
                die("predecrement not supported", lexer.line)
            load_result(value())
            emit(f"    i32.neg")
            return ExprResultKind.Value
        elif lexer.try_next(TokenKind.Star):
            load_result(value())
            return ExprResultKind.Place
        elif lexer.try_next(TokenKind.Ampersand):
            expr_kind = value()
            if expr_kind != ExprResultKind.Place:
                die("cannot take reference to value", lexer.line)
            return ExprResultKind.Value
        elif lexer.try_next(TokenKind.OpenParen):
            expr_kind = expression(lexer, frame)
            lexer.next(TokenKind.CloseParen)
            return expr_kind
        else:
            die("expected value", lexer.line)

    def makeop(
        higher: Callable[[], ExprResultKind], ops: dict[TokenKind, str]
    ) -> Callable[[], ExprResultKind]:
        def op() -> ExprResultKind:
            higher_kind = higher()
            if lexer.peek().kind in ops.keys():
                load_result(higher_kind)
                op = lexer.next()
                load_result(higher())
                emit(f"    {ops[op.kind]}")
                return ExprResultKind.Value
            return higher_kind

        return op

    muldiv = makeop(
        value,
        {
            TokenKind.Star: "i32.mul",
            TokenKind.Slash: "i32.div_s",
            TokenKind.Percent: "i32.rem_s",
        },
    )
    plusminus = makeop(muldiv, {TokenKind.Plus: "i32.add", TokenKind.Minus: "i32.sub"})
    shlr = makeop(plusminus, {TokenKind.Shl: "i32.shl", TokenKind.Shr: "i32.shr_s"})
    bitand = makeop(shlr, {TokenKind.Ampersand: "i32.and"})
    bitor = makeop(bitand, {TokenKind.Pipe: "i32.or"})
    xor = makeop(bitor, {TokenKind.Caret: "i32.xor"})

    def assign() -> ExprResultKind:
        lhs_kind = xor()
        if lexer.try_next(TokenKind.Equals):
            if lhs_kind != ExprResultKind.Place:
                die("lhs of assignment cannot be value", lexer.line)
            emit(f"    call $__dup_i32")  # save addr
            load_result(assign())
            emit(f"    i32.store")
            emit(f"    i32.load")  # use dup'd addr
            return ExprResultKind.Value
        return lhs_kind

    return assign()


def bracketed_block_or_single_statement(lexer: Lexer, frame: StackFrame) -> None:
    if lexer.try_next(TokenKind.OpenCurly):
        while lexer.try_next(TokenKind.CloseCurly) is None:
            statement(lexer, frame)
    else:
        statement(lexer, frame)


def statement(lexer: Lexer, frame: StackFrame) -> None:
    if lexer.try_next(TokenKind.Return):
        if lexer.peek().kind != TokenKind.Semicolon:
            load_result(expression(lexer, frame))
        lexer.next(TokenKind.Semicolon)
        emit_return(frame)
    elif lexer.try_next(TokenKind.If):
        emit("    ;; if")
        emit("    block")  # for else
        emit("    block")
        lexer.next(TokenKind.OpenParen)
        load_result(expression(lexer, frame))
        lexer.next(TokenKind.CloseParen)
        emit("    i32.eqz")
        emit("    br_if 0")  # exit into else block
        emit("    ;; if body")
        bracketed_block_or_single_statement(lexer, frame)
        emit("    br 1")  # skip to end of else block
        emit("    end")
        if lexer.try_next(TokenKind.Else):
            if lexer.try_next(TokenKind.If):
                die("else if not supported", lexer.line)
            bracketed_block_or_single_statement(lexer, frame)
        emit("    end")
    elif lexer.try_next(TokenKind.While):
        emit(";; while")
        emit("block")
        emit("loop")
        lexer.next(TokenKind.OpenParen)
        load_result(expression(lexer, frame))
        lexer.next(TokenKind.CloseParen)
        emit("    i32.eqz")
        emit("    br_if 1")  # exit loop by jumping forward to end of enclosing block
        emit("    ;; while body")
        bracketed_block_or_single_statement(lexer, frame)
        emit("    br 0")  # jump to beginning of loop
        emit("    end")
        emit("    end")
    elif lexer.try_next(TokenKind.Do):
        emit("    ;; do-while")
        emit("    block")
        emit("    loop")
        bracketed_block_or_single_statement(lexer, frame)
        lexer.next(TokenKind.While)
        lexer.next(TokenKind.OpenParen)
        load_result(expression(lexer, frame))
        lexer.next(TokenKind.CloseParen)
        emit("    i32.eqz")
        emit("    br_if 1")  # exit loop by jumping forward to end of enclosing block
        emit("    br 0")  # otherwise jump to beginning of loop
        emit("    end")
        emit("    end")
        lexer.next(TokenKind.Semicolon)
    elif lexer.try_next(TokenKind.For):
        lexer.next(TokenKind.OpenParen)
        emit("    ;; for")
        emit("    block")
        if lexer.peek().kind != TokenKind.Semicolon:
            emit("    ;; initializer")
            expression(lexer, frame)
            emit("    drop")
        lexer.next(TokenKind.Semicolon)
        emit("    loop")
        if lexer.peek().kind != TokenKind.Semicolon:
            emit("    ;; test")
            load_result(expression(lexer, frame))
            emit("    i32.eqz")
            emit("    br_if 1")
        lexer.next(TokenKind.Semicolon)
        saved_lexer = None
        if lexer.peek().kind != TokenKind.CloseParen:
            # the nastiest hack every hack'd
            saved_lexer = lexer.clone()
            with no_emit():
                expression(lexer, frame)  # advance past expr
        lexer.next(TokenKind.CloseParen)
        bracketed_block_or_single_statement(lexer, frame)
        if saved_lexer != None:
            emit("    ;; advancement (redux, for real this time)")
            expression(saved_lexer, frame)
        emit("    br 0")
        emit("    end")
        emit("    end")
    elif lexer.try_next(TokenKind.Semicolon):
        pass
    else:
        expression(lexer, frame)
        lexer.next(TokenKind.Semicolon)
        emit("    drop")


def variable_declaration(lexer: Lexer, frame: StackFrame) -> None:
    type = parse_type(lexer)
    varname = lexer.next(TokenKind.Name)
    frame.add_var(varname.content, type)

    while lexer.try_next(TokenKind.Comma):
        pointer_level = parse_pointer_level(lexer)
        varname = lexer.next(TokenKind.Name)
        frame.add_var(varname.content, CType(type.token, pointer_level))

    lexer.next(TokenKind.Semicolon)


def func_decl(lexer: Lexer) -> None:
    return_type = parse_type(lexer)
    function_name = lexer.next(TokenKind.Name)

    frame = StackFrame()
    # parameters (TODO)
    lexer.next(TokenKind.OpenParen)
    lexer.next(TokenKind.CloseParen)

    lexer.next(TokenKind.OpenCurly)

    # declarations (up top, c89 only yolo)
    while lexer.peek().kind == TokenKind.Type:
        variable_declaration(lexer, frame)

    emit(f"  (func ${function_name.content} (result {ctype_to_wasmtype(return_type)})")
    emit(f"    ;; prelude--adjust stack pointer (grows down)")
    emit(f"    global.get $__stack_pointer")
    emit(f"    i32.const {frame.frame_size}")
    emit(f"    i32.sub")
    emit(f"    global.set $__stack_pointer")
    emit(f"    ;; end prelude")

    while lexer.peek().kind != TokenKind.CloseCurly:
        statement(lexer, frame)
    lexer.next(TokenKind.CloseCurly)

    # wasmer seems to not understand that
    # `(func $x (result i32) block i32.const 0 return end)` doesn't have an implicit
    # return, so this is only there to provide a dummy stack value for the validator
    emit(f"    i32.const 0xdeadb33f ;; validator hack")
    # TODO: for void functions we need to add an addl emit_return for implicit returns
    emit(f"  )")


def compile(src: str) -> None:
    # prelude
    emit("(module")
    emit("  (memory 2)")
    emit("  (global $__stack_pointer (mut i32) (i32.const 66560))")
    emit("  (func $__dup_i32 (param i32) (result i32 i32)")
    emit("    (local.get 0)")
    emit("    (local.get 0))")

    lexer = Lexer(src)

    while lexer.peek().kind != TokenKind.Eof:
        func_decl(lexer)

    emit('  (export "main" (func $main))')

    # postlude
    emit(")")


if __name__ == "__main__":
    import fileinput

    with fileinput.input(encoding="utf-8") as fi:
        compile("".join(fi))  # todo: make this line-at-a-time?

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
_emit_indent = 0


def emit(code: str) -> None:
    if not _emit_disabled:
        print(" " * _emit_indent + code)


@contextlib.contextmanager
def emit_block(start: str, end: str):
    global _emit_indent
    emit(start)
    _emit_indent += 2
    yield
    _emit_indent -= 2
    emit(end)


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
    OpenSq = "["
    CloseSq = "]"
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
    typename_token: Token
    pointer_level: int  # 0 = not a pointer, 1 = int *x, 2 = int **x, etc.
    array_size_token: Token | None  # None = not an array, because 0 is a valid array size...

    @property
    def typename(self) -> str:
        return self.typename_token.content

    @property
    def array_size(self) -> int | None:
        return int(self.array_size_token.content) if self.array_size_token else None


def parse_type_and_name(lexer: Lexer, type: Token | None = None) -> tuple[CType, Token]:
    if type is None:
        type = lexer.next(TokenKind.Type)

    pointer_level = 0
    while lexer.try_next(TokenKind.Star):
        pointer_level += 1

    varname = lexer.next(TokenKind.Name)

    if lexer.try_next(TokenKind.OpenSq):
        array_size = lexer.next(TokenKind.IntConst)
        lexer.next(TokenKind.CloseSq)
    else:
        array_size = None

    return CType(type, pointer_level, array_size), varname


def ctype_to_wasmtype(c_type: CType) -> str:
    is_pointy = c_type.pointer_level > 0 or c_type.array_size is not None
    if is_pointy or c_type.typename == "int":
        return "i32"
    else:
        die(f"unknown type: {c_type.typename}", c_type.typename_token.line)


def sizeof_c_type(c_type: CType) -> int:
    wasmtype = ctype_to_wasmtype(c_type)
    if wasmtype in ("i32", "f32"):
        return 4 * (c_type.array_size or 1)
    elif wasmtype in ("i64", "f64"):
        return 8 * (c_type.array_size or 1)
    else:
        die(f"unrecognized ctype: {c_type} ({wasmtype})")


@dataclasses.dataclass
class Variable:
    name: str
    type: CType


@dataclasses.dataclass
class FrameSlot:
    variable: Variable
    local_offset: int
    is_parameter: bool


class StackFrame:
    def __init__(self, parent: "StackFrame | None" = None):
        self.parent = parent
        self.variables: dict[str, FrameSlot] = {}
        self.frame_size = 0
        self.frame_offset = 0
        if parent is not None:
            self.frame_offset = parent.frame_offset + parent.frame_size

    def add_var(self, name: str, type: CType, is_parameter: bool = False) -> None:
        self.variables[name] = FrameSlot(
            Variable(name, type), self.frame_size, is_parameter
        )
        self.frame_size += sizeof_c_type(type)

    def lookup_var_and_offset(self, name: str) -> tuple[FrameSlot, int] | None:
        if name in self.variables:
            slot = self.variables[name]
            return slot, self.frame_offset + slot.local_offset
        elif self.parent is not None:
            return self.parent.lookup_var_and_offset(name)
        else:
            return None

    def get_offset(self, name: Token | str) -> int:
        n = name if isinstance(name, str) else name.content
        slot_and_offset = self.lookup_var_and_offset(n)
        if slot_and_offset is None:
            die(f"unknown variable {n}", None if isinstance(name, str) else name.line)
        return slot_and_offset[1]


def emit_return(frame: StackFrame) -> None:
    emit(";; return--adjust stack pointer")
    emit("global.get $__stack_pointer")
    emit(f"i32.const {frame.frame_size}")
    emit("i32.add")
    emit("global.set $__stack_pointer")
    emit("return")


class ExprResultKind(enum.Enum):
    Value = "Value"
    # a place (corresponding to an address) that can be loaded from / stored to
    # not all addresses are places, e.g. &x is a value (&x = 1 is meaningless)
    Place = "Place"


def load_result(erk: ExprResultKind) -> None:
    if erk == ExprResultKind.Place:
        emit("i32.load")


def expression(lexer: Lexer, frame: StackFrame) -> ExprResultKind:
    def value() -> ExprResultKind:
        if const := lexer.try_next(TokenKind.IntConst):
            emit(f"i32.const {const.content}")
            return ExprResultKind.Value
        elif lexer.try_next(TokenKind.OpenParen):
            expr_kind = expression(lexer, frame)
            lexer.next(TokenKind.CloseParen)
            return expr_kind
        elif varname := lexer.try_next(TokenKind.Name):
            if lexer.try_next(TokenKind.OpenParen):
                if lexer.peek().kind != TokenKind.CloseParen:
                    while True:
                        load_result(expression(lexer, frame))
                        if not lexer.try_next(TokenKind.Comma):
                            break
                lexer.next(TokenKind.CloseParen)
                emit(f"call ${varname.content}")
                return ExprResultKind.Value
            else:
                emit(f";; load {varname.content}")
                emit("global.get $__stack_pointer")
                emit(f"i32.const {frame.get_offset(varname)}")
                emit("i32.add")
                return ExprResultKind.Place
        else:
            die("expected value", lexer.line)

    def accessor() -> ExprResultKind:
        lhs_kind = value()  # TODO: this is wrong for x[0][0], right?
        if lexer.try_next(TokenKind.OpenSq):
            load_result(lhs_kind)
            load_result(expression(lexer, frame))
            lexer.next(TokenKind.CloseSq)
            emit("i32.const 4")  # TODO: this is wrong for non-4-byte types
            emit("i32.mul")
            emit("i32.add")
            return ExprResultKind.Place
        else:
            return lhs_kind

    def prefix() -> ExprResultKind:
        if lexer.try_next(TokenKind.Ampersand):
            if prefix() != ExprResultKind.Place:
                die("cannot take reference to value")
            return ExprResultKind.Value
        elif lexer.try_next(TokenKind.Star):
            load_result(prefix())
            return ExprResultKind.Place
        elif lexer.try_next(TokenKind.Minus):
            if lexer.peek().kind == TokenKind.Minus:
                die("predecrement not supported", lexer.line)
            load_result(prefix())
            emit("i32.neg")
            return ExprResultKind.Value
        elif lexer.try_next(TokenKind.Plus):
            if lexer.peek().kind == TokenKind.Plus:
                die("preincrement not supported", lexer.line)
            load_result(prefix())
            return ExprResultKind.Value
        else:
            return accessor()

    def makeop(
        higher: Callable[[], ExprResultKind], ops: dict[TokenKind, str]
    ) -> Callable[[], ExprResultKind]:
        def op() -> ExprResultKind:
            higher_kind = higher()
            if lexer.peek().kind in ops.keys():
                load_result(higher_kind)
                op_token = lexer.next()
                load_result(op())
                emit(f"{ops[op_token.kind]}")
                return ExprResultKind.Value
            return higher_kind

        return op

    muldiv = makeop(
        prefix,
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
            emit("call $__dup_i32")  # save addr
            load_result(assign())
            emit("i32.store")
            emit("i32.load")  # use dup'd addr
            return ExprResultKind.Value
        return lhs_kind

    return assign()


def bracketed_block_or_single_statement(lexer: Lexer, frame: StackFrame) -> None:
    if lexer.try_next(TokenKind.OpenCurly):
        while lexer.try_next(TokenKind.CloseCurly) is None:
            statement(lexer, frame)
    else:
        statement(lexer, frame)


def parenthesized_test(lexer: Lexer, frame: StackFrame) -> None:
    lexer.next(TokenKind.OpenParen)
    load_result(expression(lexer, frame))
    lexer.next(TokenKind.CloseParen)
    emit("i32.eqz")


def statement(lexer: Lexer, frame: StackFrame) -> None:
    if lexer.try_next(TokenKind.Return):
        if lexer.peek().kind != TokenKind.Semicolon:
            load_result(expression(lexer, frame))
        lexer.next(TokenKind.Semicolon)
        emit_return(frame)
    elif lexer.try_next(TokenKind.If):
        with emit_block("block ;; if statement", "end"):
            with emit_block("block", "end"):
                parenthesized_test(lexer, frame)
                emit("br_if 0 ;; jump to else")
                bracketed_block_or_single_statement(lexer, frame)
                emit("br 1 ;; exit if")  # skip to end of else block
            if lexer.try_next(TokenKind.Else):
                # single statement might be "if" of "else if"
                bracketed_block_or_single_statement(lexer, frame)
    elif lexer.try_next(TokenKind.While):
        with emit_block("block ;; while", "end"):
            with emit_block("loop", "end"):
                parenthesized_test(lexer, frame)
                emit("br_if 1 ;; exit loop")
                bracketed_block_or_single_statement(lexer, frame)
                emit("br 0 ;; repeat loop")
    elif lexer.try_next(TokenKind.Do):
        with emit_block("block ;; do-while", "end"):
            with emit_block("loop", "end"):
                bracketed_block_or_single_statement(lexer, frame)
                lexer.next(TokenKind.While)
                parenthesized_test(lexer, frame)
                emit("br_if 1 ;; exit loop")
                emit("br 0 ;; repeat loop")
                lexer.next(TokenKind.Semicolon)
    elif lexer.try_next(TokenKind.For):
        lexer.next(TokenKind.OpenParen)
        with emit_block("block ;; for", "end"):
            if lexer.peek().kind != TokenKind.Semicolon:
                emit(";; for initializer")
                expression(lexer, frame)
                emit("drop")
            lexer.next(TokenKind.Semicolon)
            with emit_block("loop", "end"):
                if lexer.peek().kind != TokenKind.Semicolon:
                    emit(";; for test")
                    load_result(expression(lexer, frame))
                    emit("i32.eqz")
                    emit("br_if 1 ;; exit loop")
                lexer.next(TokenKind.Semicolon)
                saved_lexer = None
                if lexer.peek().kind != TokenKind.CloseParen:
                    # save lexer position to emit advance stmt later (nasty hack)
                    saved_lexer = lexer.clone()
                    with no_emit():
                        expression(lexer, frame)  # advance past expr
                lexer.next(TokenKind.CloseParen)
                emit(";; for body")
                bracketed_block_or_single_statement(lexer, frame)
                if saved_lexer != None:
                    emit(";; for advancement")
                    expression(saved_lexer, frame)  # use saved lexer
                emit("br 0 ;; repeat loop")
    elif lexer.try_next(TokenKind.Semicolon):
        pass  # nothing to emit
    else:
        expression(lexer, frame)
        lexer.next(TokenKind.Semicolon)
        emit("drop")


def variable_declaration(lexer: Lexer, frame: StackFrame) -> None:
    type, varname = parse_type_and_name(lexer)
    frame.add_var(varname.content, type)

    while lexer.try_next(TokenKind.Comma):
        type, varname = parse_type_and_name(lexer, type=type.typename_token)
        frame.add_var(varname.content, type)

    lexer.next(TokenKind.Semicolon)


def func_decl(lexer: Lexer) -> None:
    rtype, name = parse_type_and_name(lexer)
    if rtype.array_size is not None:
        die("no function array return, nice try")

    frame = StackFrame()
    lexer.next(TokenKind.OpenParen)
    if lexer.peek().kind != TokenKind.CloseParen:
        while True:
            type, varname = parse_type_and_name(lexer)
            frame.add_var(varname.content, type, is_parameter=True)
            if not lexer.try_next(TokenKind.Comma):
                break
    lexer.next(TokenKind.CloseParen)

    lexer.next(TokenKind.OpenCurly)

    # declarations (up top, c89 only yolo)
    while lexer.peek().kind == TokenKind.Type:
        variable_declaration(lexer, frame)

    with emit_block(f"(func ${name.content}", ")"):
        for v in frame.variables.values():
            if v.is_parameter:
                emit(f"(param ${v.variable.name} {ctype_to_wasmtype(v.variable.type)})")
        emit(f"(result {ctype_to_wasmtype(rtype)})")
        emit(";; fn prelude")
        emit("global.get $__stack_pointer")
        emit(f"i32.const {frame.frame_size}")
        emit("i32.sub")
        emit("global.set $__stack_pointer")
        for v in reversed(frame.variables.values()):
            if v.is_parameter:
                emit("global.get $__stack_pointer")
                emit(f"i32.const {frame.get_offset(v.variable.name)}")
                emit("i32.add")
                emit(f"local.get ${v.variable.name}")
                emit("i32.store")

        while lexer.peek().kind != TokenKind.CloseCurly:
            statement(lexer, frame)
        lexer.next(TokenKind.CloseCurly)

        # wasmer seems to not understand that
        # `(func $x (result i32) block i32.const 0 return end)` doesn't have an implicit
        # return, so this is only there to provide a dummy stack value for the validator
        emit("i32.const 0xdeadb33f ;; validator hack")
        # TODO: for void functions we need to add an addl emit_return for implicit returns


def compile(src: str) -> None:
    with emit_block("(module", ")"):
        emit("(memory 2)")
        emit("(global $__stack_pointer (mut i32) (i32.const 66560))")
        emit("(func $__dup_i32 (param i32) (result i32 i32)")
        emit("  (local.get 0)")
        emit("  (local.get 0))")

        lexer = Lexer(src)
        while lexer.peek().kind != TokenKind.Eof:
            func_decl(lexer)

        emit('(export "main" (func $main))')


if __name__ == "__main__":
    import fileinput

    with fileinput.input(encoding="utf-8") as fi:
        compile("".join(fi))  # todo: make this line-at-a-time?

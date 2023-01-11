import contextlib
import dataclasses
import enum
import re
import sys
import traceback
from typing import Callable, NoReturn


def die(message: str, line: int | None = None) -> NoReturn:
    print("\n" + "-" * 30 + "\n", file=sys.stderr)
    traceback.print_stack()
    print("\n" + "-" * 30 + "\n", file=sys.stderr)
    location = f" on line {line + 1}" if line is not None else ""
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
    Typedef = "typedef"
    If = "if"
    Else = "else"
    While = "while"
    Do = "do"
    For = "for"
    Return = "return"
    Increment = "++"
    Decrement = "--"
    Shl = "<<"
    Shr = ">>"
    LogAnd = "&&"
    LogOr = "||"
    CmpEq = "=="
    CmpLtEq = "<="
    CmpGtEq = ">="
    CmpNeq = "!="
    CmpLt = "<"
    CmpGt = ">"
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
    Comma = ","
    Bang = "!"
    Tilde = "~"


@dataclasses.dataclass
class Token:
    kind: TokenKind
    content: str
    line: int


# default types
TYPES: dict[str, str] = {"int": "int"}


class Lexer:
    def __init__(self, src: str, loc=0, line=0, types: dict[str, str] = TYPES) -> None:
        self.src = src
        self.loc = loc
        self.line = line
        self.types = types

    def clone(self) -> "Lexer":
        return Lexer(self.src, self.loc, self.line, self.types.copy())

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


class CType:
    def __init__(
        self,
        typename: str,
        pointer_level: int = 0,
        array_size: int | None = None,
        line: int | None = None,
    ) -> None:
        self.typename = typename
        self.pointer_level = pointer_level
        self.array_size = array_size
        self.decl_line = line
        is_pointy = pointer_level > 0 or array_size is not None
        if is_pointy or typename == "int":
            self.wasmtype = "i32"
        else:
            die(f"unknown type: {typename}", line)

    def sizeof(self) -> int:
        if self.wasmtype in ("i32", "f32"):
            return 4 * (self.array_size or 1)
        return 8 * (self.array_size or 1)

    def is_ptr(self) -> bool:
        return self.pointer_level > 0

    def less_ptr(self) -> "CType":
        assert self.is_ptr(), f"bug: not a pointer: {self}"
        return CType(self.typename, self.pointer_level - 1, self.array_size)

    def more_ptr(self) -> "CType":
        return CType(self.typename, self.pointer_level + 1, self.array_size)

    def is_arr(self) -> bool:
        return self.array_size is not None

    def as_non_array(self) -> "CType":
        assert self.is_arr(), f"bug: not an array: {self}"
        return CType(self.typename, self.pointer_level, None)

    def __str__(self) -> str:
        s = self.typename + "*" * self.pointer_level
        if self.array_size is not None:
            s += f"[{self.array_size}]"
        return s


def parse_type_and_name(lexer: Lexer, type: str | None = None) -> tuple[CType, Token]:
    if type is None:
        type = lexer.types[lexer.next(TokenKind.Type).content] # resolve typedef

    pointer_level = 0
    while lexer.try_next(TokenKind.Star):
        pointer_level += 1

    varname = lexer.next(TokenKind.Name)

    if lexer.try_next(TokenKind.OpenSq):
        array_size = int(lexer.next(TokenKind.IntConst).content)
        lexer.next(TokenKind.CloseSq)
    else:
        array_size = None

    return CType(type, pointer_level, array_size, varname.line), varname


@dataclasses.dataclass
class FrameVar:
    name: str
    type: CType
    local_offset: int
    is_parameter: bool


class StackFrame:
    def __init__(self, parent: "StackFrame | None" = None):
        self.parent = parent
        self.variables: dict[str, FrameVar] = {}
        self.frame_size = 0
        self.frame_offset = 0
        if parent is not None:
            self.frame_offset = parent.frame_offset + parent.frame_size

    def add_var(self, name: str, type: CType, is_parameter: bool = False) -> None:
        self.variables[name] = FrameVar(name, type, self.frame_size, is_parameter)
        self.frame_size += type.sizeof()

    def get_var_and_offset(self, name: Token | str) -> tuple[FrameVar, int]:
        n = name if isinstance(name, str) else name.content
        if slot := self.variables.get(n):
            return slot, self.frame_offset + slot.local_offset
        elif self.parent is not None:
            return self.parent.get_var_and_offset(name)
        else:
            die(f"unknown variable {n}", None if isinstance(name, str) else name.line)


def emit_return(frame: StackFrame) -> None:
    emit(";; return--adjust stack pointer")
    emit("global.get $__stack_pointer")
    emit(f"i32.const {frame.frame_size}")
    emit("i32.add")
    emit("global.set $__stack_pointer")
    emit("return")


@dataclasses.dataclass
class ExprMeta:
    # a place (corresponding to an address) that can be loaded from / stored to
    # not all addresses are places, e.g. &x is a value (&x = 1 is meaningless)
    is_place: bool
    type: CType


def load_result(em: ExprMeta) -> ExprMeta:
    if em.is_place:
        emit("i32.load")
    return ExprMeta(False, em.type)


def expression(lexer: Lexer, frame: StackFrame) -> ExprMeta:
    def value() -> ExprMeta:
        if const := lexer.try_next(TokenKind.IntConst):
            emit(f"i32.const {const.content}")
            return ExprMeta(False, CType("int"))
        elif lexer.try_next(TokenKind.OpenParen):
            meta = expression(lexer, frame)
            lexer.next(TokenKind.CloseParen)
            return meta
        else:
            varname = lexer.next(TokenKind.Name)
            if lexer.try_next(TokenKind.OpenParen):
                if lexer.peek().kind != TokenKind.CloseParen:
                    while True:
                        load_result(expression(lexer, frame))
                        if not lexer.try_next(TokenKind.Comma):
                            break
                lexer.next(TokenKind.CloseParen)
                emit(f"call ${varname.content}")
                return ExprMeta(False, CType("int"))  # TODO!!
            else:
                var, offset = frame.get_var_and_offset(varname)
                emit(f";; load {varname.content}")
                emit("global.get $__stack_pointer")
                emit(f"i32.const {offset}")
                emit("i32.add")
                return ExprMeta(True, var.type)

    def accessor() -> ExprMeta:
        lhs_meta = value()  # TODO: this is wrong for x[0][0], right?
        if lexer.try_next(TokenKind.OpenSq):
            lhs_meta = load_result(lhs_meta)
            lhs_type = lhs_meta.type
            if not (lhs_type.is_arr() or lhs_type.is_ptr()):
                die(f"not an array or pointer: {lhs_meta.type}", lexer.line)
            el_type = (
                lhs_type.as_non_array() if lhs_type.is_arr() else lhs_type.less_ptr()
            )

            load_result(expression(lexer, frame))
            lexer.next(TokenKind.CloseSq)
            emit(f"i32.const {el_type.sizeof()}")
            emit("i32.mul")
            emit("i32.add")
            return ExprMeta(True, el_type)
        else:
            return lhs_meta

    def prefix() -> ExprMeta:
        if lexer.try_next(TokenKind.Ampersand):
            meta = prefix()
            if not meta.is_place:
                die("cannot take reference to value", lexer.line)
            return ExprMeta(False, meta.type.more_ptr())
        elif lexer.try_next(TokenKind.Star):
            meta = load_result(prefix())
            if not meta.type.is_ptr():
                die("cannot dereference non-pointer", lexer.line)
            return ExprMeta(True, meta.type.less_ptr())
        elif lexer.try_next(TokenKind.Minus):
            emit("i32.const 0")
            meta = load_result(prefix())
            emit("i32.sub")
            return meta
        elif lexer.try_next(TokenKind.Plus):
            return load_result(prefix())
        elif lexer.try_next(TokenKind.Bang):
            meta = load_result(prefix())
            emit("i32.eqz")
            return meta
        elif lexer.try_next(TokenKind.Tilde):
            meta = load_result(prefix())
            emit("i32.const 0xffffffff")
            emit("i32.xor")
            return meta
        else:
            return accessor()

    def makeop(
        higher: Callable[[], ExprMeta], ops: dict[TokenKind, str]
    ) -> Callable[[], ExprMeta]:
        def op() -> ExprMeta:
            lhs_meta = higher()
            if lexer.peek().kind in ops.keys():
                lhs_meta = load_result(lhs_meta)
                op_token = lexer.next()
                load_result(op())
                emit(f"{ops[op_token.kind]}")
                return ExprMeta(False, lhs_meta.type)
            return lhs_meta

        return op

    muldiv = makeop(
        prefix,
        {
            TokenKind.Star: "i32.mul",
            TokenKind.Slash: "i32.div_s",
            TokenKind.Percent: "i32.rem_s",
        },
    )

    def plusminus() -> ExprMeta:
        lhs_meta = muldiv()
        if lexer.peek().kind in (TokenKind.Plus, TokenKind.Minus):
            lhs_meta = load_result(lhs_meta)
            op_token = lexer.next()
            rhs_meta = load_result(plusminus())

            lhs_type, rhs_type, res_type = lhs_meta.type, rhs_meta.type, lhs_meta.type

            if lhs_meta.type.pointer_level == rhs_meta.type.pointer_level:
                pass
            elif lhs_meta.type.is_ptr() and rhs_meta.type.is_ptr():
                die(f"cannot {op_token.content} {lhs_meta.type} and {rhs_meta.type}")
            elif lhs_meta.type.is_ptr() and not rhs_meta.type.is_ptr():
                emit(f"i32.const {lhs_meta.type.less_ptr().sizeof()}")
                emit("i32.mul")
            elif not lhs_meta.type.is_ptr() and rhs_meta.type.is_ptr():
                res_type = rhs_meta.type
                emit("call $__swap_i32")
                emit(f"i32.const {rhs_meta.type.less_ptr().sizeof()}")
                emit("i32.mul")
                emit("call $__swap_i32")

            if op_token.kind == TokenKind.Plus:
                emit("i32.add")
            else:
                emit("i32.sub")
                if lhs_type.is_ptr() and rhs_type.is_ptr():
                    emit(f"i32.const {rhs_meta.type.less_ptr().sizeof()}")
                    emit(f"i32.div_s")

            return ExprMeta(False, res_type)
        return lhs_meta

    shlr = makeop(plusminus, {TokenKind.Shl: "i32.shl", TokenKind.Shr: "i32.shr_s"})
    cmplg = makeop(
        shlr,
        {
            TokenKind.CmpLt: "i32.lt_s",
            TokenKind.CmpGt: "i32.gt_s",
            TokenKind.CmpLtEq: "i32.le_s",
            TokenKind.CmpGtEq: "i32.ge_s",
        },
    )
    cmpe = makeop(cmplg, {TokenKind.CmpEq: "i32.eq", TokenKind.CmpNeq: "i32.ne"})
    bitand = makeop(cmpe, {TokenKind.Ampersand: "i32.and"})
    bitor = makeop(bitand, {TokenKind.Pipe: "i32.or"})
    xor = makeop(bitor, {TokenKind.Caret: "i32.xor"})

    def assign() -> ExprMeta:
        lhs_meta = xor()
        if lexer.try_next(TokenKind.Equals):
            if not lhs_meta.is_place:
                die("lhs of assignment cannot be value", lexer.line)
            emit("call $__dup_i32")  # save addr
            rhs_meta = load_result(assign())
            emit("i32.store")
            emit("i32.load")  # use dup'd addr
            return rhs_meta
        return lhs_meta

    return assign()


def bracketed_block_or_single_statement(lexer: Lexer, frame: StackFrame) -> None:
    if lexer.try_next(TokenKind.OpenCurly):
        while lexer.try_next(TokenKind.CloseCurly) is None:
            statement(lexer, frame)
    else:
        statement(lexer, frame)


def statement(lexer: Lexer, frame: StackFrame) -> None:
    def parenthesized_test() -> None:
        lexer.next(TokenKind.OpenParen)
        load_result(expression(lexer, frame))
        lexer.next(TokenKind.CloseParen)
        emit("i32.eqz")

    if lexer.try_next(TokenKind.Return):
        if lexer.peek().kind != TokenKind.Semicolon:
            load_result(expression(lexer, frame))
        lexer.next(TokenKind.Semicolon)
        emit_return(frame)
    elif lexer.try_next(TokenKind.If):
        with emit_block("block ;; if statement", "end"):
            with emit_block("block", "end"):
                parenthesized_test()
                emit("br_if 0 ;; jump to else")
                bracketed_block_or_single_statement(lexer, frame)
                emit("br 1 ;; exit if")  # skip to end of else block
            if lexer.try_next(TokenKind.Else):
                # single statement might be "if" of "else if"
                bracketed_block_or_single_statement(lexer, frame)
    elif lexer.try_next(TokenKind.While):
        with emit_block("block ;; while", "end"):
            with emit_block("loop", "end"):
                parenthesized_test()
                emit("br_if 1 ;; exit loop")
                bracketed_block_or_single_statement(lexer, frame)
                emit("br 0 ;; repeat loop")
    elif lexer.try_next(TokenKind.Do):
        with emit_block("block ;; do-while", "end"):
            with emit_block("loop", "end"):
                bracketed_block_or_single_statement(lexer, frame)
                lexer.next(TokenKind.While)
                parenthesized_test()
                emit("br_if 1 ;; exit loop")
                emit("br 0 ;; repeat loop")
                lexer.next(TokenKind.Semicolon)
    elif lexer.try_next(TokenKind.For):
        lexer.next(TokenKind.OpenParen)
        with emit_block("block ;; for", "end"):
            if lexer.peek().kind != TokenKind.Semicolon:
                expression(lexer, frame)
                emit("drop ;; discard for initializer")
            lexer.next(TokenKind.Semicolon)
            with emit_block("loop", "end"):
                if lexer.peek().kind != TokenKind.Semicolon:
                    load_result(expression(lexer, frame))
                    emit("i32.eqz ;; for test")
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
        emit("drop ;; discard statement expr result")


def variable_declaration(lexer: Lexer, frame: StackFrame) -> None:
    type, varname = parse_type_and_name(lexer)
    frame.add_var(varname.content, type)

    while lexer.try_next(TokenKind.Comma):
        type, varname = parse_type_and_name(lexer, type=type.typename)
        frame.add_var(varname.content, type)

    lexer.next(TokenKind.Semicolon)


def decl(global_frame: StackFrame, lexer: Lexer) -> None:
    if lexer.try_next(TokenKind.Typedef):
        type = lexer.next(TokenKind.Type)
        name = lexer.next(TokenKind.Name)
        lexer.types[name.content] = type.content
        lexer.next(TokenKind.Semicolon)
        return

    decl_type, name = parse_type_and_name(lexer)

    if lexer.try_next(TokenKind.Semicolon):
        global_frame.add_var(name.content, decl_type, False)
    else:
        if decl_type.is_arr():
            die("no function array return, nice try")

        frame = StackFrame(global_frame)
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
                    emit(f"(param ${v.name} {v.type.wasmtype})")
            emit(f"(result {decl_type.wasmtype})")
            emit(";; fn prelude")
            emit("global.get $__stack_pointer")
            emit(f"i32.const {frame.frame_size}")
            emit("i32.sub")
            emit("global.set $__stack_pointer")
            for v in reversed(frame.variables.values()):
                if v.is_parameter:
                    emit("global.get $__stack_pointer")
                    emit(f"i32.const {frame.get_var_and_offset(v.name)[1]}")
                    emit("i32.add")
                    emit(f"local.get ${v.name}")
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
        emit("  (local.get 0) (local.get 0))")
        emit("(func $__swap_i32 (param i32) (param i32) (result i32 i32)")
        emit("  (local.get 1) (local.get 0))")

        global_frame = StackFrame()
        lexer = Lexer(src)
        while lexer.peek().kind != TokenKind.Eof:
            decl(global_frame, lexer)

        emit('(export "main" (func $main))')


if __name__ == "__main__":
    import fileinput

    with fileinput.input(encoding="utf-8") as fi:
        compile("".join(fi))  # todo: make this line-at-a-time?

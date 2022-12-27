import dataclasses
import enum
import re
import sys
from typing import NoReturn


def die(message: str, line: int | None = None) -> NoReturn:
    location = f" on line {line + 1}" if line is not None else ""
    print(f"error{location}: {message}", file=sys.stderr)
    sys.exit(1)


class TokenKind(enum.Enum):
    Invalid = "Invalid"
    Eof = "Eof"
    Type = "Type"
    Name = "Name"
    IntConst = "IntConst"
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
    Ampersand = "&"


@dataclasses.dataclass
class Token:
    kind: TokenKind
    content: str
    line: int


class Lexer:
    def __init__(self, src: str) -> None:
        self.src = src
        self.loc = 0
        self.line = 0
        self.types = ["int"]  # TODO: lexer hack

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
            die(f"expected {kind.value}", self.line)
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


def parse_type(lexer: Lexer) -> CType:
    name = lexer.next(TokenKind.Type)
    pointer_level = 0
    while lexer.try_next(TokenKind.Star):
        pointer_level += 1
    return CType(name, pointer_level)


def ctype_to_wasmtype(c_type: CType) -> str:
    if c_type.pointer_level > 0 or c_type.typename == "int":
        return "i32"
    else:
        die(f"unknown type: {c_type.typename}", c_type.token.line)


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
        self.frame_size += 1

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
    print(f"    ;; return--adjust stack pointer")
    print(f"    global.get $__stack_pointer")
    print(f"    i32.const {frame.frame_size}")
    print(f"    i32.add")
    print(f"    global.set $__stack_pointer")
    print(f"    return")


def emit_load(frame: StackFrame, name: Token) -> None:
    print(f"    ;; load {name.content}")
    print(f"    global.get $__stack_pointer")
    print(f"    i32.const {frame.get_offset(name)}")
    print(f"    i32.sub")
    print(f"    i32.load")


def expression(lexer: Lexer, frame: StackFrame) -> None:
    def value() -> None:
        if const := lexer.try_next(TokenKind.IntConst):
            print(f"    i32.const {const.content}")
        elif varname := lexer.try_next(TokenKind.Name):
            emit_load(frame, varname)
        elif lexer.try_next(TokenKind.Minus):
            if lexer.peek().kind == TokenKind.Minus:
                die("predecrement not supported", lexer.line)
            value()
            print(f"    i32.neg")
        elif lexer.try_next(TokenKind.Star):
            expression(lexer, frame)
            print(f"    i32.load")
        elif lexer.try_next(TokenKind.Ampersand):
            name = lexer.next(TokenKind.Name)
            print(f"    ;; &{name.content}")
            print(f"    global.get $__stack_pointer")
            print(f"    i32.const {frame.get_offset(name)}")
            print(f"    i32.sub")
        elif lexer.try_next(TokenKind.OpenParen):
            expression(lexer, frame)
            lexer.next(TokenKind.CloseParen)
        else:
            die("expected value", lexer.line)

    def muldiv() -> None:
        value()
        if lexer.peek().kind in (TokenKind.Star, TokenKind.Slash):
            op = lexer.next()
            value()
            if op.kind == TokenKind.Star:
                print("    i32.mul")
            else:
                print("    i32.div")

    muldiv()
    if lexer.peek().kind in (TokenKind.Plus, TokenKind.Minus):
        op = lexer.next()
        muldiv()
        if op.kind == TokenKind.Plus:
            print("    i32.add")
        else:
            print("    i32.sub")


def statement(lexer: Lexer, frame: StackFrame) -> None:
    if lexer.try_next(TokenKind.Return):
        if lexer.peek().kind != TokenKind.Semicolon:
            expression(lexer, frame)
        lexer.next(TokenKind.Semicolon)
        emit_return(frame)
    elif lexer.peek().kind in (TokenKind.Name, TokenKind.Star):
        # parse lhs
        pointer_level = 0
        while lexer.try_next(TokenKind.Star):
            pointer_level += 1
        name = lexer.next(TokenKind.Name)
        _ = frame.get_offset(name)  # assert var exists in case of `x;` with no store
        print(f"    ;; lhs {'*' * pointer_level}{name.content}")
        print(f"    global.get $__stack_pointer")
        print(f"    i32.const {frame.get_offset(name)}")
        print(f"    i32.sub")
        for _ in range(pointer_level):
            print(f"    i32.load")
        if lexer.try_next(TokenKind.Equals):
            print(f"    ;; rhs {'*' * pointer_level}{name.content}")
            expression(lexer, frame)
            print(f"    ;; store {'*' * pointer_level}{name.content}")
            print(f"    i32.store")
        else:
            print(f"    drop") # dead load ¯\_(ツ)_/¯
        lexer.next(TokenKind.Semicolon)
    else:
        die("expected statement", lexer.line)


def variable_declaration(lexer: Lexer, frame: StackFrame) -> None:
    type = parse_type(lexer)
    varname = lexer.next(TokenKind.Name)
    lexer.next(TokenKind.Semicolon)
    frame.add_var(varname.content, type)


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

    print(f"  (func ${function_name.content} (result {ctype_to_wasmtype(return_type)})")
    print(f"    ;; prelude--adjust stack pointer (grows down)")
    print(f"    global.get $__stack_pointer")
    print(f"    i32.const {frame.frame_size}")
    print(f"    i32.sub")
    print(f"    global.set $__stack_pointer")
    print(f"    ;; end prelude")

    while lexer.peek().kind != TokenKind.CloseCurly:
        statement(lexer, frame)
    lexer.next(TokenKind.CloseCurly)

    # TODO: for void functions we need to add an addl emit_return for implicit returns
    print(f"  )")


def compile(src: str) -> None:
    # prelude
    print("(module")
    print("  (memory 2)")
    print("  (global $__stack_pointer (mut i32) (i32.const 66560))")

    lexer = Lexer(src)

    while lexer.peek().kind != TokenKind.Eof:
        func_decl(lexer)

    print('  (export "main" (func $main))')

    # postlude
    print(")")


if __name__ == "__main__":
    import fileinput

    with fileinput.input(encoding="utf-8") as fi:
        compile("".join(fi))  # todo: make this line-at-a-time?

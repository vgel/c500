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
                kind=TokenKind.Name,
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


def map_type(c_type: str) -> str:
    if c_type == "int":
        return "i32"
    else:
        die(f"unknown type: {c_type}")


def expression_value(lexer: Lexer) -> None:
    if const := lexer.try_next(TokenKind.IntConst):
        print(f"    i32.const {const.content}")
    elif varname := lexer.try_next(TokenKind.Name):
        print(f"    local.get ${varname.content}")
    elif lexer.try_next(TokenKind.Minus):
        # todo: support --?
        expression_value(lexer)
        print(f"    i32.neg")
    elif lexer.try_next(TokenKind.OpenParen):
        expression(lexer)
        lexer.next(TokenKind.CloseParen)
    else:
        die("expected value", lexer.line)


def expression_muldiv(lexer: Lexer) -> None:
    expression_value(lexer)
    if lexer.peek().kind in (TokenKind.Star, TokenKind.Slash):
        op = lexer.next()
        expression_value(lexer)
        if op.kind == TokenKind.Star:
            print("    i32.mul")
        else:
            print("    i32.div")


def expression_plusminus(lexer: Lexer) -> None:
    expression_muldiv(lexer)
    if lexer.peek().kind in (TokenKind.Plus, TokenKind.Minus):
        op = lexer.next()
        expression_muldiv(lexer)
        if op.kind == TokenKind.Plus:
            print("    i32.add")
        else:
            print("    i32.sub")


def expression(lexer: Lexer) -> None:
    expression_plusminus(lexer)


def statement(lexer: Lexer) -> None:
    if lexer.try_next(TokenKind.Return):
        if lexer.peek().kind != TokenKind.Semicolon:
            expression(lexer)
        lexer.next(TokenKind.Semicolon)
        print("    return")
    elif name := lexer.try_next(TokenKind.Name):
        if lexer.peek().kind == TokenKind.Name:
            # `type name = ...` declaration
            typename = name
            name = lexer.next()
            print(f"    (local ${name.content} {map_type(typename.content)})")

        if lexer.try_next(TokenKind.Equals):
            expression(lexer)
            print(f"    local.set ${name.content}")

        lexer.next(TokenKind.Semicolon)


def decl(lexer: Lexer) -> None:
    if lexer.peek().kind == TokenKind.Name:
        func_decl(lexer)
    else:
        die("expected declaration", lexer.line)


def func_decl(lexer: Lexer) -> None:
    return_type = lexer.next(TokenKind.Name).content
    function_name = lexer.next(TokenKind.Name).content
    lexer.next(TokenKind.OpenParen)
    lexer.next(TokenKind.CloseParen)
    print(f"  (func ${function_name} (result {map_type(return_type)})")

    lexer.next(TokenKind.OpenCurly)
    while lexer.peek().kind != TokenKind.CloseCurly:
        statement(lexer)
    lexer.next(TokenKind.CloseCurly)

    print("  )")


def compile(src: str) -> None:
    # prelude
    print("(module")
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

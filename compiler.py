import contextlib
import dataclasses
import re
import sys
import traceback
from typing import Callable, NoReturn


def die(message: str, line: int | None = None) -> NoReturn:
    """Print the message with a traceback, along with a line number if supplied, and exit."""
    print("\n" + "-" * 30 + "\n", file=sys.stderr)
    traceback.print_stack()
    print("\n" + "-" * 30 + "\n", file=sys.stderr)
    location = f" on line {line + 1}" if line is not None else ""
    print(f"error{location}: {message}", file=sys.stderr)
    sys.exit(1)


_emit_disabled = False
_emit_indent = 0


def emit(code: str) -> None:
    """Emit the given webassembly at the current indent level, unless emit is disabled."""
    if not _emit_disabled:
        print(" " * _emit_indent + code)


@contextlib.contextmanager
def emit_block(start: str, end: str):
    """A content manager that emits `start`, then runs the contained code, then emits `end`."""
    global _emit_indent
    emit(start)
    _emit_indent += 2
    yield
    _emit_indent -= 2
    emit(end)


@contextlib.contextmanager
def no_emit():
    """A context manager that disables `emit()` for the contained code by setting a global flag."""
    global _emit_disabled
    try:
        _emit_disabled = True
        yield
    finally:
        _emit_disabled = False


LITERAL_TOKENS = "typedef if else while do for return ++ -- << >> && || == <= >= != < > ( ) { } [ ] ; = + - * / % & | ^ , ! ~".split()
TOK_INVALID = "Invalid"
TOK_EOF = "Eof"
TOK_TYPE = "Type"
TOK_NAME = "Name"
TOK_INTCONST = "IntConst"


@dataclasses.dataclass
class Token:
    kind: str
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

    def peek(self) -> Token:
        """Peek at the next token without consuming it. Consumes whitespace."""

        # skip past whitespace
        while self.loc < len(self.src) and self.src[self.loc] in " \t\n":
            if self.src[self.loc] == "\n":
                self.line += 1
            self.loc += 1

        if self.loc >= len(self.src):
            return Token(kind=TOK_EOF, content="", line=self.line)

        # identifiers and identifier-like tokens
        # we check identifiers before literal tokens so that "return0" isn't lexed as
        # "return", "0", but this means that we need to explicitly check for
        # identifier-like tokens so "return" isn't lexed as a Name just because it's `[a-z]+`
        m = re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*", self.src[self.loc :])
        if m is not None:
            tok = m.group(0)

            if tok in LITERAL_TOKENS:
                return Token(kind=tok, content=tok, line=self.line)

            kind = TOK_TYPE if tok in self.types else TOK_NAME  # lexer hack
            return Token(kind=kind, content=tok, line=self.line)

        # int constants
        m = re.match(r"^[0-9]+", self.src[self.loc :])
        if m is not None:
            return Token(kind=TOK_INTCONST, content=m.group(0), line=self.line)

        # other tokens not caught by the identifier-like-token check above
        for token_kind in LITERAL_TOKENS:
            if self.src[self.loc :].startswith(token_kind):
                return Token(kind=token_kind, content=token_kind, line=self.line)

        content = self.src[self.loc : self.loc + 10]  # arbitrary amount of context
        return Token(kind=TOK_INVALID, content=content, line=self.line)

    def next(self, kind: str | None = None) -> Token:
        """Consume the next token. If `kind` is specified, die if the token doesn't match."""
        token = self.peek()

        if kind is not None and token.kind != kind:
            die(f"expected {kind}, got {token.content!r}", self.line)

        if token.kind != TOK_INVALID:
            self.loc += len(token.content)

        return token

    def try_next(self, kind: str) -> Token | None:
        """If a token of the given kind is present, consume and return it. Otherwise do nothing."""
        if self.peek().kind != kind:
            return None
        return self.next()


class CType:
    """
    Represents a C type.
    * `typename`: the underlying, typedef-resolved typename, like "int"
    * `pointer_level`: what level of pointer this type is. 0 is a regular value, 1 is Ty*, etc.
    * `array_size`: if None, this type isn't an array. otherwise, the value in Ty v[array_size]
    * `decl_line`: if provided, used as the line number when printing errors related to this type
    """

    def __init__(
        self,
        typename: str,
        pointer_level: int = 0,
        array_size: int | None = None,
        decl_line: int | None = None,
    ) -> None:
        self.typename = typename
        self.pointer_level = pointer_level
        self.array_size = array_size
        self.decl_line = decl_line
        is_pointy = pointer_level > 0 or array_size is not None
        if is_pointy or typename == "int":
            self.wasmtype = "i32"
        else:
            die(f"unknown type: {typename}", decl_line)

    def sizeof(self) -> int:
        """Size of this type, in bytes."""

        if self.wasmtype in ("i32", "f32"):
            return 4 * (self.array_size or 1)

        return 8 * (self.array_size or 1)

    def is_ptr(self) -> bool:
        """Whether this type is a pointer or not. Returns false for arrays of non-pointers like int _[5]."""
        return self.pointer_level > 0

    def less_ptr(self) -> "CType":
        """
        Makes a new type one level of pointer less than this type, e.g. int** -> int*.
        Errors if the type isn't a pointer.
        """
        assert self.is_ptr(), f"bug: not a pointer: {self}"
        return CType(self.typename, self.pointer_level - 1, self.array_size)

    def more_ptr(self) -> "CType":
        """Makes a new type one level of pointer higher than this type, e.g. int -> int*"""
        return CType(self.typename, self.pointer_level + 1, self.array_size)

    def is_arr(self) -> bool:
        """Whether this type is an array."""
        return self.array_size is not None

    def as_non_array(self) -> "CType":
        """Makes a new type that's the same as this type, except it isn't an array"""
        assert self.is_arr(), f"bug: not an array: {self}"
        return CType(self.typename, self.pointer_level, None)

    def __str__(self) -> str:
        s = self.typename + "*" * self.pointer_level
        if self.array_size is not None:
            s += f"[{self.array_size}]"
        return s


def parse_type_and_name(lexer: Lexer, type: str | None = None) -> tuple[CType, Token]:
    """
    Parse a type and variable name like `int** x[5]`. If `type` is provided,
    it will be used instead of trying to eat a new type token from the lexer,
    to support parsing a type in a comma-separated declaration like `int x, *y;`.
    """
    if type is None:
        # resolve the typedef, if applicable, which was previously stored in the lexer
        # (lexer hack!) otherwise will use the identity mapping (e.g. int: int) added
        # to the lexer on init.
        type = lexer.types[lexer.next(TOK_TYPE).content]

    pointer_level = 0
    while lexer.try_next("*"):
        pointer_level += 1

    varname = lexer.next(TOK_NAME)
    if lexer.try_next("["):
        array_size = int(lexer.next(TOK_INTCONST).content)
        lexer.next("]")
    else:
        array_size = None

    return CType(type, pointer_level, array_size, varname.line), varname


@dataclasses.dataclass
class FrameVar:
    """
    Variable in a StackFrame.
    * `name`: name of the variable
    * `type`: the variable's type
    * `local_offset`: how many bytes from the top of this frame does the value start
    * `is_parameter`: whether the value is a parameter (True) or a local var (False)
    """

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
    """
    Metadata returned after generating code for an expression.
    * `is_place`: whether the expression was a place or a bare value.
      places are represented on the stack as an address, but not all addresses are
      places--places are things that can be assigned to. for example x[5] is a place,
      but &x is not. values are things that can be operated on, e.g. (x + 1) is a value
      because (x + 1) = 2 is meaningless.
      use `load_result` to turn a place into a value.
    * `type`: the type of the expression
    """

    is_place: bool
    type: CType


def load_result(em: ExprMeta) -> ExprMeta:
    """Load a place `ExprMeta`, turning it into a value `ExprMeta` of the same type"""
    if em.is_place:
        emit("i32.load")
    return ExprMeta(False, em.type)


def expression(lexer: Lexer, frame: StackFrame) -> ExprMeta:
    def value() -> ExprMeta:
        if const := lexer.try_next(TOK_INTCONST):
            emit(f"i32.const {const.content}")
            return ExprMeta(False, CType("int"))
        elif lexer.try_next("("):
            meta = expression(lexer, frame)
            lexer.next(")")
            return meta
        else:
            varname = lexer.next(TOK_NAME)
            if lexer.try_next("("):
                if lexer.peek().kind != ")":
                    while True:
                        load_result(expression(lexer, frame))
                        if not lexer.try_next(","):
                            break
                lexer.next(")")
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
        if lexer.try_next("["):
            lhs_meta = load_result(lhs_meta)
            lhs_type = lhs_meta.type
            if not (lhs_type.is_arr() or lhs_type.is_ptr()):
                die(f"not an array or pointer: {lhs_meta.type}", lexer.line)
            el_type = (
                lhs_type.as_non_array() if lhs_type.is_arr() else lhs_type.less_ptr()
            )

            load_result(expression(lexer, frame))
            lexer.next("]")
            emit(f"i32.const {el_type.sizeof()}")
            emit("i32.mul")
            emit("i32.add")
            return ExprMeta(True, el_type)
        else:
            return lhs_meta

    def prefix() -> ExprMeta:
        if lexer.try_next("&"):
            meta = prefix()
            if not meta.is_place:
                die("cannot take reference to value", lexer.line)
            return ExprMeta(False, meta.type.more_ptr())
        elif lexer.try_next("*"):
            meta = load_result(prefix())
            if not meta.type.is_ptr():
                die("cannot dereference non-pointer", lexer.line)
            return ExprMeta(True, meta.type.less_ptr())
        elif lexer.try_next("-"):
            emit("i32.const 0")
            meta = load_result(prefix())
            emit("i32.sub")
            return meta
        elif lexer.try_next("+"):
            return load_result(prefix())
        elif lexer.try_next("!"):
            meta = load_result(prefix())
            emit("i32.eqz")
            return meta
        elif lexer.try_next("~"):
            meta = load_result(prefix())
            emit("i32.const 0xffffffff")
            emit("i32.xor")
            return meta
        else:
            return accessor()

    # function for generating simple operator precedence levels from declarative
    # dictionaries of { token: instruction_to_emit }
    def makeop(
        higher: Callable[[], ExprMeta], ops: dict[str, str]
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

    muldiv = makeop(prefix, {"*": "i32.mul", "/": "i32.div_s", "%": "i32.rem_s"})

    def plusminus() -> ExprMeta:
        lhs_meta = muldiv()
        if lexer.peek().kind in ("+", "-"):
            lhs_meta = load_result(lhs_meta)
            op_token = lexer.next()
            rhs_meta = load_result(plusminus())

            lhs_type, rhs_type, res_type = lhs_meta.type, rhs_meta.type, lhs_meta.type

            # handle pointer math: `((int*)4) - 1 == (int*)0` because int is 4 bytes
            # (this makes for (char* c = arr; c < arr+size; c++) work)
            if lhs_meta.type.pointer_level == rhs_meta.type.pointer_level:
                # will handle this later
                pass
            elif lhs_meta.type.is_ptr() and rhs_meta.type.is_ptr():
                die(f"cannot {op_token.content} {lhs_meta.type} and {rhs_meta.type}")
            elif lhs_meta.type.is_ptr() and not rhs_meta.type.is_ptr():
                # left hand side is pointer: multiply rhs by sizeof
                emit(f"i32.const {lhs_meta.type.less_ptr().sizeof()}")
                emit("i32.mul")
            elif not lhs_meta.type.is_ptr() and rhs_meta.type.is_ptr():
                # right hand side is pointer: juggle the stack to get rhs on top,
                # then multiply and juggle back
                res_type = rhs_meta.type
                emit("call $__swap_i32")
                emit(f"i32.const {rhs_meta.type.less_ptr().sizeof()}")
                emit("i32.mul")
                emit("call $__swap_i32")

            emit("i32.add" if op_token.kind == "+" else "i32.sub")
            if op_token.kind == "-" and lhs_type.is_ptr() and rhs_type.is_ptr():
                # handle pointer subtraction case we skipped before:
                # `((int*)8) - ((int*)4) == 1`, so we need to divide by sizeof
                # (we could use shl, but the webassembly engine will almost
                #  certainly do the strength reduction for us)
                emit(f"i32.const {rhs_meta.type.less_ptr().sizeof()}")
                emit(f"i32.div_s")

            return ExprMeta(False, res_type)
        return lhs_meta

    shlr = makeop(plusminus, {"<<": "i32.shl", ">>": "i32.shr_s"})
    cmplg = makeop(
        shlr,
        {
            "<": "i32.lt_s",
            ">": "i32.gt_s",
            "<=": "i32.le_s",
            ">=": "i32.ge_s",
        },
    )
    cmpe = makeop(cmplg, {"==": "i32.eq", "!=": "i32.ne"})
    bitand = makeop(cmpe, {"&": "i32.and"})
    bitor = makeop(bitand, {"|": "i32.or"})
    xor = makeop(bitor, {"^": "i32.xor"})

    def assign() -> ExprMeta:
        lhs_meta = xor()
        if lexer.try_next("="):
            if not lhs_meta.is_place:
                die("lhs of assignment cannot be value", lexer.line)
            emit("call $__dup_i32")  # save copy of addr for later
            rhs_meta = load_result(assign())
            emit("i32.store")
            # use the saved address to immediately reload the value
            # this is slower than saving the value we just wrote, but easier to codegen :-)
            # this is needed for expressions like x = (y = 1)
            emit("i32.load")
            return rhs_meta
        return lhs_meta

    return assign()


def statement(lexer: Lexer, frame: StackFrame) -> None:
    def parenthesized_test() -> None:
        """Helper to parse a parenthesized condition like `(x == 1)`"""
        lexer.next("(")
        load_result(expression(lexer, frame))
        lexer.next(")")
        emit("i32.eqz")

    def bracketed_block_or_single_statement(lexer: Lexer, frame: StackFrame) -> None:
        """Helper to parse the block of a control flow statement"""
        if lexer.try_next("{"):
            while lexer.try_next("}") is None:
                statement(lexer, frame)
        else:
            statement(lexer, frame)

    if lexer.try_next("return"):
        if lexer.peek().kind != ";":
            load_result(expression(lexer, frame))
        lexer.next(";")
        emit_return(frame)
    elif lexer.try_next("if"):
        # we emit two nested blocks:
        #
        # block
        #   block
        #     ;; if test
        #     br_if 0
        #     ;; if body
        #     br 1
        #   end
        #   ;; else body
        # end
        #
        # the first br_if 0 will conditionally jump to the end of the inner block
        # if the test results in `0`, running the `;; else body` code.
        # the second, unconditional `br 1` will jump to the end of the outer block,
        # skipping `;; else body` if `;; if body` already ran.
        with emit_block("block ;; if statement", "end"):
            with emit_block("block", "end"):
                parenthesized_test()
                emit("br_if 0 ;; jump to else")
                bracketed_block_or_single_statement(lexer, frame)
                emit("br 1 ;; exit if")  # skip to end of else block
            if lexer.try_next("else"):
                # single statement might be "if" of "else if", so we don't
                # need to support that explicitly (we implicitly treat
                # `if (...) { ... } else if (...) { ... }`
                #  as
                # `if (...) { ... } else { if (...) { ... } })
                bracketed_block_or_single_statement(lexer, frame)
    elif lexer.try_next("while"):
        # we again emit two nested blocks, but one is a loop:
        #
        # block
        #   loop
        #     ;; test
        #     br_if 1
        #     ;; loop body
        #     br 0
        #   end
        # end
        #
        # `while` statements don't have else blocks, so this isn't for the same reason
        # as `if`. instead, it's because of how loop blocks work. a branch (br or br_if)
        # in a loop block jumps to the *beginning* of the block, not the end, so to exit
        # early we need an outer, regular block that we can jump to the end of.
        # `br_if 1` jumps to the end of that outer block if the test fails,
        # skipping the loop body.
        # `br 0` jumps back to the beginning of the loop to re-run the test if the loop
        # body finishes.
        with emit_block("block ;; while", "end"):
            with emit_block("loop", "end"):
                parenthesized_test()
                emit("br_if 1 ;; exit loop")
                bracketed_block_or_single_statement(lexer, frame)
                emit("br 0 ;; repeat loop")
    elif lexer.try_next("do"):
        # `do` is very similar to `while`, but the test is at the end instead.
        with emit_block("block ;; do-while", "end"):
            with emit_block("loop", "end"):
                bracketed_block_or_single_statement(lexer, frame)
                lexer.next("while")
                parenthesized_test()
                emit("br_if 1 ;; exit loop")
                emit("br 0 ;; repeat loop")
                lexer.next(";")
    elif lexer.try_next("for"):
        # for is the most complicated control structure. it's also the hardest to
        # deal with in our setup, because the third "advancement" statement (e.g. i++ in
        # a typical range loop) needs to be generated *after* the body, even though it
        # comes before. We'll handle this by relexing it, which I'll show in a second.
        # just like the while and do loop, we'll have two levels of blocks:
        #
        # block
        #   ;; for initializer
        #   drop ;; discard the result of the initializer
        #   loop
        #     ;; for test
        #     br_if 1
        #     ;; (we lex over the advancement statement here, but don't emit any code)
        #     ;; for body
        #     ;; for advancement (re-lexed)
        #     br 0
        #   end
        # end
        #
        # Just like in the `while` or `do`, we emit the test, `br_if` to the end of the outer block
        # if the test fails, and otherwise `br 0` to the beginning of the inner loop.
        # The differences are:
        # 1. We code for the initializer and to discard its value at the beginning of the block.
        #    This code only runs once, it's just grouped into the for block for ease of reading the
        #    WebAssembly.
        # 2. We have the for advancement statement emitted *after* the body. How?
        #    Right before lexing it the first time, we save a copy of the current lexer,
        #    including its position.
        #    Then, we disable `emit()` with the `no_emit` context manager and call `expression`.
        #    This will skip over the expression without emitting any code.
        #    Next, we lex and emit code for the body as usual.
        #    Finally, before parsing the closing curly brace for the for loop, we use the saved
        #    lexer to go over the advancement statement *again*, but this time emitting code.
        #    This places the code for the advancement statement in the right place.
        #    Perfect! All it took was some minor crimes against the god of clean code :-) 
        lexer.next("(")
        with emit_block("block ;; for", "end"):
            if lexer.peek().kind != ";":
                expression(lexer, frame)
                emit("drop ;; discard for initializer")
            lexer.next(";")
            with emit_block("loop", "end"):
                if lexer.peek().kind != ";":
                    load_result(expression(lexer, frame))
                    emit("i32.eqz ;; for test")
                    emit("br_if 1 ;; exit loop")
                lexer.next(";")
                saved_lexer = None
                if lexer.peek().kind != ")":
                    # save lexer position to emit advance stmt later (nasty hack)
                    saved_lexer = lexer.clone()
                    with no_emit():
                        expression(lexer, frame)  # advance past expr
                lexer.next(")")
                emit(";; for body")
                bracketed_block_or_single_statement(lexer, frame)
                if saved_lexer != None:
                    emit(";; for advancement")
                    expression(saved_lexer, frame)  # use saved lexer
                emit("br 0 ;; repeat loop")
    elif lexer.try_next(";"):
        pass  # nothing to emit
    else:
        expression(lexer, frame)
        lexer.next(";")
        emit("drop ;; discard statement expr result")


def variable_declaration(lexer: Lexer, frame: StackFrame) -> None:
    # parse a variable declaration like `int x, *y[2], **z`.
    # the only thing each element in that list shares is the typename.
    # adds each parsed variable to the provided stack frame.

    type, varname = parse_type_and_name(lexer)
    frame.add_var(varname.content, type)

    while lexer.try_next(","):
        # used the saved type from before
        type, varname = parse_type_and_name(lexer, type=type.typename)
        frame.add_var(varname.content, type)

    lexer.next(";")


def decl(global_frame: StackFrame, lexer: Lexer) -> None:
    # parse a global declaration -- typedef, global variable, or function.

    if lexer.try_next("typedef"):
        type = lexer.next(TOK_TYPE)
        name = lexer.next(TOK_NAME)
        # lexer hack!
        lexer.types[name.content] = type.content
        lexer.next(";")
        return

    decl_type, name = parse_type_and_name(lexer)

    if lexer.try_next(";"):
        global_frame.add_var(name.content, decl_type, False)
    else:
        if decl_type.is_arr():
            die("no function array return, nice try")

        frame = StackFrame(global_frame)
        lexer.next("(")
        if lexer.peek().kind != ")":
            while True:
                type, varname = parse_type_and_name(lexer)
                frame.add_var(varname.content, type, is_parameter=True)
                if not lexer.try_next(","):
                    break
        lexer.next(")")

        lexer.next("{")
        # declarations (up top, c89 only yolo)
        while lexer.peek().kind == TOK_TYPE:
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

            while lexer.peek().kind != "}":
                statement(lexer, frame)
            lexer.next("}")

            # wasmer seems to not understand that
            # `(func $x (result i32) block i32.const 0 return end)` doesn't have an implicit
            # return, so this is only there to provide a dummy stack value for the validator
            emit("i32.const 0xdeadb33f ;; validator hack")
            # TODO: for void functions we need to add an addl emit_return for implicit returns


def compile(src: str) -> None:
    # compile an entire file

    with emit_block("(module", ")"):
        emit("(memory 2)")
        emit("(global $__stack_pointer (mut i32) (i32.const 66560))")
        emit("(func $__dup_i32 (param i32) (result i32 i32)")
        emit("  (local.get 0) (local.get 0))")
        emit("(func $__swap_i32 (param i32) (param i32) (result i32 i32)")
        emit("  (local.get 1) (local.get 0))")

        global_frame = StackFrame()
        lexer = Lexer(src)
        while lexer.peek().kind != TOK_EOF:
            decl(global_frame, lexer)

        emit('(export "main" (func $main))')


if __name__ == "__main__":
    import fileinput

    with fileinput.input(encoding="utf-8") as fi:
        compile("".join(fi))  # todo: make this line-at-a-time?

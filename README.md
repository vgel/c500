# c500

This is a single-pass C→WebAssembly compiler in 500 lines of (nicely formatted, non-comment, non-docstring) Python:

```bash
$ sloccount compiler.py | grep python:
python:         500 (100.00%)
```

It currently supports:
* Arithmetic
* Pointers
* `int`, `char`, and string constants (no floating point)
* `if`, `while`, `do while`, `for`
* Functions
* Typedefs
* (Single dimensional) arrays
* Some other stuff

Notably, it does *not* support preprocessor directives or structs.

## Running code
You can use the `./run-test` script to compile and run C code. Simply run `./run-test <c file>` after running `mkdir scratch`.
You'll need [bat](https://github.com/sharkdp/bat), [wabt](https://github.com/WebAssembly/wabt), and [wasmer](https://wasmer.io/) installed.

## Running tests
You will need to clone the c-testsuite into the repo: https://github.com/c-testsuite/c-testsuite
You'll also need all the tools listed in the Running Code section above.
Then you can `./run-all`. It currently passes 34/220 test cases.

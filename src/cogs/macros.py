import math
import re
from random import random, seed
from cmath import log
from functools import reduce
from typing import Optional, Callable
import json
import time
import base64
import zlib

from .. import constants, errors
from ..types import Bot, BuiltinMacro


class MacroCog:

    def __init__(self, bot: Bot):
        self.debug = []
        self.bot = bot
        self.variables = {}
        self.builtins: dict[str, BuiltinMacro] = {}
        self.found = 0

        def builtin(name: str):
            def wrapper(func: Callable):
                self.builtins[name] = BuiltinMacro(func.__doc__, func)
                return func

            return wrapper

        @builtin("to_float")
        def to_float(v):
            """Casts a value to a float."""
            if "j" in v:
                return complex(v)
            return float(v)

        @builtin("to_boolean")
        def to_boolean(v: str):
            """Casts a value to a boolean."""
            if v in ("true", "1", "True", "1.0", "1.0+0.0j"):
                return True
            elif v in ("false", "0", "False", "0.0", "0.0+0.0j"):
                return False
            else:
                raise AssertionError(f"could not convert string to boolean: '{v}'")

        @builtin("add")
        def add(*args: str):
            assert len(args) >= 2, "add macro must receive 2 or more arguments"
            return str(reduce(lambda x, y: x + to_float(y), args, 0))

        @builtin("is_number")
        def is_number(value: str):
            """Checks if a value is a number."""
            try:
                to_float(value)
                return "true"
            except (TypeError, ValueError):
                return "false"

        @builtin("pow")
        def pow_(a: str, b: str):
            """Raises a value to another value."""
            a, b = to_float(a), to_float(b)
            return str(a ** b)

        @builtin("log")
        def log_(x: str, base: str | None = None):
            """Takes the natural log of a value, or with an optional second argument, a specified base."""
            x = to_float(x)
            if base is None:
                return str(log(x))
            else:
                base = to_float(base)
                return str(log(x, base))

        @builtin("real")
        def real(value: str):
            """Gets the real component of a complex value."""
            value = to_float(value) + 0j
            return str(value.real)

        @builtin("imag")
        def imag(value: str):
            """Gets the imaginary component of a complex value."""
            value = to_float(value) + 0j
            return str(value.imag)

        @builtin("rand")
        def rand(seed_: str | None = None):
            """Gets a random value, optionally with a seed."""
            if seed_ is not None:
                seed_ = to_float(seed_)
                assert isinstance(seed_, float), "Seed cannot be complex"
                seed(seed_)
            return str(random())

        @builtin("subtract")
        def subtract(a: str, b: str):
            """Subtracts a value from another."""
            a, b = to_float(a), to_float(b)
            return str(a - b)

        @builtin("hash")
        def hash_(value: str):
            """Gets the hash of a value."""
            return str(hash(value))

        @builtin("replace")
        def replace(value: str, pattern: str, replacement: str):
            """Uses regex to replace a pattern in a string with another string."""
            print(value, pattern, replacement)
            return re.sub(pattern, replacement, value)
        
        @builtin("ureplace")
        def ureplace(value: str, pattern: str, replacement: str):
            """Uses regex to replace a pattern in a string with another string. This version unescapes the pattern sent in."""
            print(value, pattern, replacement)
            return re.sub(unescape(pattern), replacement, value)

        @builtin("multiply")
        def multiply(*args: str):
            assert len(args) >= 2, "multiply macro must receive 2 or more arguments"
            return str(reduce(lambda x, y: x * to_float(y), args, 1))

        @builtin("divide")
        def divide(a: str, b: str):
            """Divides a value by another value."""
            a, b = to_float(a), to_float(b)
            try:
                return str(a / b)
            except ZeroDivisionError:
                if type(a) is complex:
                    return "nan"
                elif a > 0:
                    return "inf"
                elif a < 0:
                    return "-inf"
                else:
                    return "nan"

        @builtin("mod")
        def mod(a: str, b: str):
            """Takes the modulus of a value."""
            a, b = to_float(a), to_float(b)
            try:
                return str(a % b)
            except ZeroDivisionError:
                if a > 0:
                    return "inf"
                elif a < 0:
                    return "-inf"
                else:
                    return "nan"

        @builtin("int")
        def int_(value: str, base: str = "10"):
            """Converts a value to an integer, optionally with a base."""
            try:
                return str(int(value, base=int(to_float(base))))
            except (ValueError, TypeError):
                return str(int(to_float(value)))

        @builtin("hex")
        def hex_(value: str):
            """Converts a value to hexadecimal."""
            return str(hex(int(to_float(value))))

        @builtin("oct")
        def oct_(value: str):
            """Converts a value to octal."""
            return str(oct(int(to_float(value))))

        @builtin("bin")
        def bin_(value: str):
            """Converts a value to binary."""
            return str(bin(int(to_float(value))))

        @builtin("chr")
        def chr_(value: str):
            """Gets a character from a unicode codepoint."""
            self.found += 1
            return str(chr(int(to_float(value))))

        @builtin("ord")
        def ord_(value: str):
            """Gets the unicode codepoint of a character."""
            return str(ord(value))

        @builtin("len")
        def len_(value: str):
            """Gets the length of a string."""
            return str(len(value))

        @builtin("split")
        def split(value: str, delim: str, index: str):
            """Splits a value by a delimiter, then returns an index into the list of splits."""
            index = int(to_float(index))
            return value.split(delim)[index]

        @builtin("if")
        def if_(*args: str):
            """Decides between arguments to take the form of with preceding conditions, """
            """with an ending argument that is taken if none else are."""

            assert len(args) >= 3, "must have at least three arguments"
            assert len(args) % 2 == 1, "must have at an odd number of arguments"
            conditions = args[::2]
            replacements = args[1::2]
            print(conditions, replacements)
            for (condition, replacement) in zip(conditions, replacements):
                if to_boolean(condition):
                    return replacement
            return conditions[-1]

        @builtin("equal")
        def equal(a: str, b: str):
            """Checks if two strings are equal."""
            return str(a == b).lower()

        @builtin("less")
        def less(a: str, b: str):
            """Checks if a value is less than another."""
            a, b = to_float(a), to_float(b)
            return str(a < b).lower()

        @builtin("not")
        def not_(value: str):
            """Logically negates a boolean."""
            return str(not to_boolean(value)).lower()

        @builtin("and")
        def and_(*args: str):
            assert len(args) >= 2, "and macro must receive 2 or more arguments"
            return str(reduce(lambda x, y: x and to_boolean(y), args, True)).lower()

        @builtin("or")
        def or_(*args: str):
            assert len(args) >= 2, "or macro must receive 2 or more arguments"
            return str(reduce(lambda x, y: x or to_boolean(y), args, False)).lower()
        
        @builtin("error")
        def error(_message: str):
            """Raises an error with a specified message."""
            raise errors.CustomMacroError(f"custom error: {_message}")

        @builtin("assert")
        def assert_(value: str, message: str):
            """If the first argument doesn't evaluate to true, errors with a specified message."""
            if not to_boolean(value):
                raise errors.CustomMacroError(f"assertion failed: {message}")
            return ""

        @builtin("slice")
        def slice_(string: str, start: str | None = None, end: str | None = None, step: str | None = None):
            """Slices a string."""
            start = int(to_float(start)) if start is not None and len(start) != 0 else None
            end = int(to_float(end)) if end is not None and len(end) != 0 else None
            step = int(to_float(step)) if step is not None and len(step) != 0 else None
            slicer = slice(start, end, step)
            return string[slicer]
        
        @builtin("find")
        def find(string: str, substring: str, start: str | None = None, end: str | None = None):
            """Returns the index of the second argument in the first, optionally between the third and fourth."""
            if start is not None:
                start = int(start)
            if end is not None:
                end = int(end)
            return str(string.index(substring, start, end))
        
        @builtin("count")
        def count(string: str, substring: str, start: str | None = None, end: str | None = None):
            """Returns the number of occurences of the second argument in the first, """
            """optionally between the third and fourth arguments."""
            if start is not None:
                start = int(start)
            if end is not None:
                end = int(end)
            return string.count(substring, start, end)
        
        @builtin("join")
        def join(joiner: str, *strings: str):
            """Joins all arguments with the first argument."""
            return joiner.join(strings)

        @builtin("store")
        def store(name: str, value: str):
            """Stores a value in a variable."""
            assert len(self.variables) < 16, "cannot have more than 16 variables at once"
            assert len(value) <= 256, "values must be at most 256 characters long"
            self.variables[name] = value
            return ""

        @builtin("get")
        def get(name: str, value: str):
            """Gets the value of a variable, or a default."""
            try:
                return self.variables[name]
            except KeyError:
                self.variables[name] = value
                return self.variables[name]

        @builtin("load")
        def load(name):
            """Gets the value of a variable, erroring if it doesn't exist."""
            return self.variables[name]

        @builtin("drop")
        def drop(name):
            """Deletes a variable."""
            del self.variables[name]
            return ""

        @builtin("is_stored")
        def is_stored(name):
            """Checks if a variable is stored."""
            return str(name in self.variables).lower()
        
        @builtin("variables")
        def varlist():
            """Returns all variables as a JSON object."""
            return json.dumps(self.variables, separators=(",", ":")).replace("[", "\\[").replace("]", "\\]")

        @builtin("repeat")
        def repeat(amount: str, string: str, joiner: str = ""):
            """Repeats the second argument N times, where N is the first argument, optionally joined by the third."""
            # Allow floats, rounding up, for historical reasons
            amount = max(math.ceil(float(amount)), 0)
            # Precalculate the length
            length = amount * len(string) + max(amount - 1, 0) * len(joiner)
            # Reject if too long
            assert length < 4096, "repeated string is too long (max is 4096 characters)"
            return joiner.join([string] * amount)

        @builtin("concat")
        def concat(*args):
            """Concatenates all arguments into one string."""
            return "".join(args)

        @builtin("unescape")
        def unescape(string: str):
            """Unescapes a string, replacing \\\\/ with /, \\\\[ with [, and \\\\] with ]."""
            self.found += 1
            return string.replace("\\/", "/").replace("\\[", "[").replace("\\]", "]")

        @builtin("json.get")
        def jsonget(data: str, key: str):
            """Gets a value from a JSON object."""
            data = data.replace("\\[", "[").replace("\\]", "]")
            assert len(data) <= 256, "json data must be at most 256 characters long"
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            if isinstance(data, list):
                key = int(key)
            return json.dumps(data[key]).replace("[", "\\[").replace("]", "\\]")

        @builtin("json.set")
        def jsonset(data: str, key: str, value: str):
            """Sets a value in a JSON object."""
            assert len(data) <= 256, "json data must be at most 256 characters long"
            assert len(value) <= 256, "json data must be at most 256 characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            value = value.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            value = json.loads(value)
            if isinstance(data, list):
                key = int(key)
            data[key] = value
            return json.dumps(data).replace("[", "\\[").replace("]", "\\]")

        @builtin("json.remove")
        def jsonremove(data: str, key: str):
            """Removes a value from a JSON object."""
            assert len(data) <= 256, "json data must be at most 256 characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            if isinstance(data, list):
                key = int(key)
            del data[key]
            return json.dumps(data).replace("[", "\\[").replace("]", "\\]")

        @builtin("json.len")
        def jsonlen(data: str):
            """Gets the length of a JSON object."""
            assert len(data) <= 256, "json data must be at most 256 characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            return len(data)

        @builtin("json.append")
        def jsonappend(data: str, value: str):
            """Appends a value to a JSON array."""
            assert len(data) <= 256, "json data must be at most 256 characters long"
            assert len(value) <= 256, "json data must be at most 256 characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            value = value.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, list), "json must be an array"
            value = json.loads(value)
            data.append(value)
            return json.dumps(data).replace("[", "\\[").replace("]", "\\]")

        @builtin("json.insert")
        def jsoninsert(data: str, index: str, value: str):
            """Inserts a value into a JSON array at an index."""
            assert len(data) <= 256, "json data must be at most 256 characters long"
            assert len(value) <= 256, "json data must be at most 256 characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            value = value.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, list), "json must be an array"
            value = json.loads(value)
            index = int(index)
            data.insert(index, value)
            return json.dumps(data).replace("[", "\\[").replace("]", "\\]")
        
        @builtin("json.keys")
        def jsonkeys(data: str):
            """Gets the keys of a JSON object as a JSON array."""
            assert len(data) <= 256, "json data must be at most 256 characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, dict), "json must be an object"
            return json.dumps(list(data.keys())).replace("[", "\\[").replace("]", "\\]")
        
        @builtin("unixtime")
        def unixtime():
            """Returns the current Unix timestamp, or the number of seconds since midnight on January 1st, 1970 in UTC."""
            return str(time.time())
        
        @builtin("try")
        def try_(code: str):
            """Runs some escaped MacroScript code. Returns two slash-seperated arguments: if the code errored, and the output/error message (depending on whether it errored.)"""
            self.found += 1
            try:
                result, _ = self.parse_macros(unescape(code), False, init = False)
            except errors.FailedBuiltinMacro as e:
                return f"false/{e.message}"
            except AssertionError as e:
                return f"false/{e}"
            return f"true/{result}"
        
        @builtin("lower")
        def lower(text: str):
            return text.lower()
        
        @builtin("upper")
        def upper(text: str):
            return text.upper()
        
        @builtin("title")
        def title(text: str):
            return text.title()

        @builtin("base64.encode")
        def base64encode(*args: str):
            assert len(args) >= 1, "base64.encode macro must receive 1 or more arguments"
            string = reduce(lambda x, y: str(x) + "/" + str(y), args)
            text_bytes = string.encode('utf-8')
            base64_bytes = base64.b64encode(text_bytes)
            return base64_bytes.decode('utf-8')
        
        @builtin("base64.decode")
        def base64decode(*args: str):
            assert len(args) >= 1, "base64.decode macro must receive 1 or more arguments"
            string = reduce(lambda x, y: str(x) + "/" + str(y), args)
            base64_bytes = string.encode('utf-8')
            text_bytes = base64.b64decode(base64_bytes)
            return text_bytes.decode('utf-8')

        @builtin("zlib.compress")
        def zlibcompress(*args: str):
            assert len(args) >= 1, "zlib.compress macro must receive 1 or more arguments"
            data = reduce(lambda x, y: str(x) + "/" + str(y), args)
            text_bytes = data.encode('utf-8')
            compressed_bytes = zlib.compress(text_bytes)
            base64_compressed = base64.b64encode(compressed_bytes)
            return base64_compressed.decode('utf-8')
        
        @builtin("zlib.decompress")
        def zlibdecompress(*args: str):
            assert len(args) >= 1, "zlib.decompress macro must receive 1 or more arguments"
            data = reduce(lambda x, y: str(x) + "/" + str(y), args)
            base64_compressed = data.encode('utf-8')
            compressed_bytes = base64.b64decode(base64_compressed)
            text_bytes = zlib.decompress(compressed_bytes)
            return text_bytes.decode('utf-8')

        self.builtins = dict(sorted(self.builtins.items(), key=lambda tup: tup[0]))

    def parse_macros(self, objects: str, debug_info: bool, macros=None, cmd="x", init=True) -> tuple[Optional[str], Optional[list[str]]]:
        if init:
            self.debug = []
            self.variables = {}
            self.found = 0
        if macros is None:
            macros = self.bot.macros

        # Find each outmost pair of brackets

        while match := re.search(r"(?<!(?<!\\)\\)\[((?:\\[\[\]])?(?:[^\[\]]|(?:[^\\]\\[\[\]]))*?(?<!(?<!\\)\\))]", objects, re.RegexFlag.M): # there's probably a much better way to do this regex but i haven't found it
            self.found += 1
            if debug_info:
                if self.found > constants.MACRO_LIMIT:
                    self.debug.append(f"[Error] Reached step limit of {constants.MACRO_LIMIT}.")
                    return None, self.debug
            else:
                assert self.found <= constants.MACRO_LIMIT, f"Too many macros in one render! The limit is {constants.MACRO_LIMIT}, while you reached {self.found}."
            terminal = match.group(1)
            if debug_info:
                self.debug.append(f"[Step {self.found}] {objects}")
            try:
                objects = (
                        objects[:match.start()] +
                        self.parse_term_macro(terminal, macros, self.found, cmd, debug_info) +
                        objects[match.end():]
                )
            except errors.FailedBuiltinMacro as err:
                if debug_info:
                    self.debug.append(f"[Error] Error in \"{err.raw}\": {err.message}")
                    return None, self.debug
                raise err
        if debug_info:
            self.debug.append(f"[Out] {objects}")
        return objects, self.debug if len(self.debug) else None

    def parse_term_macro(self, raw_variant, macros, step = 0, cmd = "x", debug_info = False) -> str:
        raw_macro, *macro_args = re.split(r"(?<!(?<!\\)\\)/", raw_variant)
        if raw_macro in self.builtins:
            try:
                macro = self.builtins[raw_macro].function(*macro_args)
                self.found -= 1
            except Exception as err:
                raise errors.FailedBuiltinMacro(raw_variant, err, isinstance(err, errors.CustomMacroError))
        elif raw_macro in macros:
            macro = macros[raw_macro].value
            macro = macro.replace("$#", str(len(macro_args)))
            macro = macro.replace("$!", cmd)
            macro_args = ["/".join(macro_args), *macro_args]
            arg_amount = 0
            iters = None
            while iters != 0 and arg_amount <= constants.MACRO_ARG_LIMIT:
                iters = 0
                matches = [*re.finditer(r"\$(-?\d+|#|!)", macro)]
                for match in reversed(matches):
                    iters += 1
                    arg_amount += 1
                    if arg_amount > constants.MACRO_ARG_LIMIT:
                        break
                    argument = match.group(1)
                    if argument == "#":
                        self.debug.append(f"[Step {step}:{arg_amount}:#] {len(macro_args) - 1} arguments")
                        infix = str(len(macro_args) - 1)
                    elif argument == "!":
                        infix = cmd
                    else:
                        argument = int(argument)
                        try:
                            infix = macro_args[argument]
                        except IndexError:
                            infix = "\0" + str(argument)
                    if debug_info:
                        self.debug.append(f"[Step {step}:{arg_amount}] {macro}")
                    macro = macro[:match.start()] + infix + macro[match.end():]
        else:
            raise AssertionError(f"Macro `{raw_macro}` of `{raw_variant}` not found in the database!")
        return str(macro).replace("\0", "$")


async def setup(bot: Bot):
    bot.macro_handler = MacroCog(bot)

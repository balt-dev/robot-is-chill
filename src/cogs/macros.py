import re
from random import random, seed
from cmath import log
from typing import Optional, Callable
import json

from discord.ext import commands

from .. import constants, errors
from ..types import Bot

class MacroCog:

    def __init__(self, bot: Bot):
        self.bot = bot
        self.variables = {}
        self.builtins = {}

        def builtin(name: str):
            def wrapper(func: Callable):
                self.builtins[name] = func
                return func

            return wrapper

        @builtin("to_float")
        def to_float(v):
            if "j" in v:
                return complex(v)
            return float(v)

        @builtin("to_boolean")
        def to_boolean(v: str):
            if v in ("true", "1", "True", "1.0", "1.0+0.0j"):
                return True
            elif v in ("false", "0", "False", "0.0", "0.0+0.0j"):
                return False
            else:
                raise AssertionError(f"could not convert string to boolean: '{v}'")

        @builtin("add")
        def add(a: str, b: str):
            a, b = to_float(a), to_float(b)
            return str(a + b)

        @builtin("is_number")
        def is_number(value: str):
            try:
                to_float(value)
                return "true"
            except (TypeError, ValueError):
                return "false"

        @builtin("pow")
        def pow(a: str, b: str):
            a, b = to_float(a), to_float(b)
            return str(a ** b)

        @builtin("log")
        def log(x: str, base: str | None = None):
            x = to_float(x)
            if base is None:
                return str(log(x))
            else:
                base = to_float(base)
                return str(log(x, base))

        @builtin("real")
        def real(value: str):
            value = to_float(value) + 0j
            return str(value.real)

        @builtin("imag")
        def imag(value: str):
            value = to_float(value) + 0j
            return str(value.imag)

        @builtin("rand")
        def rand(seed_: str | None = None):
            if seed_ is not None:
                seed_ = to_float(seed_)
                assert isinstance(seed_, float), "Seed cannot be complex"
                seed(seed_)
            return str(random())

        @builtin("subtract")
        def subtract(a: str, b: str):
            a, b = to_float(a), to_float(b)
            return str(a - b)

        @builtin("replace")
        def replace(value: str, pattern: str, replacement: str):
            print(value, pattern, replacement)
            return re.sub(pattern, replacement, value)

        @builtin("multiply")
        def multiply(a: str, b: str):
            a, b = to_float(a), to_float(b)
            return str(a * b)

        @builtin("divide")
        def divide(a: str, b: str):
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
            try:
                return str(int(value, base=int(to_float(base))))
            except (ValueError, TypeError):
                return str(int(to_float(value)))

        @builtin("hex")
        def hex_(value: str):
            return str(hex(int(to_float(value))))

        @builtin("chr")
        def chr_(value: str):
            return str(chr(int(to_float(value))))

        @builtin("ord")
        def ord_(value: str):
            return str(ord(value))

        @builtin("len")
        def len_(value: str):
            return str(len(value))

        @builtin("split")
        def split(value: str, delim: str, index: str):
            index = int(to_float(index))
            return value.split(delim)[index]

        @builtin("if")
        def if_(*args: str):
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
            return str(a == b).lower()

        @builtin("less")
        def less(a: str, b: str):
            a, b = to_float(a), to_float(b)
            return str(a < b).lower()

        @builtin("not")
        def not_(value: str):
            return str(not to_boolean(value)).lower()

        @builtin("and")
        def and_(a: str, b: str):
            return str(to_boolean(a) and to_boolean(b)).lower()

        @builtin("or")
        def or_(a: str, b: str):
            return str(to_boolean(a) or to_boolean(b)).lower()

        @builtin("error")
        def error(_message: str):
            raise AssertionError(f"custom error: {_message}")

        @builtin("assert")
        def assert_(value: str, message: str):
            assert to_boolean(value), message
            return ""

        @builtin("slice")
        def slice_(string: str, start: str | None = None, end: str | None = None, step: str | None = None):
            start = int(to_float(start)) if start is not None and len(start) != 0 else None
            end = int(to_float(end)) if end is not None and len(end) != 0 else None
            step = int(to_float(step)) if step is not None and len(step) != 0 else None
            slicer = slice(start, end, step)
            return string[slicer]

        @builtin("store")
        def store(name: str, value: str):
            assert len(self.variables) < 16, "cannot have more than 16 variables at once"
            assert len(value) <= 256, "values must be at most 256 characters long"
            self.variables[name] = value
            return ""

        @builtin("get")
        def get(name: str, value: str):
            try:
                return self.variables[name]
            except KeyError:
                self.variables[name] = value
                return self.variables[name]

        @builtin("load")
        def load(name):
            return self.variables[name]

        @builtin("drop")
        def drop(name):
            del self.variables[name]
            return ""

        @builtin("is_stored")
        def is_stored(name):
            return str(name in self.variables).lower()

        @builtin("concat")
        def concat(*args):
            return "".join(args)
        
        @builtin("unescape")
        def unescape(string: str):
            return string.replace("\\/", "/")
        
        @builtin("json.get")
        def jsonget(data: str, key: str):
            assert len(data) <= 256, "json data must be at most 256 characters long"
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            if isinstance(data, list):
                key = int(key)
            return json.dumps(data[key])
        
        @builtin("json.set")
        def jsonset(data: str, key: str, value: str):
            assert len(data) <= 256, "json data must be at most 256 characters long"
            assert len(value) <= 256, "json data must be at most 256 characters long"
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            value = json.loads(value)
            if isinstance(data, list):
                key = int(key)
            data[key] = value
            return json.dumps(data)
        
        @builtin("json.remove")
        def jsonremove(data: str, key: str):
            assert len(data) <= 256, "json data must be at most 256 characters long"
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            if isinstance(data, list):
                key = int(key)
            del data[key]
            return json.dumps(data)
        
        @builtin("json.len")
        def jsonlen(data: str):
            assert len(data) <= 256, "json data must be at most 256 characters long"
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            return len(data)
        
        @builtin("json.append")
        def jsonappend(data: str, value: str):
            assert len(data) <= 256, "json data must be at most 256 characters long"
            assert len(value) <= 256, "json data must be at most 256 characters long"
            data = json.loads(data)
            assert isinstance(data, list), "json must be an array"
            value = json.loads(value)
            data.append(value)
            return json.dumps(data)
        
        @builtin("json.insert")
        def jsoninsert(data: str, index: str, value: str):
            assert len(data) <= 256, "json data must be at most 256 characters long"
            assert len(value) <= 256, "json data must be at most 256 characters long"
            data = json.loads(data)
            assert isinstance(data, list), "json must be an array"
            value = json.loads(value)
            index = int(index)
            data.insert(index, value)
            return json.dumps(data)
        
    def parse_macros(self, objects: str, debug_info: bool, macros = None) -> tuple[Optional[str], Optional[list[str]]]:
        self.variables = {}
        if macros is None:
            macros = self.bot.macros

        debug = None
        if debug_info:
            debug = []

        # Find each outmost pair of brackets
        found = 0
        while match := re.search(r"(?!(?!\\)\\)\[([^\[]*?)]", objects, re.RegexFlag.M):
            found += 1
            if debug_info:
                if found > constants.MACRO_LIMIT:
                    debug.append(f"[Error] Reached step limit of {constants.MACRO_LIMIT}.")
                    return None, debug
            else:
                assert found <= constants.MACRO_LIMIT, f"Too many macros in one render! The limit is {constants.MACRO_LIMIT}, while you reached {found}."
            terminal = match.group(1)
            if debug_info:
                debug.append(f"[Step {found}] {objects}")
            try:
                objects = (
                        objects[:match.start()] +
                        self.parse_term_macro(terminal, macros) +
                        objects[match.end():]
                )
            except errors.FailedBuiltinMacro as err:
                if debug_info:
                    debug.append(f"[Error] Error in \"{err.raw}\": {err.message}")
                    return None, debug
                raise err
        if debug_info:
            debug.append(f"[Out] {objects}")
        return objects, debug

    def parse_term_macro(self, raw_variant, macros) -> str:
        raw_macro, *macro_args = re.split(r"(?<!(?<!\\)\\)/", raw_variant)
        if raw_macro in self.builtins:
            try:
                macro = self.builtins[raw_macro](*macro_args)
            except Exception as err:
                raise errors.FailedBuiltinMacro(raw_variant, err)
        elif raw_macro in macros:
            macro = macros[raw_macro].value
            macro = macro.replace("$#", str(len(macro_args)))
            macro_args = ["/".join(macro_args), *macro_args]
            arg_amount = 0
            while (match := re.search(r"\$(-?\d+|#)", macro)) is not None:
                arg_amount += 1
                if arg_amount > 100:
                    break
                argument = match.group(1)
                if argument == "#":
                    infix = str(len(macro_args))
                else:
                    argument = int(argument)
                    try:
                        infix = macro_args[argument]
                    except IndexError:
                        infix = "\0" + str(argument)
                macro = macro[:match.start()] + infix + macro[match.end():]
        else:
            raise AssertionError(f"Macro `{raw_macro}` of `{raw_variant}` not found in the database!")
        return str(macro).replace("\0", "$")



async def setup(bot: Bot):
    bot.macro_handler = MacroCog(bot)

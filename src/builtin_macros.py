# imma keep it a buck i do not care about code quality in this codebase anymore
# it's already shit so why care tbh

from random import random

variables = {}


def reset_vars():
    variables = {}


def to_float(v: str):
    if "j" in v:
        return complex(v)
    return float(v)


# bool("false") == True because it's not empty
def to_boolean(v: str):
    if v in ("true", "1", "True", "1.0", "1.0+0.0j"):
        return True
    elif v in ("false", "0", "False", "0.0", "0.0+0.0j"):
        return False
    else:
        raise AssertionError(f"could not convert string to boolean: '{v}'")


def add(a: str, b: str):
    a, b = to_float(a), to_float(b)
    return str(a + b)


def pow(a: str, b: str):
    a, b = to_float(a), to_float(b)
    return str(a ** b)


def real(value: str):
    value = to_float(value) + 0j
    return str(value.real)


def imag(value: str):
    value = to_float(value) + 0j
    return str(value.imag)


def rand():
    return str(random())


def subtract(a: str, b: str):
    a, b = to_float(a), to_float(b)
    return str(a - b)


def multiply(a: str, b: str):
    a, b = to_float(a), to_float(b)
    return str(a * b)


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


def modulo(a: str, b: str):
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


def int_(value: str, base: str = "10"):
    try:
        return str(int(value, base=int(to_float(base))))
    except (ValueError, TypeError):
        return str(int(to_float(value)))


def hex_(value: str):
    return str(hex(int(to_float(value))))


def chr_(value: str):
    return str(chr(int(to_float(value))))


def ord_(value: str):
    return str(ord(value))


def split(value: str, delim: str, index: str):
    index = int(to_float(index))
    return value.split(delim)[index]


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


def equal(a: str, b: str):
    return str(a == b).lower()


def less(a: str, b: str):
    a, b = to_float(a), to_float(b)
    return str(a < b).lower()


def not_(value: str):
    return str(not to_boolean(value)).lower()


def and_(a: str, b: str):
    return str(to_boolean(a) and to_boolean(b)).lower()


def or_(a: str, b: str):
    return str(to_boolean(a) or to_boolean(b)).lower()


def error(_message: str):
    raise AssertionError(f"custom error")


def slice_(string: str, start: str | None = None, end: str | None = None, step: str | None = None):
    start = int(to_float(start)) if start is not None and len(start) != 0 else None
    end = int(to_float(end)) if end is not None and len(end) != 0 else None
    step = int(to_float(step)) if step is not None and len(step) != 0 else None
    slicer = slice(start, end, step)
    return string[slicer]


def store(name: str, value: str):
    assert len(variables) <= 16, "cannot have more than 16 variables at once"
    assert len(value) <= 256, "values must be at most 256 characters long"
    variables[name] = value
    return ""


def load(name):
    return variables[name]


def drop(name):
    del variables[name]
    return ""


def concat(*args):
    return "".join(args)

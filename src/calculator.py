import ast
import operator
import re

BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node):
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return node.value
        raise ValueError("Invalid number")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in BINARY_OPS:
            raise ValueError("Unsupported operator")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero")
        result = BINARY_OPS[op_type](left, right)
        if abs(result) > 1e100:
            raise ValueError("Result too large")
        return result

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in UNARY_OPS:
            raise ValueError("Unsupported operator")
        return UNARY_OPS[op_type](_evaluate(node.operand))

    raise ValueError("Unsupported expression")


def _safe_eval(expr):
    result = _evaluate(ast.parse(expr, mode="eval").body)
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def calculate(question):
    q = question.lower().strip()

    replacements = [
        (r"\bmultiplied by\b|\btimes\b", "*"),
        (r"\bdivided by\b", "/"),
        (r"\bplus\b|\badded to\b", "+"),
        (r"\bminus\b", "-"),
        (r"\bequals to\b|\bequals\b", "="),
        (r"\bwhat is\b|\bcalculate\b|\bcompute\b", ""),
    ]
    for pattern, replacement in replacements:
        q = re.sub(pattern, replacement, q)

    q = re.sub(r"[?=.\s]+$", "", q).strip()

    if not q or not re.fullmatch(r"[0-9+\-*/().\s]+", q):
        return None

    try:
        return _safe_eval(q)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None


# alias for backwards compatibility
calculate_simple = calculate

#!/usr/bin/env python3
"""Fast math-answer verification via latex2sympy + sympy (lean path).

Faster than math-verify's full chain; good enough for correct/incorrect
splitting on DAPO-style answers. Returns True/False. Enforces a hard timeout
on the symbolic parse (latex2sympy can blow up on pathological input).
"""
import re
import signal


class _Timeout(Exception):
    pass


def _handler(signum, frame):
    raise _Timeout()


def _strip(s):
    s = s.strip()
    s = re.sub(r"\$", "", s)
    s = re.sub(r"\\,", "", s)
    return s


def _extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def _to_float(s):
    m = re.search(r"-?\d+(\.\d+)?", s)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def verify_fast(text, label, timeout=5.0):
    box = _extract_boxed(text)
    if box is None:
        return False
    a, b = _strip(box), _strip(str(label))
    if a == b:
        return True

    def run_parse():
        from latex2sympy_extended import parse_latex  # noqa
        import sympy as sp

        def norm(s):
            return s.replace("\\operatorname", "").strip()

        ea = parse_latex(norm(a))
        eb = parse_latex(norm(b))
        if ea is None or eb is None:
            return False
        if ea == eb:
            return True
        diff = sp.simplify(sp.expand(ea - eb))
        if diff == 0:
            return True
        na = float(sp.N(ea))
        nb = float(sp.N(eb))
        return abs(na - nb) < 1e-5 * max(1.0, abs(nb))

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return bool(run_parse())
    except _Timeout:
        pass
    except Exception:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    # fallback: numeric regex comparison for simple answers
    if "frac" not in a and "sqrt" not in a:
        na = _to_float(a)
        nb = _to_float(b)
        if na is not None and nb is not None:
            return abs(na - nb) < 1e-5 * max(1.0, abs(nb))
    return a.replace(" ", "") == b.replace(" ", "")


if __name__ == "__main__":
    import time
    pairs = [
        (r"The answer is \boxed{34}", "34", True),
        (r"\boxed{\frac{3}{4}}", r"\frac{3}{4}", True),
        (r"\boxed{5}", "6", False),
        (r"\boxed{\sqrt{2}}", r"\sqrt{2}", True),
        (r"no box here", "34", False),
    ]
    for text, lab, exp in pairs:
        t0 = time.time()
        got = verify_fast(text, lab)
        print(("OK " if got == exp else "BAD"), text[:40], "->", got,
              f"({time.time()-t0:.2f}s)")

#!/usr/bin/env python3
"""Math answer extraction & verification helpers (standalone).

Extracts the last \\boxed{...} and verifies against a label via math-verify
(with a hard timeout), falling back to numeric comparison.
"""
import re
import signal


class _Timeout(Exception):
    pass


def _handler(signum, frame):
    raise _Timeout()


def last_boxed_content(text):
    """Return (start, end) char range of the content inside the last \\boxed{},
    or None. start points at the first char of content."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    content_start = idx + len("\\boxed{")
    depth = 0
    for i in range(content_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return content_start, i
            depth -= 1
    return None


def _strip_latex(s):
    s = s.strip()
    s = re.sub(r"\$", "", s)
    s = re.sub(r"\\,", "", s)
    s = re.sub(r"\\ ", " ", s)
    return s


def _to_float(s):
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\frac", "FRAC").replace("\\sqrt", "SQRT")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def verify_answer(text, label, use_math_verify=True, timeout=8.0):
    m = last_boxed_content(text)
    if m is None:
        return False
    ans = text[m[0]:m[1]]
    ans_s = _strip_latex(ans)
    lab_s = _strip_latex(str(label))

    def run_mv():
        from math_verify import LatexExtractionConfig, parse, verify
        gold = parse(lab_s, extraction_config=[LatexExtractionConfig()])
        pred = parse(ans_s, extraction_config=[LatexExtractionConfig()])
        return bool(verify(gold, pred))

    if use_math_verify:
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return bool(run_mv())
        except _Timeout:
            pass
        except Exception:
            pass
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    # fallback: numeric equality via regex-extracted leading number
    if "FRAC" not in ans_s.replace("\\frac", "FRAC") and \
       "SQRT" not in ans_s.replace("\\sqrt", "SQRT"):
        na = _to_float(ans_s)
        nl = _to_float(lab_s)
        if na is not None and nl is not None:
            return abs(na - nl) < 1e-6 * max(1.0, abs(nl))
    return ans_s.replace(" ", "") == lab_s.replace(" ", "")

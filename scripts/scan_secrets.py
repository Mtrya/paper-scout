#!/usr/bin/env python3
"""Scan repo content for secrets, credentials, and personal/machine-identifying info.

Modes (exactly one):
  --staged      added lines in the staged diff (pre-commit)
  --range A..B  added lines introduced by commits A..B; A...B uses the merge base (pre-push / CI)
  --all         full content of every tracked file (audit; default)

Exit code is 1 when anything is found, 0 when clean. Findings are printed as
`path:line: [rule] text` with secret-category matches masked (first 4 chars +
length) so output is safe for CI logs; personal/machine matches are shown
verbatim because they are ordinary text the fixer must locate.

Allowlist: scripts/scan_secrets.allowlist holds one literal string per line
(`#` comments); a matching line is skipped. Binary files, .git,
workspace/papers/ (third-party paper cache), this scanner, and the allowlist
itself are never scanned.
"""

import argparse
import os
import re
import subprocess
import sys

REPO_ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True, text=True, check=True,
).stdout.strip()
SCANNER_PATH = os.path.relpath(os.path.abspath(__file__), REPO_ROOT)
ALLOWLIST_PATH = "scripts/scan_secrets.allowlist"
EXCLUDED_PREFIXES = ("workspace/papers/",)
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

PLACEHOLDER = re.compile(
    r"(?i)^(?:x+|\*+|<[^>]*>|\$\{.*\}|your[_-]?\w*|example|sample|placeholder"
    r"|dummy|redacted|none|null|changeme|todo|xxx)"
)
EMAIL_IGNORE = re.compile(r"(?i)(@example\.|@.*\.invalid|noreply|@localhost)")

RULES = [
    ("secret/private-key-block", "secret", re.compile(
        r"-----BEGIN (?:[A-Z0-9 ]* )?PRIVATE KEY(?: BLOCK)?-----")),
    ("secret/github-token", "secret", re.compile(
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{22,}")),
    ("secret/aws-access-key", "secret", re.compile(
        r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("secret/jwt", "secret", re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("secret/feishu-token", "secret", re.compile(
        r"\b[tu]-[A-Za-z0-9]{24,}\b")),
    ("secret/bearer-token", "secret", re.compile(
        r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}")),
    ("secret/credential-assignment", "secret", re.compile(
        r"(?i)\b(?:api[_-]?key|app[_-]?secret|client[_-]?secret|access[_-]?token"
        r"|auth[_-]?token|secret[_-]?key|secret|password|passwd)\b"
        r"\s*[:=：]\s*[\"']?([A-Za-z0-9_/+.\-]{12,})")),
    ("personal/email", "personal", re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("personal/cn-mobile", "personal", re.compile(
        r"(?<![\d.])1[3-9]\d{9}(?![\d.])")),
    ("personal/feishu-id", "personal", re.compile(
        r"\b(?:ou|on|oc)_[0-9a-zA-Z]{20,}\b")),
    ("machine/abs-home-path", "machine", re.compile(
        r"/home/(?!<)[a-z][a-z0-9_-]*")),
    ("machine/internal-ip", "machine", re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b")),
]


def load_allowlist():
    path = os.path.join(REPO_ROOT, ALLOWLIST_PATH)
    entries = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    entries.append(line)
    return entries


def is_binary(data):
    return b"\0" in data[:8192]


def excluded(path):
    return (
        path == SCANNER_PATH
        or path == ALLOWLIST_PATH
        or path.startswith(EXCLUDED_PREFIXES)
    )


def mask(text, match, category):
    if category != "secret":
        return text
    secret = match.group(1) if match.lastindex else match.group(0)
    masked = secret[:4] + "..." + f"({len(secret)} chars)"
    return text.replace(secret, masked)


def scan_line(path, lineno, line, allowlist, findings):
    if any(lit in line for lit in allowlist):
        return
    for name, category, pattern in RULES:
        for m in pattern.finditer(line):
            if name == "secret/credential-assignment" and PLACEHOLDER.match(m.group(1)):
                continue
            if name == "personal/email" and EMAIL_IGNORE.search(m.group(0)):
                continue
            text = mask(line.strip()[:200], m, category)
            findings.append((path, lineno, name, text))
            break  # one finding per rule per line is enough


def scan_working_tree(allowlist, findings):
    files = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT,
        capture_output=True, check=True,
    ).stdout.decode().split("\0")
    for path in files:
        if not path or excluded(path):
            continue
        full = os.path.join(REPO_ROOT, path)
        try:
            with open(full, "rb") as f:
                data = f.read()
        except OSError:
            continue
        if is_binary(data):
            continue
        for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            scan_line(path, lineno, line, allowlist, findings)


def scan_diff(diff_text, allowlist, findings):
    path = None
    new_lineno = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            path = target[2:] if target.startswith("b/") else None
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            new_lineno = int(m.group(1)) if m else None
        elif line.startswith("+") and not line.startswith("+++"):
            if path and new_lineno is not None and not excluded(path):
                scan_line(path, new_lineno, line[1:], allowlist, findings)
            if new_lineno is not None:
                new_lineno += 1
        elif not line.startswith("-"):
            if new_lineno is not None:
                new_lineno += 1


def resolve_range(spec):
    if "..." in spec:
        a, b = spec.split("...", 1)
        base = subprocess.run(
            ["git", "merge-base", a, b], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return base, b
    a, b = spec.split("..", 1)
    return a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true",
                       help="scan added lines in the staged diff")
    group.add_argument("--range", metavar="A..B",
                       help="scan added lines introduced by commits A..B")
    group.add_argument("--all", action="store_true",
                       help="scan full content of every tracked file (default)")
    args = parser.parse_args()

    allowlist = load_allowlist()
    findings = []

    if args.staged:
        diff = subprocess.run(
            ["git", "diff", "--cached", "--unified=0", "--diff-filter=ACMR"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        scan_diff(diff, allowlist, findings)
    elif args.range:
        a, b = resolve_range(args.range)
        diff = subprocess.run(
            ["git", "diff", "--unified=0", "--diff-filter=ACMR", a, b],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        scan_diff(diff, allowlist, findings)
    else:
        scan_working_tree(allowlist, findings)

    for path, lineno, name, text in findings:
        print(f"{path}:{lineno}: [{name}] {text}")
    if findings:
        print(f"\n{len(findings)} finding(s). Fix the content, or add a "
              f"commented literal to {ALLOWLIST_PATH} for verified false positives.")
        return 1
    print("clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fail on likely committed credentials while printing locations, never values."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".parquet", ".pyc", ".db"}
PATTERNS = {
    "Supabase management token": re.compile(r"\bsbp_[A-Fa-f0-9]{20,}\b"),
    "Supabase secret key": re.compile(r"\bsb_secret_[A-Za-z0-9_-]{20,}\b"),
    "Google API key": re.compile(r"\bAQ\.[A-Za-z0-9_-]{20,}\b"),
    "credentialed PostgreSQL URL": re.compile(
        r"postgresql(?:\+psycopg)?://[^\s:*]+:[^\s@*$<{][^\s@]*@"
    ),
    "authorization bearer token": re.compile(
        r"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._-]{16,}"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def repository_files() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(ROOT / line for line in result.stdout.splitlines() if line)


def scan() -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []
    for path in repository_files():
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, 1):
            if "secret-scan: allow-test" in line:
                continue
            relative = path.relative_to(ROOT).as_posix()
            safe_fixture = (
                relative.startswith("tests/") or relative == ".github/workflows/ci.yml"
            ) and ("localhost" in line or "db.example" in line)
            if safe_fixture:
                continue
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((relative, line_number, kind))
    return findings


def main() -> None:
    findings = scan()
    for path, line, kind in findings:
        print(f"{path}:{line}: {kind}")
    if findings:
        raise SystemExit(f"Secret scan failed with {len(findings)} finding(s).")
    print("Secret scan passed: no likely committed credentials found.")


if __name__ == "__main__":
    main()

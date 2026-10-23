"""Check a Decode & Detect submission before uploading it.

Usage:
  python3 check_submission.py csv submission.csv --test-file data/test/transactions.parquet [--expected-rows 40101]
  python3 check_submission.py zip code.zip

Exit code 0 means no problems were found, 1 means at least one problem.
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

CSV_LIMIT = 12 * 1024 * 1024
ZIP_LIMIT = 8 * 1024 * 1024


def report(problems, warnings):
    for p in problems:
        print(f"PROBLEM  {p}")
    for w in warnings:
        print(f"WARNING  {w}")
    print("RESULT   " + ("NOT READY: fix the problems above" if problems else "OK to upload"))
    return 1 if problems else 0


def check_csv(args):
    import pandas as pd

    path, problems, warnings = Path(args.path), [], []
    if path.stat().st_size > CSV_LIMIT:
        problems.append(f"file is {path.stat().st_size / 2**20:.1f} MiB; the limit is 12 MiB")
    sub = pd.read_csv(path, dtype={"TRANSACTION_HASH": str})
    if list(sub.columns) != ["TRANSACTION_HASH", "score"]:
        problems.append(f"columns must be exactly TRANSACTION_HASH,score (found {','.join(map(str, sub.columns))})")
        return report(problems, warnings)

    dupes = sub.TRANSACTION_HASH.duplicated().sum()
    if dupes:
        problems.append(f"{dupes:,} transactions appear more than once (one row per transaction)")
    if args.test_file:
        test_file = Path(args.test_file)
        reader = pd.read_parquet if test_file.suffix == ".parquet" else pd.read_csv
        expected = set(reader(test_file, columns=["TRANSACTION_HASH"]).TRANSACTION_HASH)
        given = set(sub.TRANSACTION_HASH)
        if expected - given:
            problems.append(f"{len(expected - given):,} test transactions are missing")
        if given - expected:
            problems.append(f"{len(given - expected):,} rows are not test transactions (scored transfers by mistake?)")
    else:
        warnings.append("no --test-file given, so missing or extra transactions were not checked")
    if args.expected_rows and len(sub) != args.expected_rows:
        problems.append(f"{len(sub):,} rows, expected {args.expected_rows:,}")

    scores = pd.to_numeric(sub.score, errors="coerce")
    if scores.isna().any():
        problems.append(f"{scores.isna().sum():,} scores are missing or not numbers")
    if (scores < 0).any() or (scores > 1).any():
        problems.append("scores must be between 0 and 1")
    distinct = scores.nunique()
    if distinct <= 2:
        problems.append("only 0/1 style scores: use continuous scores, since only the order counts")
    tied = scores.duplicated(keep=False).mean()
    if tied > 0.01:
        warnings.append(f"{tied:.1%} of rows share a score with another row; ties are ranked as one block")
    print(f"CHECKED  {len(sub):,} rows, {distinct:,} distinct scores, range {scores.min():.4f} to {scores.max():.4f}")
    return report(problems, warnings)


BANNED_PARTS = {"site-packages", ".venv", "venv", "env", "__pycache__", ".git", "node_modules",
                ".ipynb_checkpoints", ".DS_Store", "__MACOSX"}
SECRET_NAMES = re.compile(r"(^|/)(\.env(\..*)?|id_rsa|id_ed25519|.*\.pem|.*\.key|credentials.*)$", re.I)
SECRET_TEXT = re.compile(rb"(sk-ant-[A-Za-z0-9_-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{20,})")
DATA_SUFFIXES = {".parquet", ".feather", ".pkl", ".pickle", ".h5"}


def check_zip(args):
    path, problems, warnings = Path(args.path), [], []
    if path.stat().st_size > ZIP_LIMIT:
        problems.append(f"ZIP is {path.stat().st_size / 2**20:.1f} MiB; the limit is 8 MiB")
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        for n in names:
            parts = set(Path(n).parts)
            if parts & BANNED_PARTS:
                problems.append(f"remove {n} (installed packages, caches or git internals)")
            elif SECRET_NAMES.search(n):
                problems.append(f"remove {n}: it looks like a secrets or key file")
            elif Path(n).suffix.lower() in DATA_SUFFIXES:
                warnings.append(f"{n} looks like a data file; only include data the rules allow you to share")
            elif z.getinfo(n).file_size < 2**20 and SECRET_TEXT.search(z.read(n)):
                problems.append(f"{n} seems to contain an API key or private key")
    lowered = [n.lower() for n in names]
    checks = {
        "a README with the exact run command": any(Path(n).name.startswith("readme") for n in lowered),
        "a dependencies file (requirements.txt, pyproject.toml or environment.yml)":
            any(Path(n).name in {"requirements.txt", "pyproject.toml", "environment.yml"} for n in lowered),
        "a LICENSE file": any(Path(n).name.startswith(("license", "licence")) for n in lowered),
        "a method write-up (METHOD.md, PDF or slides)":
            any(Path(n).name.startswith("method") or n.endswith((".pdf", ".pptx")) for n in lowered),
    }
    for what, found in checks.items():
        if not found:
            warnings.append(f"missing {what}")
    print(f"CHECKED  {len(names):,} files, {path.stat().st_size / 2**20:.2f} MiB")
    return report(problems, warnings)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = parser.add_subparsers(dest="kind", required=True)
p_csv = sub.add_parser("csv", help="check the predictions CSV")
p_csv.add_argument("path")
p_csv.add_argument("--test-file", help="file listing every test TRANSACTION_HASH (parquet or CSV)")
p_csv.add_argument("--expected-rows", type=int)
p_zip = sub.add_parser("zip", help="check the code ZIP")
p_zip.add_argument("path")
args = parser.parse_args()
sys.exit(check_csv(args) if args.kind == "csv" else check_zip(args))

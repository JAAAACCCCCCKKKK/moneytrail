"""Keep OUTPUTS.md up to date: one row per predictions file we have produced.

The CSVs are committed in outputs/ (but left out of the release code ZIP, see
.gitattributes); this ledger says which file is which, whether it passed the
checker, and where it came from. Rows are keyed by the file's SHA-256, so a regenerated
file gets a new row and the old one stays as history. The Notes column is
hand-written and is preserved across updates.

The SHA-256 ignores line endings (CRLF is read as LF). With core.autocrlf, git stores
these CSVs with LF but checks them out with CRLF on Windows, so a raw hash of the same
file differed between team laptops. Rows keyed by an older raw hash are re-keyed
automatically the first time the tool sees the file.

Run from the repo root after writing any file into outputs/:
  python tools/track_outputs.py
  python tools/track_outputs.py --note predictions_shapes_v2.csv "Run 3 candidate ..."
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LEDGER = ROOT / "OUTPUTS.md"
CHECKER = ROOT / ".claude/skills/submission-check/scripts/check_submission.py"
TEST_HASHES = ROOT / "data/test-data/transactions.parquet"
EXPECTED_ROWS = 40101
COLUMNS = ["File", "Written (UTC)", "Rows", "Distinct scores", "SHA-256", "Check", "Status", "Notes"]

HEADER = """# Outputs ledger

Every predictions file we have produced, whether or not it was uploaded. The CSVs are
committed in `outputs/` but left out of the release code ZIP (see `.gitattributes`). Update it with
`python tools/track_outputs.py` after writing a file. Rows are keyed by SHA-256 with line
endings ignored, so the same file has the same key on Windows and macOS, and a
regenerated file gets a new row. "Check" is the `submission-check` result against all
40,101 test transactions. Leaderboard scores and the commit behind each upload live in
`SUBMISSIONS.md`.
"""


def raw_sha256(path: Path) -> str:
    """Hash of the exact bytes on disk (only used to re-key old rows)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256(path: Path) -> str:
    """Hash with CRLF read as LF, so it is the same on every OS and matches git's copy."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check(path: Path) -> str:
    """Run the team checker; return 'OK', 'OK, n warnings' or 'NOT READY: ...'."""
    cmd = [sys.executable, str(CHECKER), "csv", str(path)]
    if TEST_HASHES.exists():
        cmd += ["--test-file", str(TEST_HASHES), "--expected-rows", str(EXPECTED_ROWS)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    problems = [l[len("PROBLEM"):].strip() for l in out.splitlines() if l.startswith("PROBLEM")]
    warnings = [l for l in out.splitlines() if l.startswith("WARNING")]
    if problems:
        return "NOT READY: " + "; ".join(problems)
    return "OK" + (f", {len(warnings)} warning(s)" if warnings else "")


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| File") or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if len(cells) == len(COLUMNS):
            rows.append(dict(zip(COLUMNS, cells)))
    return rows


def write_ledger(rows: list[dict]) -> None:
    lines = [HEADER, "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in sorted(rows, key=lambda r: r["Written (UTC)"]):
        lines.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in COLUMNS) + " |")
    LEDGER.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", nargs=2, metavar=("FILE", "TEXT"), action="append", default=[],
                        help="set the Notes cell for the current version of FILE")
    args = parser.parse_args()

    rows = {r["SHA-256"]: r for r in read_ledger()}
    for r in rows.values():
        r["Status"] = "gone"          # re-marked below if the file still exists unchanged
    notes = {name: text for name, text in args.note}

    for path in sorted(OUTPUTS.glob("*.csv")):
        digest = sha256(path)[:12]
        raw = raw_sha256(path)[:12]
        if raw != digest and raw in rows:      # a row keyed by the old, line-ending-dependent hash
            old = rows.pop(raw)
            if digest in rows:                 # same file already listed: keep any note it had
                if not rows[digest]["Notes"] and old["Notes"]:
                    rows[digest]["Notes"] = old["Notes"]
                print(f"merged  {path.name}  {raw} -> {digest} (same file, different line endings)")
            else:
                old["SHA-256"] = digest
                rows[digest] = old
                print(f"re-key  {path.name}  {raw} -> {digest}")
        row = rows.get(digest)
        if row is None:
            scores = pd.read_csv(path, usecols=["score"]).score
            written = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            row = {"File": f"`{path.name}`", "Written (UTC)": written.strftime("%Y-%m-%d %H:%M"),
                   "Rows": f"{len(scores):,}", "Distinct scores": f"{scores.nunique():,}",
                   "SHA-256": digest, "Check": check(path), "Notes": ""}
            rows[digest] = row
            print(f"new     {path.name}  {digest}  {row['Check']}")
        row["Status"] = "present"
        if path.name in notes:
            row["Notes"] = notes.pop(path.name)
    for name in notes:
        print(f"warning: no file outputs/{name} to attach a note to")

    write_ledger(list(rows.values()))
    print(f"{LEDGER.name}: {len(rows)} row(s)")


if __name__ == "__main__":
    main()

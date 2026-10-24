"""The organisers' code-ZIP limits that the submission-check skill does not test.

From the data README: at most 8 MiB compressed (checked by the skill), 64 MiB
uncompressed and 2,000 files; encrypted, multi-part and ZIP64 archives are rejected.

Usage: python tools/check_zip_limits.py code.zip      (exit 1 on any problem)
"""
import sys
import zipfile
from pathlib import Path

MAX_FILES = 2000
MAX_UNCOMPRESSED = 64 * 1024 * 1024
ZIP64_LIMIT = 0xFFFFFFFF


def main(path: Path) -> int:
    problems = []
    with zipfile.ZipFile(path) as z:
        files = [i for i in z.infolist() if not i.is_dir()]
        total = sum(i.file_size for i in files)
        if len(files) > MAX_FILES:
            problems.append(f"{len(files):,} files; the limit is {MAX_FILES:,}")
        if total > MAX_UNCOMPRESSED:
            problems.append(f"{total / 2**20:.1f} MiB uncompressed; the limit is 64 MiB")
        encrypted = [i.filename for i in files if i.flag_bits & 0x1]
        if encrypted:
            problems.append(f"encrypted entries: {', '.join(encrypted[:5])}")
        if len(z.infolist()) >= 0xFFFF or any(
                max(i.file_size, i.compress_size, i.header_offset) >= ZIP64_LIMIT for i in z.infolist()):
            problems.append("archive needs ZIP64")
    for p in problems:
        print(f"PROBLEM  {p}")
    print(f"LIMITS   {len(files):,} files, {total / 2**20:.2f} MiB uncompressed: "
          + ("NOT OK" if problems else "within the organisers' limits"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))

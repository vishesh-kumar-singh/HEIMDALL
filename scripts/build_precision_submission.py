#!/usr/bin/env python3
"""Targeted builder for precision-gated-submission adhering strictly to scripts/build.py specification."""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build import archive, directory_entries, DIST

def main():
    sol_dir = ROOT / "solutions" / "precision-gated-submission"
    if not sol_dir.is_dir():
        raise SystemExit(f"Directory not found: {sol_dir}")
    
    entries = directory_entries(sol_dir)
    # 1. Build Small Model Track ZIP (with small.txt)
    entries_small = dict(entries)
    for k in list(entries_small.keys()):
        if "/" not in k and k.casefold() == "small.txt":
            del entries_small[k]
    entries_small["small.txt"] = b""
    out_zip_small = DIST / "solution-precision-gated-submission-small.zip"
    archive(out_zip_small, entries_small)
    print(f"Successfully built Small Model Track ZIP: {out_zip_small} ({out_zip_small.stat().st_size} bytes)")

    # 2. Build Main Track ZIP (without small.txt)
    entries_main = dict(entries)
    for k in list(entries_main.keys()):
        if "/" not in k and k.casefold() == "small.txt":
            del entries_main[k]
    out_zip_main = DIST / "solution-precision-gated-submission.zip"
    archive(out_zip_main, entries_main)
    print(f"Successfully built Main Track ZIP: {out_zip_main} ({out_zip_main.stat().st_size} bytes)")

if __name__ == "__main__":
    main()

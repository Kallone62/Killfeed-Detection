#!/usr/bin/env python3
"""Standalone CS2-video -> killfeed JSON/PNG CLI; algorithm stays SHA-locked."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Detect CS2 killfeed events in gameplay MP4 videos")
    p.add_argument("--input", type=Path, required=True, help="MP4 file or directory of MP4 files")
    p.add_argument("--output", type=Path, required=True, help="Directory for JSON/TSV/PNG results")
    p.add_argument("--owner", required=True, help="Player name shown in the killfeed (minimum 3 characters)")
    p.add_argument("--timeout", type=int, default=350, help="Per-video analysis timeout in seconds")
    args = p.parse_args(argv)
    if len(args.owner.strip()) < 3:
        p.error("--owner must contain at least 3 characters")
    if args.timeout < 1:
        p.error("--timeout must be positive")
    if not args.input.exists():
        p.error("--input does not exist")
    if args.input.is_file() and args.input.suffix.lower() != ".mp4":
        p.error("--input must be an MP4 or a directory containing MP4 files")
    cmd = [sys.executable, str(ROOT / "rev12" / "entrypoint.py"),
           "--input", str(args.input), "--output", str(args.output),
           "--owner", args.owner.strip(), "--timeout", str(args.timeout),
           "--strict-qa"]
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main())

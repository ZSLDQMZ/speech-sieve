"""Command-line entrypoint for speech-sieve: check and run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigError, load_preset, read_jsonl, write_jsonl_atomic
from .pipeline import audit_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speech-sieve", description="Modular speech dataset quality profiling CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Validate a YAML configuration file.")
    p_check.add_argument("--config", "--preset", dest="preset", required=True, help="Path to YAML configuration file.")

    p_run = sub.add_parser("run", help="Run batch speech audit on a JSONL manifest.")
    p_run.add_argument("--input", required=True, help="Input manifest JSONL path.")
    p_run.add_argument("--config", "--preset", dest="preset", required=True, help="Path to YAML configuration file.")
    p_run.add_argument("--output", required=True, help="Output report JSONL path.")
    p_run.add_argument("--audio-root", default=None, help="Base folder for relative audio_path entries (defaults to input parent).")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        try:
            cfg = load_preset(args.preset)
        except ConfigError as err:
            print(f"invalid config: {err}", file=sys.stderr)
            return 2
        print(f"OK config={cfg['preset_name']} mode={cfg['mode']} nfft={cfg['features']['nfft']} device={cfg['models']['device']}")
        return 0

    if args.command == "run":
        try:
            cfg = load_preset(args.preset)
        except ConfigError as err:
            print(f"invalid config: {err}", file=sys.stderr)
            return 2
        in_path = Path(args.input)
        rows = read_jsonl(in_path)
        audio_root = Path(args.audio_root) if args.audio_root else in_path.parent
        reports = audit_batch(rows, cfg, audio_root=audio_root)
        write_jsonl_atomic(Path(args.output), reports)
        routes: dict[str, int] = {}
        for r in reports:
            rt = r["route"]
            routes[rt] = routes.get(rt, 0) + 1
        print(f"Processed {len(reports)} records -> {args.output} | routes={routes}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

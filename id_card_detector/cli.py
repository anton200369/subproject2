from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import process_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect an ID card and extract its components with sub-pixel accuracy.")
    parser.add_argument("input", type=Path, help="Path to the input photograph containing the ID card.")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Directory to store detected crops and intermediate files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    process_image(args.input, args.output)
    print(f"Extraction complete. Artifacts stored in: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

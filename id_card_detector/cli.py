from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import process_image, process_webcam


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect an ID card and extract its components with sub-pixel accuracy.")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Choose between static photo or live webcam input.")

    photo = subparsers.add_parser("photo", help="Process a single photo containing the ID card.")
    photo.add_argument("input", type=Path, help="Path to the input photograph containing the ID card.")
    photo.add_argument("--output", type=Path, default=Path("artifacts"), help="Directory to store detected crops and intermediate files.")

    webcam = subparsers.add_parser("webcam", help="Capture from a webcam until the ID card is detected.")
    webcam.add_argument("--output", type=Path, default=Path("artifacts"), help="Directory to store detected crops and intermediate files.")
    webcam.add_argument("--device", type=int, default=0, help="Webcam index passed to OpenCV's VideoCapture (default: 0).")
    webcam.add_argument("--max-frames", type=int, default=900, help="Maximum frames to read before giving up (default: 900).")
    webcam.add_argument("--stride", type=int, default=2, help="Process every Nth frame to save CPU (default: 2).")
    webcam.add_argument("--warmup", type=int, default=5, help="Frames to skip so the camera can auto-expose (default: 5).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "photo":
        process_image(args.input, args.output)
    else:
        process_webcam(
            output_dir=args.output,
            device_index=args.device,
            max_frames=args.max_frames,
            frame_stride=args.stride,
            warmup_frames=args.warmup,
        )

    print(f"Extraction complete. Artifacts stored in: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

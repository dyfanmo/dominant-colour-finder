import argparse
import sys
from pathlib import Path

from PIL import Image

from dominant_colour.cli import add_colours_argument, add_verbose_argument, measured, report_error
from dominant_colour.image_editing import black_out, default_output_path
from dominant_colour.image_loading import load_image_as_pixels


def build_parser() -> argparse.ArgumentParser:
    """Set up the command line arguments the script accepts."""
    parser = argparse.ArgumentParser(description="Black out every pixel of a given colour.")
    parser.add_argument("image_path", help="Path to the input image file.")
    add_colours_argument(parser, "colours", "Colours to black out.")
    parser.add_argument(
        "--output",
        help="Where to save the edited image (default: alongside the original).",
    )
    add_verbose_argument(parser)

    return parser


def main(argv=None) -> int:
    """Run the script, returning the exit code."""
    args = build_parser().parse_args(argv)
    output_path = Path(args.output) if args.output else default_output_path(args.image_path)

    try:
        with measured("load", args.verbose):
            pixels = load_image_as_pixels(args.image_path)
    except (FileNotFoundError, ValueError) as exc:
        return report_error(exc)

    with measured("black out", args.verbose):
        edited, changed = black_out(pixels, args.colours)

    with measured("save", args.verbose):
        Image.fromarray(edited).save(output_path)

    print(f"Blacked out {', '.join(args.colours)} - {changed:,} pixels changed")
    print(f"Saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

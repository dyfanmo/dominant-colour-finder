import sys

from dominant_colour.cli import build_parser, print_results
from dominant_colour.colour_analysis import DominantColourFinder
from dominant_colour.image_loading import load_image_as_pixels


def main() -> int:
    """Run the script, returning the exit code."""
    args = build_parser().parse_args()

    try:
        pixels = load_image_as_pixels(args.image_path)
        results = DominantColourFinder(pixels).find(
            top_n=args.top_n,
            bucket_size=args.bucket_size,
            colours_to_ignore=args.ignore,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())

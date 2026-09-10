# Dominant Colour Finder

Finds the most common colour in an image, and tells you what a person
would call it.

```
$ uv run scripts/find_dominant_colour.py photo.jpg
Dominant colour: blue (30, 80, 200)

$ uv run scripts/find_dominant_colour.py photo.jpg --top-n 3 --ignore white
Top 3 colours:
  1. blue (30, 80, 200)
  2. green (80, 140, 20)
  3. brown (110, 80, 50)
```

## How it works

The pipeline runs in four steps:

1. **Quantise** - round each RGB channel down into buckets (default 10),
   so near-identical shades caused by compression or camera noise get
   counted as one colour rather than thousands of slightly different ones.
2. **Count** - `np.unique` over the quantised pixels gives every distinct
   colour and how often it appears, vectorised rather than looping.
3. **Remove ignored** - drop any colours belonging to a family you asked
   to exclude, e.g. `--ignore white black` for a product photo on a
   white background.
4. **Rank** - sort by frequency and return the top N.

Naming happens after counting, so only the few thousand distinct colours
in an image need looking up rather than every pixel.

### Naming colours

Colour names come from van de Weijer's Color Naming model, which learned
the boundaries between the 11 basic colour names from real photographs
found via image search, rather than from colour chips under lab
conditions.

RGB space is split into a 32x32x32 grid, and the model provides the
probability of each cell being called each of the 11 names. So naming a
colour is one array lookup - no distance calculations, no colour space
conversions, and no boundaries to hand-tune.

> J. van de Weijer, C. Schmid, J. Verbeek, D. Larlus. Learning Color
> Names for Real-World Applications. IEEE Transactions on Image
> Processing, 2009.

Earlier versions matched each colour to the nearest of 11 hand-placed
reference colours, first in HSV and then in LAB using CIEDE2000. Both
needed the reference values tuned by hand, and both were beaten by the
lookup table on accuracy and by a wide margin on speed.

Names are one of: black, blue, brown, grey, green, orange, pink, purple,
red, white, yellow. Anything else lands on whichever it's closest to, so
cyan comes back as blue and olive as yellow.

## Setup

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Usage

```bash
uv run scripts/find_dominant_colour.py path/to/image.jpg
uv run scripts/find_dominant_colour.py path/to/image.jpg --top-n 3
uv run scripts/find_dominant_colour.py path/to/image.jpg --ignore white black
uv run scripts/find_dominant_colour.py path/to/image.jpg --bucket-size 20
```

## Development

```bash
uv run ruff check .      # lint
uv run mypy src          # type check
uv run pytest -v         # tests
```

### Test images

Some tests run against generated image fixtures. Build them with:

```bash
uv run scripts/generate_test_images.py
```

This writes three folders under `tests/test_data/`:

- `colours/` - one example of each of the 11 names
- `near_colours/` - colours between the 11, like cyan and olive
- `mixed/` - two or three colours in known proportions, for checking
  the right one wins

The images aren't committed, so those tests skip until you generate them.
To skip them deliberately:

```bash
uv run pytest -m "not image_data"
```

## Layout

```
src/dominant_colour/
    colour_matching.py     RGB -> colour name, via the lookup table
    colour_analysis.py     quantise, count, filter, rank
    colour_constants.py    example colours for tests, bucket size
    image_loading.py       image file -> pixel array (the only file I/O)
    image_generation.py    building test images from solid colours
    data/w2c.txt           the trained lookup table
scripts/
    find_dominant_colour.py    the CLI
    generate_test_images.py    writes the test fixtures
tests/
```

Image loading is kept separate from the analysis so the pipeline can be
tested on small hand-written arrays with no fixture files and no disk
access.

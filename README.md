# Dominant Colour Finder

Finds the most common colour in an image and tells you what a person would
call it.

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

Four steps:

1. **Quantise** - round each RGB channel down into buckets (default 10), so
   near-identical shades from compression or camera noise count as one colour
   rather than thousands of slightly different ones.
2. **Count** - tally how many pixels share each colour.
3. **Remove ignored** - drop colours belonging to a family you asked to
   exclude, e.g. `--ignore white black` for a product shot on a white
   background.
4. **Rank** - sort by frequency and return the top N.

Naming happens after counting, so only the few thousand distinct colours in
an image need looking up rather than every pixel.

### Naming colours

Names come from van de Weijer's Color Naming model, which learned the
boundaries between the eleven basic colour names from real photographs found
via image search, rather than from colour chips under lab conditions.

RGB space is split into a 32x32x32 grid and the model gives the probability of
each cell being called each of the eleven names. Naming a colour is one array
lookup - no distance calculations, no colour space conversions, no boundaries
to hand-tune.

Names are one of: black, blue, brown, grey, green, orange, pink, purple, red,
white, yellow. Anything else lands on whichever it is closest to, so cyan
comes back as blue and olive as yellow.

### What was tried first

Earlier versions matched each colour to the nearest of eleven hand-placed
reference colours, first in HSV with hue on a circle, then in CIELAB using
CIEDE2000 perceptual distance. Both needed the reference values tuning by
hand, and tuning them against a small set of test colours overfitted badly -
100% on the colours they were tuned against, 45% on colours they had not
seen. The lookup table beat both on accuracy and was around 17x faster.

Counting originally used `np.unique(axis=0)`, which sorts every pixel. Tallying
instead is roughly 90x faster on a 12 megapixel image and does not hold a
second copy of the array in memory.

## Accuracy

Measured against 387 colour samples from the CVC fuzzy colour naming data set,
where ten subjects each spread ten points across the eleven names per sample:

```
matches the name people chose      249/387 (64.3%)
a name a fair few people chose     283/387 (73.1%)

on samples people agreed on        220/298 (73.8%)
on samples people disagreed on      29/ 89 (32.6%)
```

The split matters more than the headline. Most of the failure is on genuinely
ambiguous colours; on samples where people broadly agreed it gets 73.8%.

The clearest weakness is that white is over-predicted. Light greys with a
slight blue tint - `(215, 216, 226)`, which ten out of ten subjects called
grey - come back as white. That accounts for grey scoring 29.8%, the worst of
the eleven, and for 38 of the 138 total errors.

Run it yourself:

```bash
uv run scripts/benchmark_colour_naming.py
```

## Seeing the naming work

`black_out_colours.py` paints every pixel of a named colour black. It exists
to check the naming visually - a before and after says more about where the
model draws its boundaries than a percentage does.

```bash
uv run scripts/black_out_colours.py photo.jpg blue
```

The flight suits go cleanly, including the folds and shadows where the blue
darkens, while the patches and name tags survive. The metal cylinders lose
their reflections too, because they genuinely are blue-tinted - the model
works on pixels, not objects.

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

Reads any format Pillow can open - JPEG, PNG, BMP, GIF, WEBP, TIFF - in any
mode, including greyscale, palette and CMYK. Transparent areas are flattened
onto white, since dropping the alpha channel would keep whatever colour sits
invisibly underneath. Images past Pillow's decompression bomb limit, around 89
megapixels, are refused with a clear message rather than a traceback.

## Development

```bash
uv run ruff check .      # lint
uv run mypy src          # type check
uv run pytest -v         # tests
```

Image loading is kept separate from the analysis so the pipeline can be tested
on small hand-written arrays with no fixture files and no disk access. The
analysis tests stub out the colour naming, so a problem with the lookup table
shows up in the naming tests rather than as a confusing analysis failure.

## Layout

```
src/dominant_colour/
    colour_matching.py     RGB -> colour name, via the lookup table
    colour_analysis.py     quantise, count, filter, rank
    constants.py           bucket size, lookup table layout
    image_loading.py       image file -> pixel array (the only file I/O)
    data/w2c.txt           the trained lookup table
    data/benchmark_data.txt  human colour judgements, for the benchmark
scripts/
    find_dominant_colour.py     the CLI
    black_out_colours.py        visual check on the naming
    benchmark_colour_naming.py  accuracy against human judgements
tests/
```

## Credits

**Colour naming model** - the `w2c.txt` lookup table is from:

> J. van de Weijer, C. Schmid, J. Verbeek, D. Larlus. Learning Color Names
> for Real-World Applications. IEEE Transactions on Image Processing, 2009.

Obtained via [Flipajs/color_naming_Weijer](https://github.com/Flipajs/color_naming_Weijer).

**Benchmark data** - the 387 human-judged colour samples are from:

> R. Benavente, M. Vanrell, R. Baldrich. A Data Set for Fuzzy Colour Naming.
> Color Research and Application, 31(1): 48-56, 2006.

Available at [cat.uab.cat/Datasets/color_naming](http://www.cat.uab.cat/Datasets/color_naming/).

**Test image** - NASA / Kennedy Space Center, public domain.

**Built with** NumPy and Pillow.

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

## Setup

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Usage

```bash
uv run scripts/find_dominant_colour.py path/to/image.jpg
uv run scripts/find_dominant_colour.py image.jpg --top-n 3 --ignore white black
uv run scripts/find_dominant_colour.py image.jpg --bucket-size 20 --verbose
uv run scripts/black_out_colours.py image.jpg blue      # visual check on the naming
uv run scripts/benchmark_colour_naming.py               # accuracy against human judgements
```

Reads any format Pillow can open, in any mode. Transparency is flattened onto
white; images past Pillow's ~89 megapixel limit are refused with a clear
message.

## Seeing the naming work

`black_out_colours.py` paints every pixel of a named colour black - a before
and after shows where the model draws its boundaries better than a percentage
does.

```bash
uv run scripts/black_out_colours.py docs/astronauts.jpg blue
```

| Before | After `blue` blacked out |
| --- | --- |
| ![Astronauts in blue flight suits](docs/astronauts.jpg) | ![The same photo with blue pixels blacked out](docs/astronauts_blue_blacked_out.jpg) |

These are resized copies; the run shown was on the 45-megapixel NASA
original, which you can download from
[images.nasa.gov](https://images.nasa.gov/details/KSC-20260902-PH-KLS01_0179)
to reproduce it at full size.

## How it works

1. **Quantise** - round each RGB channel down into buckets (default 10), so
   near-identical shades count as one colour.
2. **Count** - tally how many pixels share each colour.
3. **Remove ignored** - drop colours in any family you excluded.
4. **Rank** - sort by frequency and return the top N.

Names come from van de Weijer's Color Naming model: RGB space split into a
32x32x32 grid, each cell mapped to the most probable of the eleven basic
colour names (black, blue, brown, grey, green, orange, pink, purple, red,
white, yellow). Naming is one array lookup - it beat hand-tuned HSV and
CIELAB nearest-reference matching on both accuracy and speed (~17x).

## Accuracy

Against 387 human-judged samples from the CVC fuzzy colour naming data set:

```
matches the name people chose      249/387 (64.3%)
on samples people agreed on        220/298 (73.8%)
```

Most failures are on genuinely ambiguous colours. The clearest weakness is
white over-predicted on light greys - grey scores worst of the eleven.

## Development

```bash
uv run ruff check .      # lint
uv run mypy src          # type check
uv run pytest -v         # tests
```

Image loading is the only file I/O, so the pipeline is tested on small
hand-written arrays. The analysis tests stub out the colour naming, so a
lookup table problem fails in the naming tests, not as a confusing analysis
failure.

## Layout

```
src/dominant_colour/
    colour_analysis.py     quantise, count, filter, rank
    colour_matching.py     RGB -> colour name, via the lookup table
    image_loading.py       image file -> pixel array (the only file I/O)
    image_editing.py       black out pixels by colour family
    evaluation.py          benchmark against human judgements
    cli.py                 argparse helpers, timing, error reporting
    constants.py           bucket size, lookup table column layout
    data/w2c.txt           the trained lookup table
    data/benchmark_data.txt  human colour judgements
scripts/                   thin CLIs - argparse and printing only
tests/                     one test file per src module
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

**Test image** - [NASA / Kennedy Space Center](https://images.nasa.gov/details/KSC-20260902-PH-KLS01_0179), public domain.

**Built with** NumPy and Pillow.

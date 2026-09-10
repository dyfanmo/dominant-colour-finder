from pathlib import Path

QUANTISE_BUCKET_SIZE = 10

LOOKUP_TABLE_COLUMNS = {
    0: "black",
    1: "blue",
    2: "brown",
    3: "grey",
    4: "green",
    5: "orange",
    6: "pink",
    7: "purple",
    8: "red",
    9: "white",
    10: "yellow",
}

COLOUR_NAMES = sorted(LOOKUP_TABLE_COLUMNS.values())


BINS_PER_CHANNEL = 32
BIN_WIDTH = 8


LOOKUP_TABLE_PATH = Path(__file__).parent / "data" / "w2c.txt"

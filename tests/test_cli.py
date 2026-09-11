import argparse

import pytest

from dominant_colour.cli import add_colours_argument, add_verbose_argument, measured, positive_int, report_error


def test_positive_int_parses_a_number() -> None:
    """A whole number above zero comes back as an int."""
    assert positive_int("3") == 3


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_anything_below_one(value) -> None:
    """Zero and negatives are refused, since neither makes sense here."""
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


@pytest.mark.parametrize("value", ["abc", "1.5", ""])
def test_positive_int_rejects_anything_that_is_not_a_whole_number(value) -> None:
    """Words and decimals are refused rather than silently truncated."""
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


def parse(name, argv):
    """Build a parser with one colours argument and read a command line with it."""
    parser = argparse.ArgumentParser()
    add_colours_argument(parser, name, "Colours.")

    return parser.parse_args(argv)


def test_colours_argument_accepts_the_eleven_names() -> None:
    """Known colour names are read into a list."""
    assert parse("--ignore", ["--ignore", "blue", "green"]).ignore == ["blue", "green"]


def test_colours_argument_rejects_an_unknown_name() -> None:
    """Anything outside the eleven is refused, and argparse exits."""
    with pytest.raises(SystemExit):
        parse("--ignore", ["--ignore", "mauve"])


def test_optional_colours_argument_defaults_to_empty() -> None:
    """Leaving the flag off gives an empty list, not None."""
    assert parse("--ignore", []).ignore == []


def test_positional_colours_argument_needs_at_least_one() -> None:
    """A positional colours argument isn't optional."""
    with pytest.raises(SystemExit):
        parse("colours", [])


def test_report_error_prints_to_stderr_and_returns_one(capsys) -> None:
    """The message goes to stderr so it can be piped separately from results."""
    exit_code = report_error(FileNotFoundError("no such file: photo.jpg"))

    assert exit_code == 1
    assert "no such file: photo.jpg" in capsys.readouterr().err


def test_verbose_argument_defaults_to_off() -> None:
    """Leaving the flag off means no timing output, which is the common case."""
    parser = argparse.ArgumentParser()
    add_verbose_argument(parser)

    assert parser.parse_args([]).verbose is False


def test_verbose_argument_turns_on_with_the_flag() -> None:
    """Passing --verbose flips the flag, with no value needed after it."""
    parser = argparse.ArgumentParser()
    add_verbose_argument(parser)

    assert parser.parse_args(["--verbose"]).verbose is True


def test_measured_stays_silent_when_not_verbose(capsys) -> None:
    """Without verbose the block still runs, but nothing is printed."""
    ran = False

    with measured("load", verbose=False):
        ran = True

    assert ran
    assert capsys.readouterr().out == ""


def test_measured_reports_time_and_memory_when_verbose(capsys) -> None:
    """With verbose the label comes back with a time and a peak memory figure."""
    with measured("load", verbose=True):
        pass

    report = capsys.readouterr().out
    assert "load" in report
    assert "ms" in report
    assert "MB peak" in report

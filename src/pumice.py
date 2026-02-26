#!/usr/bin/env python
"""A Pandoc JSON filter for rendering Obsidian markdown files to LaTeX."""

import re
import sys
import json
import itertools
from typing import Any, Callable
from pathlib import Path
from io import TextIOWrapper
from argparse import ArgumentParser, ArgumentError

from loguru import logger
from psutil import Process
from icontract import require, ensure
from pandocfilters import applyJSONFilters, RawInline


@ensure(lambda result: result is None or result.cmdline()[0].endswith("pandoc"))
def find_pandoc_process(process: Process | None = None) -> Process | None:
    """Find the Pandoc process that is running a JSON filter.

    :param process: The Pandoc filter process. If omitted or None, use
        that of the current process.
    :return: The Pandoc process that invoked the filter, or None if
        the filter is not running under Pandoc.
    """
    if process is None:
        process = Process()
    for parent in process.parents():
        if parent.name() == "pandoc":
            return parent
    logger.warning("Failed to find parent Pandoc process")
    return None


class SilentArgumentParser(ArgumentParser):
    """Version of ArgumentParser that raises a ValueError instead of exiting on
    a malformed input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # see https://stackoverflow.com/a/67891066
    def error(self, message: str):
        raise ArgumentError(None, message)

    def exit(self, status: int = 0, message: str | None = None):
        raise ArgumentError(None, f"Raised exit with status {status}: message")


@require(lambda cmdline: len(cmdline) > 0 and cmdline[0].endswith("pandoc"))
def extract_document_from_pandoc_cmdline(cmdline: list[str]) -> str | None:
    """Parse a Pandoc command line to find the path of the input file being processed.

    This only implements a subset of the Pandoc commands deemed most relevant
        to this use case.

    :param cmdline: The command line arguments of the process, as returned by
        `Process.cmdline()`.
    :return: The argument that corresponds to the document being processed. May
        be `-` for stdin, or None if the arguments are malformed.
        any file.
    """
    parser = SilentArgumentParser(exit_on_error=False, add_help=False)
    parser.add_argument("inputfile", nargs="?", default="-")
    parser.add_argument("--template", nargs=1)
    parser.add_argument("-s", "--standalone", action="store_true")
    parser.add_argument("-F", "--filter", nargs=1)
    parser.add_argument("-f", "--from", nargs=1)
    parser.add_argument("-t", "--to", nargs=1)
    parser.add_argument("-o", "--output", nargs=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    try:
        args = parser.parse_args(cmdline[1:])
    except ArgumentError as exception:
        logger.warning(f"Received malformed commandline {cmdline}: {exception}")
        return None
    return args.inputfile


@require(lambda process: process is None or process.cmdline[0].endswith("pandoc"))
@ensure(lambda result: result is None or result.exists())
def get_pandoc_document(process: Process | None = None) -> Path | None:
    """Find the input file that a Pandoc process is operating on.

    :param process: The Pandoc process. If omitted or None, use the one that
        has invoked the current process.
    :return: A path to the input file it is operating on, or None if it can't
        be parsed. If the input file is STDIN, return the working directory of
        the process.
    """
    process = process or find_pandoc_process()
    if process is None:
        return None
    document = extract_document_from_pandoc_cmdline(process.cmdline())
    if document is None:
        return None
    if document == "-":
        return Path(process.cwd())
    return Path(document)


@require(lambda path: path.exists())
def is_vault_root(path: Path) -> bool:
    """Determine whether a directory path refers to the root of an Obsidian
    vault."""
    return (path / ".obsidian").is_dir()


def path_ends_with(path: Path, suffix: Path) -> bool:
    """Determine whether a path ends with another path."""
    return str(path).endswith(str(suffix))


@require(lambda path: path.exists())
@ensure(
    lambda result, filename: result is None or path_ends_with(result, Path(filename))
)
def find_note_uncle(path: Path, filename: str) -> Path | None:
    """Search for a file in a directory and all of its ancestors up to the
        Obsidian vault root.

    :param path: The path of a file or directory in an Obsidian vault.
    :param filename: The name of the file to search for.
    :return: A path to the file, or None if not found.
    """
    path = path.absolute()  # XXX: how does this work with symlinks?
    for parent in itertools.chain([path], path.parents):
        if parent.is_dir() and (uncle := parent / filename).exists():
            return uncle
        if is_vault_root(parent):
            break
    return None


def fix_equation_environments(key: str, value, format: str, meta):
    """Fix amsmath equation environments improperly nested inside DisplayMath blocks."""
    if (
        key == "Math"
        and value[0]["t"] == "DisplayMath"
        and re.search(r"^\s*\\begin\{(gather|align|equation)\*?\}", value[1])
    ):
        return RawInline("tex", value[1])


@require(lambda include: not include.startswith("/"), "must be a relative path")
@require(lambda include: len(Path(include).parts) > 0)
@require(lambda working_dir: working_dir.is_absolute())
@ensure(
    lambda result, include: result is None
    or (result.exists() and str(result).endswith(str(Path(include))))
)
def resolve_vault_relative_include(include: str, working_dir: Path) -> Path | None:
    vault_root = None
    for parent in itertools.chain([working_dir], working_dir.parents):
        if (path := parent / include).exists():
            return path
        vault_root = parent
        if is_vault_root(parent):
            break
    assert (
        vault_root != None
    ), "it should be impossible for no path to be assigned to vault_root"
    part = Path(include)
    for root, dirnames, filenames in vault_root.walk():
        if len(part.parts) > 1:
            if part.parts[0] in dirnames and (ret := root / part).exists():
                return ret
        else:
            assert len(part.parts) > 0
            if part.parts[-1] in filenames:
                ret = root / part
                assert ret.exists()
                return ret
    logger.warning(f'Failed to resolve vault-relative path "{include}"')
    return None


@require(lambda working_dir: working_dir.is_absolute())
def vault_relative_path_resolver(
    working_dir: Path,
) -> Callable[[str, Any, str, Any], Any]:
    """Resolve relative paths to images included using the `[](image.png)` syntax."""

    def resolver(key: str, value, format: str, meta):
        if key == "Image":
            [_, _, image] = value
            if (
                not (working_dir / image[0]).is_file()
                and not Path(image[0]).is_absolute()
            ):
                logger.debug(f'Found unresolved local path "{image[0]}"')
                if resolved := resolve_vault_relative_include(image[0], working_dir):
                    logger.info(f'Resolving local path to "{str(resolved)}"')
                    image[0] = str(resolved)

    return resolver


# TODO: if a preamble already exists, preserve it and absolutize it if it is relative
@require(lambda preamble: preamble.suffix in {".sty", ""})
def insert_preamble(ast: Any, preamble: Path) -> Any:
    """Insert the LaTeX preamble path into the AST's metadata block."""
    path = str(preamble.absolute().with_suffix(""))
    ast["meta"]["preamble"] = {"t": "MetaInlines", "c": [{"t": "Str", "c": path}]}
    logger.info(f'Inserted link to preamble "{path}"')
    return ast


def filter_ast(ast: Any, format: str = "") -> Any:
    """Apply various filters to a Pandoc Markdown AST to format it for rendering.

    :param ast: The JSON AST parsed by `json.loads`.
    :param format: The Pandoc output format of the document.
    :return: A dictionary representation of the filtered AST."""

    logger.info(f"Filtering JSON for output format {format}")
    filters: list[Any] = [fix_equation_environments]
    if document := get_pandoc_document():
        document = document.absolute()
        logger.info(f'Input document is "{str(document)}"')
        # Preamble from the Extended MathJax plugin
        if preamble := find_note_uncle(document, "preamble.sty"):
            ast = insert_preamble(ast, preamble)
        filters.append(vault_relative_path_resolver(document.parent))

    logger.info(f"Applying {len(filters)} AST transformations")
    ast = json.loads(applyJSONFilters(filters, json.dumps(ast), format))
    return ast


def main():
    input = TextIOWrapper(sys.stdin.buffer, encoding="utf-8").read()
    logger.info(f"Read {len(input)} bytes")
    format = sys.argv[1] if len(sys.argv) > 1 else ""
    # XXX: This is pretty inefficient as in total we are:
    #   loading -> dumping -> loading -> walking tree -> dumping -> loading -> dumping
    # Streamlining this would require reimplementation of `applyJSONFilters`.
    ast = json.loads(input)
    filtered = filter_ast(ast, format)
    output = json.dumps(filtered)
    logger.info(f"Wrote {len(output)} bytes")
    sys.stdout.write(output)


if __name__ == "__main__":
    main()

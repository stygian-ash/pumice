#!/usr/bin/env python
"""A Pandoc JSON filter for rendering Obsidian markdown files to LaTeX."""

import re
import sys
import json
import itertools
from datetime import datetime
from typing import Any, Callable, TextIO
from pathlib import Path
from io import TextIOWrapper
from argparse import ArgumentParser, ArgumentError

import yaml
from loguru import logger
from psutil import Process
from icontract import require, ensure
from pandocfilters import applyJSONFilters, RawInline, stringify, Str


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
    parser.add_argument("--bibliography", nargs=1)
    parser.add_argument("-C", "--citeproc", action="store_true")
    parser.add_argument("--csl", nargs=1)
    parser.add_argument("--natbib", action="store_true")
    parser.add_argument("--biblatex", action="store_true")
    parser.add_argument("--metadata", nargs=1)
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


@require(lambda path: path.exists())
@ensure(lambda result: result is None or is_vault_root(result))
def get_vault_root(path: Path) -> Path | None:
    """Get the root directory of the Obsidian vault containing a path."""
    data_dir = find_note_uncle(path, ".obsidian")
    if not data_dir or not data_dir.is_dir():
        return None
    return data_dir.parent


def fix_equation_environments(key: str, value, format: str, meta):
    """Fix amsmath equation environments improperly nested inside DisplayMath blocks."""
    if format != "latex":
        return
    if (
        key == "Math"
        and value[0]["t"] == "DisplayMath"
        and re.search(
            r"^\s*\\begin\{(gather|flalign|alignat|align|equation)\*?\}", value[1]
        )
    ):
        return RawInline("tex", value[1])


@require(lambda include: not include.startswith("/"), "must be a relative path")
@require(lambda include: len(Path(include).parts) > 0)
@require(lambda working_dir: working_dir.is_absolute())
@ensure(
    lambda result, include: result is None
    or (result.exists() and str(result).endswith(str(Path(include))))
)
# FIXME: How does this work with paths containing spaces? Do we need to URL
# decode them?
# TODO: replace with obsidian CLI function
def resolve_vault_relative_include(include: str, working_dir: Path) -> Path | None:
    """Resolve a vault-relative path. Search for a file matching `include` in
    the Obsidian vault containing `working_dir`."""
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


def extend_metadata_text_field(metadata: dict[str, Any], key: str, value: str) -> bool:
    """Insert a key into the Pandoc JSON metadata block only if it does not
    already exist.

    :param metadata: The metadata block from Pandoc's JSON output.
    :param key: The name of the metadata field to insert.
    :param value: The value of the metadata field to insert.
    :return: Whether the data was inserted into the block. `False` if the
        key already existed."""
    if key not in metadata:
        logger.debug(f'Adding "{key}" field "{value}"')
        metadata[key] = {"t": "MetaInlines", "c": [Str(value)]}
        return True
    return False


# TODO: if a preamble already exists, preserve it and absolutize it if it is relative
@require(lambda preamble: preamble.suffix in {".sty", ""})
def insert_preamble(ast: Any, preamble: Path) -> Any:
    """Insert the LaTeX preamble path into the AST's metadata block."""
    path = str(preamble.absolute().with_suffix(""))
    if extend_metadata_text_field(ast["meta"], "preamble", path):
        logger.info(
            'Inserted link to preamble "'
            + str(
                preamble.absolute()
                .relative_to(preamble.parent.parent)
                .with_suffix(".sty")
            )
            + '"'
        )
    return ast


def read_metadata_block(file: TextIO) -> dict[str, Any]:
    """Read the Pandoc/Obsidian metadata block from a Markdown file."""
    if file.readline().strip() != "---":
        return {}
    lines = []
    while (line := file.readline()).strip() != "---":
        lines.append(line)
    metadata = yaml.safe_load("".join(lines))
    return metadata


@require(lambda name: len(name.strip().split()) > 0)
@ensure(lambda result: result.startswith("Dr. ") or result.startswith("Prof. "))
def shorten_instructor_name(name: str) -> str:
    '''Shorten an instructor's full name and potentially add a title.

    :param name: The instructor's name from the front matter, in western order,
        with optional title and potentially surrounded by a wikilink.
    :return: A shortened form of the instructor's name, such as "Dr. Curie" or
        "Prof. Oppenheimer". When no title is present, default to "Prof."'''
    # Remove potential wikilink
    name = re.sub(r"(\[\[)?(.+?)(\]\])?", r"\2", name.strip()).strip()
    parts = name.split()
    surname = parts[-1]
    if parts[0] in {"Dr.", "Prof."}:
        title = parts[0]
    else:
        title = "Prof."
    return f"{title} {surname}"


# TODO: Should this insert all fields from the front matter, ideally in a format like `course.<field>`?
def insert_front_matter(ast: Any, front_matter: Path) -> Any:
    """Insert data from the course front matter file into the AST metadata."""
    with front_matter.open("r") as f:
        metadata = read_metadata_block(f)
    logger.info(f"Processing front matter block with {len(metadata.keys())} fields.")
    if "instructor" in metadata:
        extend_metadata_text_field(
            ast["meta"], "instructor", shorten_instructor_name(metadata["instructor"])
        )
    if "id" in metadata:
        extend_metadata_text_field(ast["meta"], "course", metadata["id"])
    return ast


def insert_title(ast: Any, document: Path) -> Any:
    """Insert an appropriate title into the metadata if none already exists."""
    extend_metadata_text_field(ast["meta"], "title", document.stem)
    return ast


def format_dates(ast: Any) -> Any:
    """Nicely format ISO 8601 dates given in the metadata."""
    for value in ast["meta"].values():
        string = stringify(value).strip()
        try:
            date = datetime.fromisoformat(string)
        except ValueError:
            continue
        date_nice = date.strftime("%B %-d, %Y")
        logger.info(f'Formatting date "{string}" as "{date_nice}"')
        value["c"] = [Str(date_nice)]
    return ast


def fix_referenced_equations(key: str, value, format: str, meta):
    """Replace any display math equations that contain a label to proper
    amsmath `equation` environments."""
    if format != "latex":
        return
    if key == "Math" and value[0]["t"] == "DisplayMath" and r"\label{" in value[1]:
        return RawInline("tex", r"\begin{equation}" + value[1] + r"\end{equation}")


# TODO: add support for headers like "## Part (a)"
# FIXME: false positives?
def shorten_heading_labels(key: str, value, format: str, meta):
    """Shorten labels to manually numbered headings with captions."""
    if key == "Header":
        [_, caption, _] = value
        if match := re.search("^(.+-[0-9])-", caption[0]):
            logger.debug(f'Shortening heading label "{caption[0]}" to "{match[1]}"')
            caption[0] = match[1]


def unwrap_horizontal_qed(key: str, value, format: str, meta):
    """Remove inline math block around the `\\hqed` (horizontal mode QED) macro."""
    if format != "latex":
        return
    if key == "Math" and value[1].strip() == r"\hqed":
        return RawInline("tex", value[1])


# TODO: support fetching CSLs if not present
def resolve_csl(ast: Any, document: Path) -> Any:
    """Resolve the absolute path to a CSL file specified in the metadata block
        if it cannot be found locally. Looks for the files in the
        `vault-root/.pandoc` directory, which is where the "Pandoc Reference List"
        extension stores them. If no CSL field is provided, specify a default of
        IEEE.

    :param ast: The JSON AST of the document.
    :param document: The path of the document on disk.
    :return: The AST modified to have a valid CSL field."""

    if "csl" not in ast["meta"]:
        extend_metadata_text_field(ast["meta"], "csl", "ieee")
    path = Path(stringify(ast["meta"]["csl"]))
    if (document.parent / path).exists():
        return ast
    if not path.suffix == ".csl":
        path = path.with_suffix(".csl")
    if not (document.parent / path).exists():
        root = get_vault_root(document)
        assert (
            root is not None
        ), "the input document must be a member of an Obsidian vault"
        pandoc_csl = root / ".pandoc" / path.name
        if pandoc_csl.exists():
            logger.debug(f'Resolved CSL file as ".pandoc/{pandoc_csl.name}"')
            ast["meta"]["csl"]["c"] = [Str(str(pandoc_csl))]
            return ast
        logger.warning(f'Failed to resolve CSL file "{path}"')
    return ast


def short_path(path: Path) -> str:
    """Format a Path as a short string for logging."""
    if len(path.parts) <= 2:
        return str(path)
    return path.parent.name + "/" + path.name


def filter_ast(ast: Any, format: str = "") -> Any:
    """Apply various filters to a Pandoc Markdown AST to format it for rendering.

    :param ast: The JSON AST parsed by `json.loads`.
    :param format: The Pandoc output format of the document.
    :return: A dictionary representation of the filtered AST."""

    logger.info(f"Filtering JSON for output format {format}")
    filters: list[Any] = [
        fix_equation_environments,
        fix_referenced_equations,
        shorten_heading_labels,
        unwrap_horizontal_qed,
    ]
    if document := get_pandoc_document():
        document = document.absolute()
        logger.info(f'Input document is "{document.name}"')
        # Preamble from the Extended MathJax plugin
        if preamble := find_note_uncle(document, "preamble.sty"):
            ast = insert_preamble(ast, preamble)
        filters.append(vault_relative_path_resolver(document.parent))
        if front_matter := find_note_uncle(document, "Front Matter.md"):
            logger.info(f'Found front matter at "{short_path(front_matter)}"')
            ast = insert_front_matter(ast, front_matter)
        ast = insert_title(ast, document)
        if bibliography := find_note_uncle(document, "bibliography.bib"):
            logger.info(f'Found bibliography at "{short_path(bibliography)}"')
            extend_metadata_text_field(
                ast["meta"], "bibliography", str(bibliography.absolute())
            )
        ast = resolve_csl(ast, document)
    ast = format_dates(ast)

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

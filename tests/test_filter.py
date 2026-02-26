import pytest
from pumice import *


@pytest.mark.parametrize(
    "cmdline,expected",
    [
        (["pandoc", "test.md", "-o", "test.pdf"], "test.md"),
        (["pandoc", "-o", "test.pdf"], "-"),
        (["pandoc"], "-"),
        (["pandoc", "-h"], None),
        (["pandoc", "-t"], None),
        (
            [
                "pandoc",
                "-f",
                "markdown",
                "-t",
                "pdf",
                "-s",
                "--template",
                "template.latex",
                "--filter",
                "filter.py",
                "~/documents/Homework 1.md",
            ],
            "~/documents/Homework 1.md",
        ),
    ],
)
def test_extract_pandoc_cmdline(cmdline: list[str], expected: str | None):
    assert extract_document_from_pandoc_cmdline(cmdline) == expected

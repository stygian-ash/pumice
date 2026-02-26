# Pumice

A little tool for rendering my Obsidian notes to PDFs with LaTeX and Pandoc.
Presently only supports rendering homework submissions.

## Usage
```bash
pandoc -t pdf --template ./template.latex --filter ./pumice.py in.md -o out.pdf
```

Supports loading the `preamble.sty` file supported by [Obsidian Extended MathJax]
if it is present in the vault.

References the following keys from the YAML metadata block, in addition
to those supported by the default LaTeX template:

| Key          | Usage                                                   |
|--------------|---------------------------------------------------------|
| `course`     | Course number of the class, for use in the title block. |
| `instructor` | Abbreviated instructor's name. For title block.         |
| `due-date`   | Date the homework assignment is due. For title block.   |

[Obsidian Extended MathJax]: https://github.com/wei2912/obsidian-latex

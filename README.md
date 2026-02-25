# Pumice

A little tool for rendering my Obsidian notes to LaTeX PDFs. 
Presently only supports rendering homework submissions.

## Usage
```bash
pandoc -t pdf --template ./template.latex --filter ./filter.py in.md -o out.pdf
```

References the following keys from the YAML metadata block, in addition
to those supported by the default LaTeX template:

| Key          | Usage                                                   |
|--------------|---------------------------------------------------------|
| `preamble`   | Name of a LaTeX package to load.                        |
| `course`     | Course number of the class, for use in the title block. |
| `instructor` | Abbreviated instructor's name. For title block.         |
| `due-date`   | Date the homework assignment is due. For title block.   |

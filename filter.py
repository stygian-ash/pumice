#!/usr/bin/env python

import sys
import re

from pandocfilters import toJSONFilters, RawInline


def fix_equation_environments(key: str, value, format: str, meta):
    """Fix amsmath equation environments improperly nested inside DisplayMath blocks."""
    if (
        key == "Math"
        and value[0]["t"] == "DisplayMath"
        and re.search(r"^\s*\\begin\{(gather|align|equation)\*?\}", value[1])
    ):
        return RawInline("tex", value[1])


if __name__ == "__main__":
    toJSONFilters([fix_equation_environments])

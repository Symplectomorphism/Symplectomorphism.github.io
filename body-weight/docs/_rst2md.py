"""Turn the reST left in quartodoc's output into Markdown, in place.

Docstrings in wdm are reST-flavored: ``.. math::`` blocks, :math:`x` inline,
and cross-reference roles such as :func:`record`. Markdown would print all of
these literally.
"""

import re
import sys
from pathlib import Path

ROLE = re.compile(r":(?:func|class|meth|mod|attr|data|obj|exc):`([^`]+)`")
INLINE_MATH = re.compile(r":math:`([^`]+)`")


def convert(text: str) -> str:
    text = INLINE_MATH.sub(r"$\1$", text)
    text = ROLE.sub(r"`\1`", text)
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        m = re.match(r"^(\s*)\.\. math::\s*(.*)$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent = len(m.group(1))
        body = [m.group(2)] if m.group(2) else []
        i += 1
        # The block is every following line that is blank or indented deeper.
        while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
            body.append(lines[i])
            i += 1
        while body and not body[-1].strip():
            body.pop()
        math = "\n".join(line.strip() for line in body if line.strip())
        if ("&" in math or "\\\\" in math) and "\\begin" not in math:
            # reST's math directive aligns multi-line blocks itself; MathJax needs it said.
            math = "\\begin{aligned}\n" + math + "\n\\end{aligned}"
        out += ["$$", math, "$$", ""]
    return "\n".join(out)


for path in sys.argv[1:]:
    p = Path(path)
    p.write_text(convert(p.read_text()))

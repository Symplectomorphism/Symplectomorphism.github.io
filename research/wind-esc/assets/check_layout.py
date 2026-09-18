#!/usr/bin/env python3
"""Flag slides whose content is likely to overrun the frame.

    python check_layout.py                      # all decks here
    python check_layout.py gradient-estimation  # one

This is a HEURISTIC, not a measurement. It adds up a rough vertical budget per
slide from the figure class, the number of cards and callouts, display equations
and prose lines, and reports anything over the threshold. Prose is wrapped by
character count rather than by source line, because Quarto rejoins hard-wrapped
source into one paragraph; display equations are charged by the number of lines
they span; inline footnotes are charged for the <aside> they render into.

An exact check would render the deck headlessly and measure each section, which
is worth doing if this proves too blunt. An attempt at that did not work: reveal
lays out only the current slide, and forcing all of them into flow to measure
them made Chrome's DOM dump unreliable.

The slide box is 760px with a 56px footer band reserved by deck.scss, so the
usable height is about 704px, or roughly 93vh of the 760.
"""

from __future__ import annotations

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 93          # vh; 760px frame less the 56px footer band

W = {"fig-tall": 73, "fig-wide": 46, "fig-band": 40, "fig": 64,
     "card": 4.5, "callout": 4.0, "heading": 9,
     "eqline": 5.0, "eqmin": 7.0, "footnote": 4.0}

CPL = 95        # characters per rendered line at .smaller, 1280px wide
LINE = 2.45     # vh per rendered line of prose
FN_CPL = 190    # the footnote aside is set much smaller
FN_LINE = 1.5


def slides(path):
    s = open(path, encoding="utf-8").read()
    if "\n## " not in s:
        return
    parts = re.split(r"(?m)^(#{1,2} .*)$", s[s.index("\n## "):])
    for i in range(1, len(parts), 2):
        yield (i + 1) // 2 + 1, re.sub(r"\s*\{.*$", "", parts[i]).lstrip("# "), parts[i + 1]


def _text_h(chunk, cpl=CPL, line=LINE):
    """Height of a prose run, wrapping by character count rather than by
    source line: Quarto rejoins hard-wrapped source into one paragraph."""
    h = 0.0
    for para in re.split(r"\n\s*\n", chunk):
        t = " ".join(x.strip() for x in para.splitlines() if x.strip())
        t = re.sub(r"\^\[[^\]]*\]", "", t)          # footnotes counted separately
        if not t:
            continue
        h += max(1, -(-len(t) // cpl)) * line
    return h


def weigh(body):
    cut = body.find("::: {.notes}")
    vis = body[:cut] if cut >= 0 else body
    total = W["heading"]

    # figures
    for m in re.finditer(r"!\[\]\([^)]*\)\{([^}]*)\}", vis):
        cls = m.group(1)
        for k in ("fig-tall", "fig-wide", "fig-band"):
            if k in cls:
                total += W[k]
                break
        else:
            total += W["fig"]

    # display equations: a $$ block is as tall as the lines it spans
    for m in re.finditer(r"\$\$(.*?)\$\$", vis, re.S):
        n = len([l for l in m.group(1).strip().splitlines() if l.strip()])
        total += max(W["eqmin"], n * W["eqline"])

    # inline footnotes render as an <aside> under the slide
    for m in re.finditer(r"\^\[([^\]]*)\]", vis):
        total += W["footnote"] + _text_h(m.group(1), FN_CPL, FN_LINE)

    # container chrome, then the prose inside everything
    total += W["card"] * len(re.findall(r"::: \{?\.card", vis))
    total += W["callout"] * len(re.findall(r"::: \{?\.(claim|warn|openq)", vis))

    stripped = re.sub(r"\$\$.*?\$\$", "", vis, flags=re.S)
    stripped = re.sub(r"!\[\]\([^)]*\)\{[^}]*\}", "", stripped)
    stripped = re.sub(r"(?m)^:::.*$", "", stripped)
    total += _text_h(stripped)
    return total


def main(only=None):
    flagged = 0
    for f in sorted(glob.glob(os.path.join(HERE, "..", "*.qmd"))):
        name = os.path.basename(f)[:-4]
        if name == "index" or (only and only not in name):
            continue
        rows = [(n, t, weigh(b)) for n, t, b in slides(f)]
        over = [r for r in rows if r[2] > THRESHOLD]
        print(f"\n{name}: {len(rows)} slides, {len(over)} over {THRESHOLD}vh")
        for n, title, v in over:
            print(f"    slide {n:>3}  ~{v:>3.0f}vh   {title}")
        flagged += len(over)
    print(f"\n{flagged} slide(s) worth checking by eye" if flagged else "\nnothing flagged")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))

#!/usr/bin/env python3
"""Fail the figure build when text collides with text, or escapes its box.

Text placement in these figures is done by hand in data coordinates, which
makes overlaps easy to introduce and easy to miss: matplotlib will happily draw
a label straight through a box edge, and the two most common causes are subtle.

  1. `ax.text` defaults to `va="baseline"`, and for a MULTI-LINE string that
     anchors the last line, so earlier lines grow UPWARD from the y you gave.
     A two-line caption placed below a heading will climb into it.
  2. A label positioned next to a box by eye, without measuring either extent.

    python check_overlaps.py            # check every registered figure
    python check_overlaps.py zap-demod  # check one

Run it after make_figures.py. It re-runs each figure function, measures the
rendered extent of every text artist, and reports collisions.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import make_figures                       # noqa: E402

# A label may legitimately sit on a wire or a shaded span; the rule is only
# that text must not run into other text, and must not cross a box's edge.
PAD = 1.0          # points of slack before two texts count as colliding


def _boxes(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts, patches = [], []
    for ax in fig.axes:
        for t in ax.texts:
            s = t.get_text().strip()
            if s:
                texts.append((s, t.get_window_extent(r)))
        for p in ax.patches:
            if isinstance(p, FancyBboxPatch):
                patches.append(p.get_window_extent(r))
    return texts, patches


def _overlap(a, b, pad=PAD):
    return (a.x0 < b.x1 - pad and b.x0 < a.x1 - pad
            and a.y0 < b.y1 - pad and b.y0 < a.y1 - pad)


def _straddles(t, p):
    """True if the text crosses a box edge: partly inside, partly outside."""
    inside = (p.x0 <= t.x0 and t.x1 <= p.x1 and p.y0 <= t.y0 and t.y1 <= p.y1)
    return _overlap(t, p) and not inside


def check(fn, name):
    fig = fn.__globals__["plt"].figure() if False else None
    out = os.path.join(HERE, name)
    fn(out)                      # regenerate, then re-open the live figure
    fig = plt.figure(plt.get_fignums()[-1]) if plt.get_fignums() else None
    if fig is None:              # save() closes it, so rebuild under a hook
        return []
    return fig


def main(only=None):
    bad = 0
    for fn, name in make_figures.JOBS:
        if only and not name.startswith(only.replace(".svg", "")):
            continue
        # run the figure with save() stubbed so the figure stays open
        import windkit
        real_save, kept = windkit.save, []
        windkit.save = lambda fig, path, pad=0.12: kept.append(fig)
        for mod in ("fig_setup", "fig_esc", "fig_open", "fig_zap", "fig_three"):
            m = sys.modules.get(mod)
            if m and hasattr(m, "save"):
                m.save = windkit.save
        try:
            fn(os.path.join(HERE, name))
        finally:
            windkit.save = real_save
            for mod in ("fig_setup", "fig_esc", "fig_open", "fig_zap", "fig_three"):
                m = sys.modules.get(mod)
                if m and hasattr(m, "save"):
                    m.save = real_save
        if not kept:
            continue
        fig = kept[-1]
        texts, patches = _boxes(fig)
        hits = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if _overlap(texts[i][1], texts[j][1]):
                    hits.append(f"text/text  {texts[i][0][:34]!r} x {texts[j][0][:34]!r}")
            for p in patches:
                if _straddles(texts[i][1], p):
                    hits.append(f"text/box   {texts[i][0][:46]!r} crosses a box edge")
        plt.close(fig)
        if hits:
            bad += len(hits)
            print(f"\n{name}")
            for hh in sorted(set(hits)):
                print(f"    {hh}")
    print(f"\n{bad} collision(s)" if bad else "\nno collisions")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))


def text_over_lines(fig, tol=0.0):
    """Report text whose box is crossed by a drawn line or span.

    check_overlaps originally compared text against other text and against box
    patches only, which let a label sit underneath a thick horizontal marker
    without complaint.  Lines are sampled along their length and any sample
    inside a text box is a hit.
    """
    import numpy as np
    r = fig.canvas.get_renderer()
    hits = []
    for ax in fig.axes:
        boxes = [(t, t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
        for ln in ax.lines:
            xy = ln.get_xydata()
            if len(xy) < 2:
                continue
            pts = ax.transData.transform(xy)
            lw = max(ln.get_linewidth(), 1.0) * fig.dpi / 72.0 / 2.0
            dense = []
            for a, b in zip(pts[:-1], pts[1:]):
                n = max(2, int(np.hypot(*(b - a)) / 3))
                dense += [a + (b - a) * t for t in np.linspace(0, 1, n)]
            for t, bb in boxes:
                for px, py in dense:
                    if (bb.x0 - tol < px < bb.x1 + tol
                            and bb.y0 - lw < py < bb.y1 + lw):
                        hits.append((t.get_text()[:40].replace("\n", " "),
                                     ln.get_label()))
                        break
                else:
                    continue
                break
    return hits

#!/usr/bin/env python3
"""Measure the real rendered height of every slide, one at a time.

    ../../../.venv/bin/python measure_slides.py                  # all decks
    ../../../.venv/bin/python measure_slides.py gradient-estimation

Reveal only lays out the current slide, so this navigates to each one, waits for
MathJax and images to settle, and reads the union of the child bounding boxes
against the slide frame. Anything at or over 100% of the usable height overflows
and will be clipped or scrolled in the browser.
"""
from __future__ import annotations
import glob, os, sys, json
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
UNDERFILL = 45          # %, below this a slide is mostly empty
SITE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_site", "research", "wind-esc"))

JS = """() => {
  // with vertical stacks BOTH the stack and the inner slide carry .present
  const sec = document.querySelector('section.present section.present')
           || document.querySelector('section.present');
  if (!sec) return null;
  const deck = document.querySelector('.reveal .slides');
  // deck.scss reserves a footer band at the bottom of every slide
  const foot = document.querySelector('.reveal .footer');
  const fh = foot ? foot.getBoundingClientRect().height : 0;
  const usable = deck.getBoundingClientRect().height - fh;
  // Flowed content must fit above the footer band.  Absolutely positioned
  // children (.footnote) are deliberately pinned into that band by deck.scss,
  // so they are only checked against the frame itself.
  const full = deck.getBoundingClientRect();
  let top = Infinity, bot = -Infinity, pinned = -Infinity;
  for (const el of sec.children) {
    if (el.tagName === 'ASIDE' && el.classList.contains('notes')) continue;
    const r = el.getBoundingClientRect();
    if (r.height === 0) continue;
    if (getComputedStyle(el).position === 'absolute') {
      pinned = Math.max(pinned, r.bottom);
      continue;
    }
    top = Math.min(top, r.top); bot = Math.max(bot, r.bottom);
  }
  if (top === Infinity) { top = full.top; bot = full.top; }
  const h = (bot - top);
  const spill = pinned > full.bottom ? pinned - full.bottom : 0;
  const title = (sec.querySelector('h1,h2') || {}).textContent || '(no title)';
  // a figure reveal has squeezed to nothing: almost always an un-classed image
  // that picked up .r-stretch on a slide with too much text
  let tiny = 0;
  for (const im of sec.querySelectorAll('img')) {
    const b = im.getBoundingClientRect();
    if (b.height > 0 && b.height < 180) tiny++;
  }
  return {pct: 100 * h / usable, title: title.trim(), h, usable, spill, tiny};
}"""

def measure(page, url):
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(1200)
    page.evaluate("() => Reveal.slide(0, 0)")
    page.wait_for_timeout(300)
    n = page.evaluate("() => Reveal.getTotalSlides()")
    out, i = [], 0
    while True:
        i += 1
        r = page.evaluate(JS)
        if r:
            out.append((i, r["title"], r["pct"], r["spill"], r["tiny"]))
        if page.evaluate("() => Reveal.isLastSlide()") or i > n + 5:
            break
        page.evaluate("() => Reveal.next()")
        page.wait_for_timeout(190)
    return out

def main(only=None):
    decks = sorted(glob.glob(os.path.join(SITE, "*.html")))
    decks = [d for d in decks if "-unused" not in d and not d.endswith("index.html")]
    worst = 0
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1600, "height": 950})
        for d in decks:
            name = os.path.basename(d)[:-5]
            if only and only not in name: continue
            rows = measure(pg, "file://" + d)
            over = [r for r in rows if r[2] >= 100 or r[3] > 1 or r[4]]
            thin = [r for r in rows if r[2] < UNDERFILL and not r[4]]
            print(f"\n{name}: {len(rows)} slides, {len(over)} overfull, "
                  f"{len(thin)} under {UNDERFILL}% full")
            for n, t, p, sp, ti in over:
                tag = ""
                if sp > 1:
                    tag += f"  (+{sp:.0f}px pinned past frame)"
                if ti:
                    tag += f"  ({ti} collapsed figure(s))"
                print(f"    OVER  {n:>3}  {p:5.1f}%  {t}{tag}")
            for n, t, p, sp, ti in thin:
                print(f"    thin  {n:>3}  {p:5.1f}%  {t}")
            if rows: worst = max(worst, max(r[2] for r in rows))
        b.close()
    print(f"\nworst slide overall: {worst:.1f}% of usable height")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))

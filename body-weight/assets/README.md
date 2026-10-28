# Figures for the body-weight dynamics deck

Two steps. The first needs the `wdm` package and the observation log from the
(private) `glorious-beard` repository; the second needs only the site venv.

    uv run --project ~/repos/glorious-beard/weight-dynamics/code python compute.py
    ../../.venv/bin/python make_figures.py

`compute.py` writes `data/*.json`; `make_figures.py` draws every `*.svg` from
those, so a figure can be restyled without rerunning the model. Re-run both
after logging more days to bring the deck up to date (`--data DIR` points
`compute.py` at a different `observations.csv`/`subject.toml`).

| file | what it holds |
|------|---------------|
| `data/real.json` | phase-1 noise fit, per-attempt trend smoother, joint filter, the plan in `code/plan.csv` |
| `data/synthetic.json` | the `wdm demo` subject: 120 logged days with known truth |
| `data/timescales.json` | one simulated diet at prior parameters |
| `data/forbes.json` | the Forbes partition curve |

Series colors are four slots (`#0f8a6c`, `#b07d06`, `#3a6fb0`, `#c0392b`),
checked as a set for color-vision-deficiency separation on every pair. The
deck's darker green `#175b48` failed that check against the blue, so figures
use a lighter green; no figure uses more than these four. Text in figures is
always ink, never a series color.

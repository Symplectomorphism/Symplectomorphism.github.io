#!/bin/sh
# Regenerate the API reference from wdm's docstrings (see _quartodoc.yml), then
# turn the docstrings' reST (math blocks, :func: roles) into Markdown.
set -e
cd "$(dirname "$0")"
uvx --from 'quartodoc==0.11.1' --with 'griffe<1.15' quartodoc build --config _quartodoc.yml
python3 _rst2md.py reference/*.qmd

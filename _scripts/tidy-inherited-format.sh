#!/bin/sh
# Reveal.js decks inherit the website's project-wide `format: html`, and Quarto
# (1.10) offers no way to switch an inherited format off for a single document.
# Each deck therefore redirects that unwanted HTML build to `<name>-unused.html`
# and this script rewrites those into no-index redirects to the real slides.
#
# The files must continue to EXIST: `quarto preview` hashes every rendered
# output after post-render, so deleting them breaks preview.
set -eu

SITE="${QUARTO_PROJECT_OUTPUT_DIR:-_site}"
[ -d "$SITE" ] || exit 0

find "$SITE" -name '*-unused.html' -type f | while IFS= read -r f; do
  slug=$(basename "$f" -unused.html)
  cat > "$f" <<EOF
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex, nofollow">
<meta http-equiv="refresh" content="0; url=./$slug.html">
<title>Redirecting to the slides</title>
</head>
<body><p><a href="./$slug.html">Continue to the slides</a>.</p></body>
</html>
EOF
done

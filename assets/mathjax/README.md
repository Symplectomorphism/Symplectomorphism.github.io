# Vendored MathJax

MathJax 3.2.2, `tex-chtml-full` build, copied from the npm tarball:

```sh
curl -sSL https://registry.npmjs.org/mathjax/-/mathjax-3.2.2.tgz | tar xz
cp package/es5/tex-chtml-full.js                 assets/mathjax/
cp -r package/es5/output/chtml/fonts/woff-v2     assets/mathjax/output/chtml/fonts/
cp package/LICENSE                               assets/mathjax/
```

Only the pieces the site actually uses are kept: the combined TeX -> CommonHTML
renderer and the 23 woff files it requests. Nothing else from the package is
needed, and the directory layout matters -- MathJax resolves its font directory
as `output/chtml/fonts/woff-v2` relative to the script it was loaded from, so
the two must stay in this arrangement.

It is vendored rather than loaded from a CDN because the fonts are fetched
lazily, one file per glyph class, while a slide is being read. A request that
does not arrive does not raise an error; CommonHTML silently falls back to the
page font, which is how the same deck can render correctly on one machine and
badly on another. Serving them from this site puts the math fonts on the same
footing as Source Sans Pro, which reveal.js has always bundled.

Referenced from `_quarto.yml` (`html-math-method.url`, and `resources:` so the
woff files are copied into `_site`) and from the `revealjs` block of each deck
that contains math.

Apache-2.0; see LICENSE.

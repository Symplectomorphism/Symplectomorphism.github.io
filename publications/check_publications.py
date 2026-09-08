"""Check rendered publication coverage, resource links and incoming anchors.
Run from the project root after `quarto render`. Uses only Python's standard library.
"""
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs

root = Path(__file__).resolve().parents[1]
class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids, self.links, self.entries, self.years = [], [], [], []
        self.feed(path.read_text())
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'a':
            self.links.append(attrs.get('href', ''))
        if 'csl-entry' in attrs.get('class', '').split():
            self.entries.append(attrs['id'])
        if tag == 'section' and attrs.get('id', '').startswith('year-'):
            self.years.append(attrs['id'][5:])

metadata = json.loads((root / 'publications/links.json').read_text())
bib = (root / 'references.bib').read_text()
keys = re.findall(r'@\w+\{([^,]+),', bib)
assert len(keys) == len(set(keys)), 'Duplicate BibTeX keys'
assert set(keys) == set(metadata), 'BibTeX and resource metadata disagree'
page = Page(root / '_site/publications.html')
assert set(page.entries) == {'ref-' + key for key in keys}, 'Missing or extra rendered entries'
assert len(page.ids) == len(set(page.ids)), 'Duplicate HTML anchors'
assert page.years == sorted(page.years, reverse=True), 'Years out of order'
assert len(page.years) == len(set(page.years)), 'Duplicate year groups'
assert all('#year-' + year in page.links for year in page.years), 'Year navigation missing'
assert (root / '_site/references.bib').read_text() == bib, 'Download is missing or stale'
for key, info in metadata.items():
    target = 'Ckg_3IcAAAAJ:' + info['scholar']
    assert any(parse_qs(urlsplit(href).query).get('citation_for_view') == [target]
               for href in page.links), 'Missing Scholar link: ' + key
    for alias in info.get('aliases', []):
        assert 'ref-' + alias in page.ids, 'Legacy anchor missing: ' + alias
for file in (root / '_site').rglob('*.html'):
    for href in Page(file).links:
        url = urlsplit(href)
        if not url.netloc and url.path.endswith('publications.html') and url.fragment:
            assert unquote(url.fragment) in page.ids, f'Broken publication anchor in {file}: {href}'
assert not re.search(r'accepted, to appear|just-accepted|under review|year\s*=\s*\{submitted', bib, re.I)
assert len(re.findall(r'class="publication-kind">Preprint<', (root / '_site/publications.html').read_text())) == 2
assert any('par.nsf.gov' in href for href in page.links), 'Open manuscript link missing'
assert any('github.com/mal-boisestate/afilament' in href for href in page.links), 'Code link missing'
print(f'PASS: {len(keys)} publications, {len(page.years)} year groups, Scholar links, downloads, aliases and all incoming publication anchors.')

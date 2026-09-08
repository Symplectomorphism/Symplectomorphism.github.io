-- Render a year-grouped bibliography from BibTeX, with curated resource links.
function Pandoc(doc)
  if not FORMAT:match('html') then return doc end
  local file = assert(io.open('publications/links.json', 'r'))
  local resources = pandoc.json.decode(file:read('*a'))
  file:close()
  local records = {}
  local kinds = {['article-journal']='Journal article', ['paper-conference']='Conference paper',
    chapter='Book chapter', thesis='Thesis', webpage='Other work'}
  for _, ref in ipairs(pandoc.utils.references(doc)) do
    local id = pandoc.utils.stringify(ref.id)
    local extra = resources[id] or {}
    local year = 'Undated'
    if ref.issued and ref.issued['date-parts'] then
      year = pandoc.utils.stringify(ref.issued['date-parts'][1][1])
    end
    records['ref-' .. id] = {
      id=id, year=year, title=pandoc.utils.stringify(ref.title),
      kind=extra.kind or kinds[pandoc.utils.stringify(ref.type)] or 'Other work',
      doi=pandoc.utils.stringify(ref.doi or ref.DOI or ''), url=pandoc.utils.stringify(ref.url or ref.URL or ''), extra=extra
    }
  end
  doc = pandoc.utils.citeproc(doc)
  return doc:walk({Div = function(div)
    if div.identifier ~= 'refs' then return nil end
    local groups, years = {}, {}
    for _, entry in ipairs(div.content) do
      local record = records[entry.identifier]
      if record then
        if not groups[record.year] then
          groups[record.year] = {}; table.insert(years, record.year)
        end
        for _, alias in ipairs(record.extra.aliases or {}) do
          entry.content:insert(1, pandoc.Plain({pandoc.Span({}, pandoc.Attr('ref-' .. alias))}))
        end
        entry.content:insert(1, pandoc.Para({pandoc.Span(record.kind, pandoc.Attr('', {'publication-kind'}))}))
        local links = pandoc.List()
        local function link(label, url)
          if #links > 0 then links:insert(pandoc.Space()); links:insert(pandoc.Str('·')); links:insert(pandoc.Space()) end
          links:insert(pandoc.Link(label, url))
        end
        if record.doi ~= '' then link('DOI', 'https://doi.org/' .. record.doi) end
        if record.url ~= '' and record.url ~= 'https://doi.org/' .. record.doi then link('Read paper', record.url) end
        if record.extra.code then link('Code', record.extra.code) end
        if record.extra.scholar then
          link('Google Scholar', 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Ckg_3IcAAAAJ&citation_for_view=Ckg_3IcAAAAJ:' .. record.extra.scholar)
        else
          local query = ('"' .. record.title .. '"'):gsub('([^%w%-_%.~])', function(c) return string.format('%%%02X', string.byte(c)) end)
          link('Search Google Scholar', 'https://scholar.google.com/scholar?q=' .. query)
        end
        entry.content:insert(pandoc.Div({pandoc.Para(links)}, pandoc.Attr('', {'publication-links'})))
        table.insert(groups[record.year], entry)
      end
    end
    table.sort(years, function(a,b)
      if a == 'Undated' then return false end
      if b == 'Undated' then return true end
      return a > b
    end)
    local blocks = pandoc.List()
    for _, year in ipairs(years) do
      blocks:insert(pandoc.Header(2, year .. ' (' .. #groups[year] .. ')', pandoc.Attr('year-' .. year)))
      for _, entry in ipairs(groups[year]) do blocks:insert(entry) end
    end
    local navigation = pandoc.List({pandoc.Str("Browse by year:"), pandoc.Space()})
    for i, year in ipairs(years) do
      if i > 1 then navigation:insert(pandoc.Space()); navigation:insert(pandoc.Str("·")); navigation:insert(pandoc.Space()) end
      navigation:insert(pandoc.Link(year .. " (" .. #groups[year] .. ")", "#year-" .. year))
    end
    blocks:insert(1, pandoc.Div({pandoc.Para(navigation)}, pandoc.Attr("", {"publication-years"})))
    div.content = blocks
    return div
  end})
end

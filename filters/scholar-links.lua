-- Derive Scholar searches from BibTeX at render time, so new papers get links too.
function Pandoc(doc)
  if not FORMAT:match('html') then return doc end
  local titles = {}
  for _, ref in ipairs(pandoc.utils.references(doc)) do
    local kind = pandoc.utils.stringify(ref.type or '')
    if ref.title and kind ~= 'webpage' then
      titles['ref-' .. pandoc.utils.stringify(ref.id)] = pandoc.utils.stringify(ref.title)
    end
  end
  doc = pandoc.utils.citeproc(doc)
  return doc:walk({Div = function(div)
    local title = titles[div.identifier]
    if not title then return nil end
    local query = ('"' .. title .. '"'):gsub('([^%w%-_%.~])', function(c)
      return string.format('%%%02X', string.byte(c))
    end)
    div.content:insert(pandoc.Para({pandoc.Link(
      'Search Google Scholar', 'https://scholar.google.com/scholar?q=' .. query,
      'Find ' .. title .. ' on Google Scholar', pandoc.Attr('', {'scholar-link'})
    )}))
    return div
  end})
end

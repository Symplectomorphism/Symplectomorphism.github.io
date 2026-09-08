# Publishing lab news

1. Copy `posts/_template.qmd` to `posts/YYYY-MM-DD-short-title.qmd` (no leading underscore).
2. Replace the title, date (YYYY-MM-DD), description, and body with confirmed details.
3. Choose categories such as Publications, People, Awards, Talks, Lab Life, or Opportunities. Categories display on News; filtering is intentionally disabled for now.
4. Keep `draft: true` while preparing the post. Set `draft: false` when ready to publish.
5. Run `quarto render` from the site root. The News page lists published posts newest first; the homepage shows the latest three.
6. Review the rendered pages before your usual deployment.

Store photographs beside the post and add alt text. Link recruitment announcements to `/positions/index.qmd` for application details. The feeds display month and year. For a month-only announcement, use the first day of that month as the sorting date; the day is not displayed. Underscore-prefixed template and authoring files are not rendered as pages.

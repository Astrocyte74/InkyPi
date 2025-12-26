## Daily Theme Card JSON

This plugin (`daily_theme_card`) renders a **left “card” panel** (quote / word / etc) with an optional blurred illustration background, plus the **right weather sidebar**.

### Where the quote JSON lives

All built-in card sources live here:

- `src/plugins/daily_theme_card/inspiration.json`
- `src/plugins/daily_theme_card/jane_austen.json`
- `src/plugins/daily_theme_card/word_of_day.json`
- `src/plugins/daily_theme_card/movie_notting_hill.json`
- LDS group index: `src/plugins/daily_theme_card/LDS_quote_categories.json`
- LDS quote files: `src/plugins/daily_theme_card/lds_quotes/*.json`

The catalog that wires these up is:

- `src/plugins/daily_theme_card/cards.json`

### Filtering long quotes

If your left panel can’t show long quotes at a readable font size (or a top banner overlaps the first lines), you can pre-filter quote JSON files with:

```bash
python src/plugins/daily_theme_card/filter_quotes_to_fit.py src/plugins/daily_theme_card/inspiration.json \
  --panel-width 560 --panel-height 480 \
  --min-body-font-size 18 \
  --reserve-top 70 \
  --write
```

This removes items that won’t fit and annotates the remaining items with a small `layout` object (useful for debugging).

### File formats

#### Leaf quote/word JSON

Each file is a single JSON object:

- `id` (string) – must match the catalog entry id (recommended)
- `label` (string) – what shows up in the UI
- `type` (string) – `quote` or `word` (default `quote`)
- `items` (list)
  - `text` (string) – required
  - `attribution` (string) – optional

Example:

```json
{
  "id": "inspiration",
  "label": "Inspirational Quote",
  "type": "quote",
  "items": [
    { "text": "Do small things with great love.", "attribution": "Mother Teresa" }
  ]
}
```

#### Group registries (expandable categories)

`cards.json` can point at a group registry (for expandable menus like LDS Quotes). A group registry looks like:

- `categories` (list)
  - `id` (string)
  - `label` (string)
  - `type` (string) – `quote` or `word`
  - `source` (string) – path to the leaf JSON file

### Device-local JSON (not in git)

Some sources may be device-local (example: family dates) and referenced by absolute path in `cards.json`.

If you want to edit those from your Mac, put them under `/home/mcdarby/...` (so they’re writable as your user) and point the relevant `..._PATH` env var to that file.

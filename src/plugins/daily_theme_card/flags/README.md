# Flags (local cache)

This folder is intended to hold a local cache of flag assets plus a JSON manifest that references them.

If you want to populate it from Flagpedia’s CDN, use `download_flagcdn_flags.py` which downloads SVG (vector) flags and writes `flags.json`.

## Notes

- Please review/comply with Flagpedia/flagcdn licensing and terms before redistributing any downloaded assets.
- Default behavior downloads **country** flags and excludes **US state** subdivision codes (which are also present in the upstream dataset).

## Usage

From this folder:

```bash
python download_flagcdn_flags.py --codes zw --formats svg
python download_flagcdn_flags.py --formats svg --limit 10
python download_flagcdn_flags.py --formats svg png --png-width 2560
python enrich_flags_wikidata.py
```

Outputs:

- `svg/<code>.svg`
- `png/w<width>/<code>.png` (optional)
- `flags.json` manifest

PNG widths on flagcdn are discrete (commonly `20, 40, 80, 160, 320, 640, 1280, 2560`). If you pass an unsupported width (e.g. `560`), the script will snap up to the next supported size unless you add `--no-snap-png-width`.

## Enriching with facts

`enrich_flags_wikidata.py` can add a few structured “facts” per country code (capital, currency, etc.) to `flags.json` using Wikidata.

```bash
python enrich_flags_wikidata.py --facts-count 3
python enrich_flags_wikidata.py --facts-order capital population area --facts-count 3
python enrich_flags_wikidata.py --facts-order area --facts-count 1 --area-overrides area_overrides.example.json
```

When using `area`, the script compares against Alberta. By default it also uses a Wikidata override for France (`fr`) so the “area” fact reflects **metropolitan France** (Europe/Corsica) instead of total France including overseas departments.

If `area` is included, the script also writes structured fields per entry:

- `area_km2`
- `area_vs_alberta` (string, without the leading `Area:`)
- `area_vs_alberta_pct` (numeric percent)

For wiring to a compact “flag + info box” UI, each entry also gets:

- `display.title` (country name)
- `display.lines` (3 lines, prioritizing capital + population + area)
- `flag.png` / `flag.svg` (normalized flag paths)

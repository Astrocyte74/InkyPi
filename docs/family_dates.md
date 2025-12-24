# Family Dates Banner

InkyPi can optionally display a small banner in the **left panel** (art/card area) for birthdays and anniversaries.

## Configure

1. Create a local file (recommended):

- `/usr/local/inkypi/family_dates.json`

2. Populate it using the same schema as `src/plugins/daily_theme_card/family.example.json`.

3. Restart the service:

- `restart`

## Web Editor

You can edit the local file from the Web UI:

- `Settings` → `Family Dates`

## Environment options

Set these in the InkyPi `.env`:

- `INKYPI_FAMILY_DATES_PATH` (default: `/usr/local/inkypi/family_dates.json`)
- `INKYPI_FAMILY_BANNER_ENABLED` (`true`/`false`, default `true`)
- `INKYPI_FAMILY_BANNER_LEAD_DAYS` (default `3`)
- `INKYPI_FAMILY_BANNER_LEAD_DAYS_MAX` (default `14`; useful to temporarily set higher for testing)
- `INKYPI_FAMILY_SHOW_BIRTHDAY_AGE` (`true`/`false`, default `false`)
- `INKYPI_FAMILY_SHOW_ANNIVERSARY_YEARS` (`true`/`false`, default `true`)

Note: these values can also live in `family_dates.json` under `"config"`. `.env` values (when set) override the JSON config.

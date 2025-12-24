# Daily Cat Weather (AI Background + Live Forecast)

`Daily Cat Weather` generates a single Gemini background image per day (storybook-style) and shows a live weather panel (current + multi-day forecast) at your chosen refresh interval.

Layout: the illustration is on the left and a fixed weather sidebar is on the right, so weather never covers key parts of the image.

## Requirements

- `GEMINI_API_KEY` (Gemini image generation)
- `OPEN_WEATHER_MAP_SECRET` (OpenWeatherMap One Call 3.0 for the overlay)

See `docs/api_keys.md` for `.env` setup.

## Recommended Setup (set-and-forget frame)

1. In the Web UI, open the `Daily Cat Weather` plugin and configure:
   - Location (lat/lon)
   - Optional `Location Label` (e.g. `Carstairs, AB`) to show at the top of the sidebar
   - Units
   - Optional `Cat Illustration Theme` (e.g. `Paper Cutout` or `Woodblock`)
   - Optional `Holiday Theming` to add subtle nearest-holiday touches to the daily background prompt
   - `New Image Time` = `04:00` (local time)
   - `Forecast Days` = `3`
   - Model = `gemini-2.5-flash-image`
2. Click **Add to Playlist** and set refresh to `Every 1 hour` (or `Every 30 minutes` if you prefer).
3. In the Web UI **Settings** page:
   - Set `Plugin Cycle Interval` to the same or faster than your plugin refresh (e.g. `1 hour` or `30 minutes`).

## How It Refreshes

- At (and after) `04:00` local time, the next refresh generates a new background image for the day.
- Between regenerations, the cached background is reused and only the weather overlay updates.

## Telegram Reroll

- Run `/cat` (or tap `🐱 New Daily Cat` from `/help`) to delete today’s cached background and force a regenerate on the next refresh.
- Run `/cat <your idea>` to set a custom scene for the rest of today (until the next `New Image Time` rollover); the idea is auto-enhanced before generating.
- Run `/cat clear` to return to the auto scene generator.
- Run `/theme` to browse and set the Daily Cat illustration theme.

## Cache Location

Cached backgrounds are stored under `mock_display_output/daily_cat_weather/` (including a `latest_bg_<cacheId>.png` convenience copy). The cached background image is the left illustration area (without the sidebar).

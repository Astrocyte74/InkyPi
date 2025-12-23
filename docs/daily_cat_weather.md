# Daily Cat Weather (AI Background + Live Forecast)

`Daily Cat Weather` generates a single Gemini background image per day (storybook-style) and overlays live weather (current + multi-day forecast) on top at your chosen refresh interval.

## Requirements

- `GEMINI_API_KEY` (Gemini image generation)
- `OPEN_WEATHER_MAP_SECRET` (OpenWeatherMap One Call 3.0 for the overlay)

See `docs/api_keys.md` for `.env` setup.

## Recommended Setup (set-and-forget frame)

1. In the Web UI, open the `Daily Cat Weather` plugin and configure:
   - Location (lat/lon)
   - Units
   - `New Image Time` = `04:00` (local time)
   - `Forecast Days` = `3`
   - Model = `gemini-2.5-flash-image`
2. Click **Add to Playlist** and set refresh to `Every 1 hour` (or `Every 30 minutes` if you prefer).
3. In the Web UI **Settings** page:
   - Set `Plugin Cycle Interval` to the same or faster than your plugin refresh (e.g. `1 hour` or `30 minutes`).

## How It Refreshes

- At (and after) `04:00` local time, the next refresh generates a new background image for the day.
- Between regenerations, the cached background is reused and only the weather overlay updates.

## Cache Location

Cached backgrounds are stored under `mock_display_output/daily_cat_weather/` (including a `latest_bg_<cacheId>.png` convenience copy).


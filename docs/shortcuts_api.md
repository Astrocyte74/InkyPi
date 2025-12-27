# iOS Shortcuts HTTP API

InkyPi includes a small, local HTTP API intended for iOS Shortcuts (and other lightweight clients). This avoids Telegram command/UI quirks and keeps Shortcuts simple.

All endpoints live under `/api/`.

## Authentication (optional)

If you set `INKYPI_SHORTCUTS_TOKEN` in your `.env`, requests must include either:

- Header: `X-InkyPi-Token: <token>`
- Query param: `?token=<token>`

If `INKYPI_SHORTCUTS_TOKEN` is not set, the API is open on your LAN.

## iPhone note: avoid `.local` timeouts

On some networks, iOS can fail to resolve `*.local` (Bonjour/mDNS) reliably. If Safari/Shortcuts time out when using `http://inkypi.local/...`, use the Pi’s IP instead (find it on the Pi with `hostname -I`), e.g.:

- `http://192.168.7.212/api/status?image=0`

For long-term stability, set a DHCP reservation in your router so the IP doesn’t change.

## `GET /api/status`

Returns the current display status as JSON.

Common usage (fast + reliable):

- `GET /api/status?image=0`

Response includes:

- `status_text` (preformatted text, ideal for “Show Result” in Shortcuts)
- `plugin_id`, `plugin_instance`
- `last_refresh` and `next_refresh`
- `image.url` (direct URL to the current display PNG, with cache-buster)
- Optionally `image.base64_png` (large; can time out on iOS)

Query params:

- `image=0|1` (default `1`) — include the image in the response payload
- `format=base64` (default) — if `image=1`, include `image.base64_png`

Shortcut pattern:

1. Get Contents of URL → `/api/status?image=0`
2. Show Result → `status_text`
3. Get Contents of URL → `image.url`
4. Quick Look

## `GET /api/next`

Advances the frame to the next item in the active playlist and returns the updated status JSON (same shape as `/api/status`).

Common usage:

- `GET /api/next?image=0`

What “next” does:

- Always advances to the next plugin instance in the currently active playlist.
- By default, it may *reuse* a plugin’s cached image (to avoid unnecessary API calls) unless forcing is enabled.

Force behavior (`force=` controls whether the next plugin is forced to re-render):

- Default (no `force=`): **smart**
  - Forces a fresh render for `world_flags` and `daily_theme_card` (so you don’t see repeats)
  - Does not force a fresh render for `daily_cat_weather` (keeps “image of the day”)
- `force=all` (also accepts `force=1`) — force-regenerate whatever you land on (may trigger AI/API calls)
- `force=none` (also accepts `force=0`) — never force regeneration; always use cached image if not due

Query params:

- `image=0|1` (default `1`)
- `force=all|none` (default: smart when omitted)

## `POST /api/cat/reroll`

For Daily Cat Weather specifically: guarantees a *new* illustration background for “today” by bumping `rerollNonce` and clearing the Daily Cat background cache before rendering.

Common usage:

- `POST /api/cat/reroll?image=0`

Notes:

- This may trigger an AI image generation API call.
- The new image becomes the “today” image and will be reused for the rest of the day unless rerolled again.

## `POST /api/banner`

Sets or clears the top banner override (same banner used by the Web UI + Telegram).

Common usage:

- Clear: `POST /api/banner?clear=1&image=0`
- Set: `POST /api/banner?text=Merry%20Christmas!&image=0`

Notes:

- By default, the API returns quickly and the banner will appear/disappear on the next cycle.
- Optional: `refresh=1` queues an immediate refresh (async) if the current slide renders the banner:
  - `POST /api/banner?clear=1&refresh=1&image=0`

## `POST /api/ai`

Generate a one-off image from an idea using the standard 2-panel layout (left image + right weather sidebar).

Common usage (temporary):

- `POST /api/ai?idea=an%20ambitious%20cat%20building%20a%20snowman&enhance=1&image=0`

Optional params:

- `model=...` (default: first entry in `TELEGRAM_AI_DEFAULT_MODEL`, else `gemini-2.5-flash-image`)
- `style=van_gogh|illustration|drawing|far_side` (optional)
- `palette=spectra6|bw` (default `spectra6`)

Make it today’s illustration-of-the-day (persists):

- `POST /api/ai?idea=...&today=1&enhance=1&image=0`

Notes:

- `today=1` updates `daily_cat_weather` settings (`customPrompt*`, bumps `rerollNonce`) and rerolls the Daily Cat background cache.

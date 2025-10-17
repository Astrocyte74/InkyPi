# Repository Guidelines

## Project Structure & Module Organization
Core Flask services live in `src/`, with `inkypi.py` as the entry point, route blueprints in `blueprints/`, display backends in `display/`, and shared helpers under `utils/`. Each plugin ships from `src/plugins/<plugin_id>` with matching Jinja fragments in `templates/` and assets in `static/`. Installation logic sits in `install/`, while developer helpers such as `scripts/venv.sh` and `scripts/test_plugin.py` streamline local work. Reference material and community guides reside in `docs/`. Development mode writes rendered frames to `mock_display_output/` so you can review updates without hardware.

## Build, Test, and Development Commands
- `bash scripts/venv.sh` creates/activates `.venv`, installs `install/requirements-dev.txt`, and exports `PYTHONPATH`.
- `python src/inkypi.py --dev` launches the Flask server at `http://localhost:8080`, using mock display output and the `config/device_dev.json` profile.
- `sudo bash install/install.sh [-W <waveshare model>]` provisions a Raspberry Pi service with hardware drivers.
- `sudo bash install/update.sh` refreshes an existing device after pulling new code.

## Coding Style & Naming Conventions
Follow PEP 8: four-space indentation, `snake_case` identifiers, and module-level logging via `logging.getLogger`. Keep plugin directories and IDs aligned (e.g., `src/plugins/weather` → `weather`). Prefer descriptive docstrings like the ones in `src/display/display_manager.py`, and reuse existing Jinja blocks to preserve consistent theming. Run `python -m compileall src` or `python -m py_compile` when introducing new modules to catch syntax issues early.

## Testing Guidelines
There is no automated test suite yet; rely on deterministic repro steps. In dev mode, exercise plugins through the web UI and confirm the image saved to `mock_display_output/latest.png`. For plugin-specific checks, update `plugin_id` and settings in `scripts/test_plugin.py` to synthesize comparison collages across resolutions. When adding logic, document manual verification steps in your PR and attach either the generated mock image or a photo of the E-Ink panel.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commits (`feat:`, `fix:`, etc.); keep subject lines imperative and reference issues or PRs (e.g., `(#318)`) when relevant. Each PR should describe the change, list platforms or displays exercised, call out config or .env updates, and include screenshots or rendered output when visual elements shift. Mention any docs touched (such as `docs/building_plugins.md`) so reviewers can cross-check user-facing instructions.

## Security & Configuration Tips
Store API credentials in a root-level `.env` as outlined in `docs/api_keys.md`, and never commit those files. When changing default resolutions or orientation, update `config/device_dev.json` alongside the production config to keep dev parity. For new third-party services, document required keys and usage limits before merging.

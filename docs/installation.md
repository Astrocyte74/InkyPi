# InkyPi Installation

Looking for a complete, end-to-end guide? Open `docs/full-setup-guide.html` in your browser for the canonical walkthrough, including Raspberry Pi imaging, first boot, `.env` setup, plugin configuration, and common troubleshooting steps.

```bash
# From the repo root
xdg-open docs/full-setup-guide.html      # Linux desktop
open docs/full-setup-guide.html          # macOS
```

If you are browsing the repository on GitHub, use the "Raw" or "Download" buttons on `docs/full-setup-guide.html` and view the file locally in any modern browser.

## Quick Reference

1. Flash Raspberry Pi OS (Bookworm) using Raspberry Pi Imager. Configure hostname, SSH, Wi-Fi, and a non-default password before writing the card.
2. Boot the Pi, complete initial setup, and SSH into the device: `ssh <user>@inkypi.local`.
3. Install prerequisites and InkyPi:
    ```bash
    sudo apt update && sudo apt install -y git
    git clone https://github.com/fatihak/InkyPi.git
    cd InkyPi
    sudo bash install/install.sh                # Pimoroni Inky
    sudo bash install/install.sh -W <model>     # Waveshare (e.g., epd7in3f)
    ```
4. After the installer reboots the Pi, open `http://inkypi.local` (or the device IP) to access the web UI.
5. Store any plugin secrets in `/usr/local/inkypi/.env`, restart the service, and customize playlists through the browser.

## Related Docs

- `docs/api_keys.md` for required environment keys.
- `docs/troubleshooting.md` for common issues and log commands.
- `docs/development.md` for running the Flask app locally with the mock display.

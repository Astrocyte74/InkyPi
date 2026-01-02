# Tailscale Remote Access (WireGuard VPN + SSH)

This project supports secure remote access to an InkyPi device using **Tailscale** (a WireGuard-based mesh VPN). This avoids exposing SSH to the public internet and typically requires **no port forwarding**.

## What We Enabled On `inkypi`

- Installed Tailscale from the official Tailscale APT repo on Debian (Raspberry Pi OS / Debian).
- Enabled and running systemd service: `tailscaled` (auto-starts on reboot).
- Connected the device to the tailnet (authenticated via the same account you use on your laptop).
- Enabled **Tailscale SSH** (`tailscale up --ssh`).

## Install + Connect (New Device)

On the Pi (or other device):

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

You’ll be given a one-time login URL. Open it on a machine already logged into your Tailscale account (e.g., your laptop) and approve the new device.

## Enable Tailscale SSH (Recommended)

On the device:

```bash
sudo tailscale up --reset --ssh --timeout=20s
```

Notes:
- Tailscale SSH access is controlled by your tailnet’s SSH policy/ACLs in the Tailscale admin console.
- If `tailscale ssh ...` is denied, enable/configure Tailscale SSH in the admin console and restrict it to your user/devices.

## How To Connect From Your Laptop

### Option A: Regular SSH over Tailscale IP

```bash
ssh <user>@<tailscale-ip>
```

Example:

```bash
ssh mcdarby@100.67.209.49
```

### Option B: Tailscale SSH

```bash
tailscale ssh <user>@<device-name>
```

Example:

```bash
tailscale ssh mcdarby@inkypi
```

## Useful Commands (On The Device)

```bash
sudo tailscale status
sudo tailscale ip -4
sudo systemctl is-enabled tailscaled
sudo systemctl is-active tailscaled
```

Disconnect (keeps it installed):

```bash
sudo tailscale down
```

Log out of Tailscale on the device:

```bash
sudo tailscale logout
```


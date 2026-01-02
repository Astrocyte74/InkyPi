# TL;DR — Quick Recovery Checklist

If Claude starts crashing, freezing, or exiting unexpectedly on the Pi Zero 2 W:

1. Check memory and swap:
   ```bash
   free -h
   swapon --show
   ```
   - If available RAM < 80 MB or swap > 900 MB → reduce MCPs immediately.

2. Check running processes:
   ```bash
   ps -eo pid,comm,rss,args --sort=-rss | head -n 15
   ```
   - Look for large `claude`, `node`, or MCP-related processes.

3. Disable Vision MCP (most common cause):
   ```bash
   npx @z_ai/coding-helper
   ```
   - Disable Vision MCP.
   - Restart Claude.

4. Check for OOM kills:
   ```bash
   dmesg -T | tail -n 120 | rg -i "oom|killed process|out of memory" || true
   ```

5. Confirm swap configuration:
   ```bash
   swapon --show
   ```
   - Expect ~1.6 GB zram (`RamMultiplier=4`).

If all checks pass, relaunch Claude with:
```bash
claude-glm
```

# Claude Code + GLM on Raspberry Pi Zero 2 W
## Stable Configuration, MCP Management, and Memory Constraints

**Hardware**
- Raspberry Pi Zero 2 W (512 MB RAM)
- ARMv8 (64-bit)
- Debian / Raspberry Pi OS
- Kernel: 6.12.x rpi-v8

**Claude**
- Claude Code v2.x
- Model: `glm-4.7`
- Provider: **z.ai Anthropic-compatible endpoint**

---

## 1. Goals of This Setup

This configuration is designed to:

- Run **Claude Code stably** on very low RAM hardware
- Use **GLM models via z.ai**, not OpenRouter
- Keep **OpenRouter available for other tools**
- Enable **useful MCP servers** without triggering OOM kills
- Provide a simple, repeatable way to toggle MCP services

---

## 2. Claude + GLM Configuration (z.ai)

Claude Code is configured to use z.ai via `~/.claude/settings.json`.

Key points:

- Claude reads **all Anthropic settings from `settings.json`**
- No Anthropic or OpenRouter overrides are exported globally
- GLM model IDs use **native z.ai names** (`glm-4.7`, not `z-ai/glm-4.7`)

### `~/.claude/settings.json` (relevant fields)

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "<z.ai token>",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.7"
  }
}
```

---

## 3. Claude Launcher (protects against OpenRouter leakage)

OpenRouter is still installed and used elsewhere, so Claude is launched via a wrapper that clears Anthropic overrides only. This wrapper intentionally does **not** unset `OPENROUTER_API_KEY`, so OpenRouter remains available for other tools.

### `~/bin/claude-glm`

```bash
#!/usr/bin/env bash
unset \
  ANTHROPIC_API_KEY \
  ANTHROPIC_AUTH_TOKEN \
  ANTHROPIC_BASE_URL \
  ANTHROPIC_DEFAULT_SONNET_MODEL \
  ANTHROPIC_DEFAULT_OPUS_MODEL \
  ANTHROPIC_DEFAULT_HAIKU_MODEL

exec claude --dangerously-skip-permissions --model glm-4.7 "$@"
```


```bash
chmod +x ~/bin/claude-glm
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Optional alias (used throughout this setup):

```bash
alias claude-glm="$HOME/bin/claude-glm"
```

Ensure it's executable and on PATH.

---

## 4. Memory Stability (Critical on Pi Zero 2 W)

### Problem

Default swap (≈ RAM × 1) is insufficient. Claude Code will be OOM-killed.

### Solution

Note: On Raspberry Pi OS/Debian, zram is managed by **rpi-swap** (`/etc/rpi/swap.conf*`) via `rpi-swap-generator`. Editing `/etc/systemd/zram-generator.conf` alone may have no effect.

Increase zram swap to RAM × 4 using `rpi-swap`.

#### Config (drop-in)

```bash
sudo mkdir -p /etc/rpi/swap.conf.d
sudo tee /etc/rpi/swap.conf.d/10-zram-size.conf >/dev/null <<'EOF'
[Zram]
RamMultiplier=4
EOF
```

Reboot required.

### Expected healthy state

```bash
free -h
swapon --show
```

- Swap: ~1.6 GB (`/dev/zram0`)
- Available RAM: ≥ 150 MB at idle
- Swap used: < 500 MB during normal use

---

## 5. MCP Service Management (via Coding Helper)

MCP services are managed using:

```bash
npx @z_ai/coding-helper
```

Navigate to **MCP Service Management**.

### Final, Proven MCP Set (Pi Zero 2 W)

#### Enabled by default (safe, low overhead)
- ✅ **Web Search MCP**
- ✅ **Web Reader MCP**
- ✅ **ZRead MCP**

#### Disabled by default

### Enabled vs Running (Important)
Enabling an MCP does not always start a resident process. Some MCPs (notably Vision MCP) spin up local Node processes only on first use. Always validate memory impact **after exercising the tool once**.
- ❌ **Vision MCP**

---

## 6. What Each MCP Is For

### Web Search MCP
- Finds URLs and search results
- Lightweight, API-backed
- Minimal memory impact
- Safe to keep enabled

### Web Reader MCP
- Fetches and extracts readable text from URLs
- Used for article/document summaries
- Lightweight when idle
- Safe to keep enabled

### ZRead MCP
- Reads and traverses code repositories
- Excellent for multi-file code analysis
- Keeps Claude RSS low; uses swap efficiently
- Ideal default MCP for coding tasks

### Vision MCP (not enabled)
- Required for image analysis
- Spawns local Node processes
- Significantly increases swap pressure
- Enable only temporarily when needed

---

## 7. How to Toggle MCPs Safely

Use the Coding Helper UI:

```bash
npx @z_ai/coding-helper
```

- Enable Vision MCP → do the task → disable again
- Other MCPs can remain enabled

No manual config edits required.

---

## 8. Health Checks (Run Anytime)

### Memory health

```bash
free -h
swapon --show
```

| Status | Available RAM | Swap Used |
|--------|---------------|-----------|
| Green | ≥ 150 MB | ≤ 500 MB |
| Yellow | 80–150 MB | 500–900 MB |
| Red | < 80 MB | > 900 MB (risk of earlyoom kill) |

### Top memory consumers

```bash
ps -eo pid,comm,rss,args --sort=-rss | head -n 15
```

Used to confirm:
- No unexpected MCP servers running
- No runaway Node processes

### OOM / earlyoom diagnostics
```bash
dmesg -T | tail -n 120 | rg -i "oom|killed process|out of memory" || true
```
Use this to confirm whether previous instability was caused by the kernel OOM killer or `earlyoom`.

---

## 8a. Troubleshooting Flow (When Things Go Wrong)

```text
Claude crashes or exits
         │
         ▼
Run free -h / swapon
         │
 ┌───────┴────────┐
 │                │
RAM OK         RAM low
 │                │
 ▼                ▼
Check MCPs     Disable Vision MCP
 │                │
 ▼                ▼
Restart Claude   Restart Claude
         │
         ▼
Still failing?
         │
         ▼
Check dmesg for OOM / earlyoom
         │
         ▼
Increase swap or reduce MCP usage
```

---

## 9. Known-Good Summary

- Claude Code stable
- GLM via z.ai only
- OpenRouter unaffected for other tools
- MCPs tuned for low-RAM hardware
- No OOM kills during normal use

---

**Last verified:** Jan 2026
**Hardware:** Raspberry Pi Zero 2 W
**Swap:** zram (RAM × 4)

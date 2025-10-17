#!/usr/bin/env bash
set -euo pipefail

REQUIRED_FILES=(
  "/usr/local/inkypi/.env"
  "/usr/local/inkypi/src/config/device.json"
  "/usr/local/inkypi/src/config/plugins.json"
  "/etc/systemd/system/inkypi.service.d/override.conf"
)
NM_PATTERN="/etc/NetworkManager/system-connections/*.nmconnection"

if [[ ${EUID} -ne 0 ]]; then
  echo "This script must run as root (use sudo)." >&2
  exit 1
fi

declare -a TAR_ARGS=()
add_path() {
  local path="$1"
  if [[ -f "$path" ]]; then
    local rel_path="${path#/}"
    TAR_ARGS+=("-C" "/" "$rel_path")
  fi
}

for file in "${REQUIRED_FILES[@]}"; do
  add_path "$file"
done

shopt -s nullglob
for nm_file in $NM_PATTERN; do
  add_path "$nm_file"
done
shopt -u nullglob

if [[ ${#TAR_ARGS[@]} -eq 0 ]]; then
  echo "No runtime files were found to back up." >&2
  exit 1
fi

if [[ -n ${SUDO_USER:-} ]]; then
  USER_HOME=$(eval echo ~"$SUDO_USER")
else
  USER_HOME="$HOME"
fi

COMMAND=${1:-backup}
TIMESTAMP=$(date +%Y%m%d)
DEFAULT_ARCHIVE="$USER_HOME/inkypi-backup-${TIMESTAMP}.tgz"

case "$COMMAND" in
  backup)
    ARCHIVE_PATH=${2:-$DEFAULT_ARCHIVE}
    ARCHIVE_DIR=$(dirname "$ARCHIVE_PATH")
    mkdir -p "$ARCHIVE_DIR"
    tar -czf "$ARCHIVE_PATH" "${TAR_ARGS[@]}"
    echo "Backup written to $ARCHIVE_PATH"
    echo "Contents:" >&2
    tar -tzf "$ARCHIVE_PATH"
    ;;
  restore)
    ARCHIVE_PATH=${2:-}
    if [[ -z "$ARCHIVE_PATH" ]]; then
      echo "Usage: sudo $0 restore <archive.tgz>" >&2
      exit 1
    fi
    if [[ ! -f "$ARCHIVE_PATH" ]]; then
      echo "Archive not found: $ARCHIVE_PATH" >&2
      exit 1
    fi
    tar -xzf "$ARCHIVE_PATH" -C /
    echo "Restore complete. Review any .nmconnection files and update permissions if needed." >&2
    echo "Run 'systemctl daemon-reload' and 'systemctl restart inkypi' to apply restored service overrides." >&2
    ;;
  *)
    cat >&2 <<USAGE
Usage: sudo $0 [backup [archive_path]|restore <archive_path>]
  backup (default)  Create archive at ~/inkypi-backup-YYYYMMDD.tgz
  restore           Extract an archive back into system paths
USAGE
    exit 1
    ;;
esac

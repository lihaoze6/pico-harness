#!/usr/bin/env bash
# Prepare a WSL/Linux driver host for FrontierHarness Eval with the Pico adapter.
#
# Runta's CLI publishes macOS and Linux binaries only ("Runta setup currently
# targets macOS and Linux. Windows is not supported yet."), so on a Windows
# machine the FrontierHarness driver loop has to run inside WSL. This script
# installs everything needed *before* authentication and deliberately stops
# there: it never reads or writes a Runta token or a provider key.
#
# Usage (inside WSL):
#   bash /mnt/d/GithubProjects/pico-harness/benchmarks/frontierharness/wsl-setup.sh
set -euo pipefail

RUNTA_NPM_PACKAGE="@runta/runta-cli"
MIN_NODE_MAJOR=18

log() { printf '\n== %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

run_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif sudo -n true 2>/dev/null; then
    sudo "$@"
  else
    printf 'wsl-setup: needing root for: %s (run it with sudo yourself)\n' "$*" >&2
    return 1
  fi
}

if [ "$(uname -s)" != "Linux" ]; then
  echo "wsl-setup: run this inside WSL/Linux, not Git Bash or MSYS." >&2
  exit 1
fi

# shellcheck disable=SC1091
log "host: $(. /etc/os-release && echo "$PRETTY_NAME") / $(uname -m) / $(whoami)"

# ---- apt prerequisites -------------------------------------------------------
log "apt prerequisites"
missing_packages=()
have curl || missing_packages+=(curl)
have jq || missing_packages+=(jq)
have git || missing_packages+=(git)
dpkg -s ca-certificates >/dev/null 2>&1 || missing_packages+=(ca-certificates)

if [ "${#missing_packages[@]}" -gt 0 ]; then
  echo "installing: ${missing_packages[*]}"
  run_root apt-get update -y
  run_root env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing_packages[@]}"
else
  echo "already satisfied: curl jq git ca-certificates"
fi

# ---- Node.js -----------------------------------------------------------------
node_major=0
if have node; then
  node_major="$(node -p "process.versions.node.split('.')[0]" 2>/dev/null || echo 0)"
fi

log "Node.js (>= ${MIN_NODE_MAJOR})"
if [ "${node_major:-0}" -lt "$MIN_NODE_MAJOR" ]; then
  echo "current Node.js major: ${node_major:-none}; installing Node 22"
  nodesource_script="$(mktemp)"
  if curl -fsSL https://deb.nodesource.com/setup_22.x -o "$nodesource_script"; then
    run_root bash "$nodesource_script"
    run_root env DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs
  else
    echo "wsl-setup: NodeSource unreachable; falling back to the distro package" >&2
    run_root env DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs npm
  fi
  rm -f "$nodesource_script"
else
  echo "already satisfied: node $(node --version)"
fi

if ! have npm; then
  echo "wsl-setup: npm is still missing after installing Node.js" >&2
  exit 1
fi

# ---- Runta CLI ---------------------------------------------------------------
log "Runta CLI (${RUNTA_NPM_PACKAGE})"
if have runta; then
  echo "already installed: $(runta --version 2>&1 | head -1)"
else
  npm_prefix="$(npm prefix -g 2>/dev/null || echo '')"
  if [ -n "$npm_prefix" ] && [ -w "$npm_prefix" ]; then
    npm install -g "$RUNTA_NPM_PACKAGE"
  else
    run_root npm install -g "$RUNTA_NPM_PACKAGE"
  fi
fi

# ---- verification (no credentials involved) ----------------------------------
log "verification"
printf '%-8s %s\n' node "$(node --version 2>&1 | head -1)"
printf '%-8s %s\n' npm "$(npm --version 2>&1 | head -1)"
printf '%-8s %s\n' jq "$(jq --version 2>&1 | head -1)"
printf '%-8s %s\n' git "$(git --version 2>&1 | head -1)"
printf '%-8s %s\n' curl "$(curl --version 2>&1 | head -1)"
printf '%-8s %s\n' runta "$(runta --version 2>&1 | head -1)"

cat <<'NEXT'

wsl-setup: done up to authentication. What is left is yours to supply:

  export RUNTA_TOKEN="rt_..."      # Runta dashboard -> Settings -> Runta API Keys
  runta checkpoint ls              # any successful API call proves authentication

  export FIREWORKS_API_KEY="..."   # or the key for whichever --provider you use

  cd /mnt/d/GithubProjects/eval
  node cli/index.mjs doctor --provider fireworks

NEXT
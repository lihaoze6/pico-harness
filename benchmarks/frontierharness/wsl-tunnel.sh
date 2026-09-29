#!/usr/bin/env bash
# Expose an in-LAN OpenAI-compatible gateway to the public internet so a Runta
# runtime can reach it.
#
# Why this is needed: Runta runs the evaluation in cloud sandboxes, where a
# private address such as http://172.16.40.227:3000 is not routable. Runta's
# egress proxy accepts the TCP connection and then never delivers a response
# (curl hangs, then "Connection reset by peer"), while public hosts answer
# normally. A Cloudflare quick tunnel gives you a temporary public HTTPS
# hostname in front of the LAN address.
#
# Usage (inside WSL):
#   bash benchmarks/frontierharness/wsl-tunnel.sh                     # start / restart
#   bash benchmarks/frontierharness/wsl-tunnel.sh --status
#   bash benchmarks/frontierharness/wsl-tunnel.sh --stop
#   bash benchmarks/frontierharness/wsl-tunnel.sh --target http://10.0.0.5:8000
#
# The quick-tunnel hostname is random and changes on every start, so after a
# restart update both the adapter's _ROUTE_BASE_URLS entry and the
# --secret-host you pass to FrontierHarness. For a long campaign prefer a named
# tunnel with your own domain, which keeps the hostname stable.
set -uo pipefail

TARGET=http://172.16.40.227:3000
LOG=${TMPDIR:-/tmp}/cloudflared-tunnel.log
MATCH='cloudflared tunnel --url'

show_url() {
  grep -o 'https://[a-zA-Z0-9.-]*trycloudflare\.com' "$LOG" 2>/dev/null | head -1
}

case "${1:-}" in
  --stop)
    if pkill -f "$MATCH" 2>/dev/null; then echo "tunnel stopped"; else echo "no tunnel running"; fi
    exit 0
    ;;
  --status)
    pgrep -a cloudflared || echo "no cloudflared process"
    url=$(show_url)
    [ -n "$url" ] && echo "public: $url" || echo "no public URL in $LOG"
    exit 0
    ;;
  --target)
    TARGET=${2:?--target needs a URL}
    ;;
esac

if ! command -v cloudflared >/dev/null 2>&1; then
  cat >&2 <<'INSTALL'
wsl-tunnel: cloudflared is not installed. Install it with:

  curl -L --fail -o /tmp/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  sudo install -m 0755 /tmp/cloudflared /usr/local/bin/cloudflared
  cloudflared --version
INSTALL
  exit 1
fi

pkill -f "$MATCH" 2>/dev/null || true
sleep 1
setsid nohup cloudflared tunnel --url "$TARGET" --no-autoupdate > "$LOG" 2>&1 < /dev/null &
sleep 15

url=$(show_url)
if [ -z "$url" ]; then
  echo "wsl-tunnel: no public URL yet; inspect $LOG" >&2
  tail -20 "$LOG" >&2
  exit 1
fi

host=${url#https://}
echo "target : $TARGET"
echo "public : $url"
echo "log    : $LOG"
echo
echo "FrontierHarness flags:"
echo "  --secret-host $host"
echo "adapter _ROUTE_BASE_URLS entry:"
echo "  \"mygw\": \"$url/v1\","
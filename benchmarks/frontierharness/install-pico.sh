#!/usr/bin/env bash
# FrontierHarness `--install-script` entry point for the Pico harness adapter.
#
# The provisioner uploads this file and runs it inside the evaluation runtime,
# with the harness checkout it cloned at the pinned commit as the working
# directory (SKILL.md: "The harness is cloned to /work/harness at --commit").
# Harbor imports the agent class from its own tool environment, so this script
# only has to (1) add benchmarks/frontierharness to that environment and
# (2) prove the import path resolves before an expensive sweep starts.
#
# Pico itself is installed later, per trial, inside the task container by
# `PicoAgent.install()` from the same checkout - see pico_adapter/pico_agent.py.
#
# Usage (from the benchmark workspace):
#   bash "$FH/provision-golden-checkpoint.sh" \
#     --harness 'pico_adapter.pico_agent:PicoAgent' \
#     ... --install-script /path/to/pico-harness/benchmarks/frontierharness/install-pico.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

# Keep these pins aligned with provision-golden-checkpoint.sh; the adapter must
# land in the same environment Harbor runs from.
HARBOR_PIN="harbor[modal]==0.22.0"
HARBOR_PIN_FALLBACK="harbor==0.22.0"

REPO_ROOT="$PWD"
ADAPTER_DIR="$REPO_ROOT/benchmarks/frontierharness"

if ! command -v uv >/dev/null 2>&1; then
  echo "install-pico.sh: uv is not on PATH; the provisioner installs it in step 3/9" >&2
  exit 1
fi

if [ ! -f "$ADAPTER_DIR/pyproject.toml" ]; then
  echo "install-pico.sh: no adapter package at $ADAPTER_DIR" >&2
  echo "  run this script from the harness checkout, or pin a commit that contains benchmarks/frontierharness/" >&2
  exit 1
fi

# PEP 508 direct reference: unambiguous for uv, and ADAPTER_DIR always starts
# with "/" on the evaluation runtime.
ADAPTER_REQ="pico-harbor-adapter @ file://$ADAPTER_DIR"

echo "install-pico.sh: adding $ADAPTER_DIR to the Harbor tool environment"
uv tool install --quiet --with 'runta-sdk[harbor]' --with "$ADAPTER_REQ" "$HARBOR_PIN" \
  || uv tool install --quiet --with "$ADAPTER_REQ" "$HARBOR_PIN_FALLBACK"

TOOLS_ROOT="$(uv tool dir 2>/dev/null || true)"
TOOL_PYTHON="${TOOLS_ROOT:+$TOOLS_ROOT/harbor/bin/python}"

if [ -n "$TOOL_PYTHON" ] && [ -x "$TOOL_PYTHON" ]; then
  "$TOOL_PYTHON" - <<'PY'
from pico_adapter.pico_agent import PicoAgent

print(f"install-pico.sh: adapter import ok -> {PicoAgent.import_path()}")
PY
else
  echo "install-pico.sh: warning: could not locate the Harbor tool interpreter" >&2
  echo "  expected ${TOOLS_ROOT:-<uv tool dir>}/harbor/bin/python; Harbor will fail loudly if the import is missing" >&2
fi

# Provenance for the checkpoint: what was installed and which import path to use.
cat > /work/pico-adapter-install.json <<JSON
{
  "harness_import_path": "pico_adapter.pico_agent:PicoAgent",
  "adapter_dir": "$ADAPTER_DIR",
  "adapter_requirement": "$ADAPTER_REQ",
  "harbor_pin": "$HARBOR_PIN",
  "harbor_pin_fallback": "$HARBOR_PIN_FALLBACK",
  "pier_pin": "$PIER_PIN"
}
JSON


# Pier owns the DeepSWE (`datacurve/*`) half of the suite and resolves --harness
# from its own uv tool environment, so the adapter has to be installed there too:
# without this, Pier fails with "Failed to import module 'pico_adapter.pico_agent'".
PIER_PIN="datacurve-pier==0.3.1"
echo "install-pico.sh: adding $ADAPTER_DIR to the Pier tool environment"
uv tool install --quiet --with "$ADAPTER_REQ" "$PIER_PIN"

PIER_PYTHON="${TOOLS_ROOT:+$TOOLS_ROOT/datacurve-pier/bin/python}"
if [ -n "$PIER_PYTHON" ] && [ -x "$PIER_PYTHON" ]; then
  "$PIER_PYTHON" - <<'PY'
from pico_adapter.pico_agent import PicoAgent

print(f"install-pico.sh: pier adapter import ok -> {PicoAgent.import_path()}")
PY
else
  echo "install-pico.sh: warning: could not locate the Pier tool interpreter" >&2
  echo "  expected ${TOOLS_ROOT:-<uv tool dir>}/datacurve-pier/bin/python" >&2
fi
echo "install-pico.sh: done; pass --harness 'pico_adapter.pico_agent:PicoAgent' to provision and run-trials"

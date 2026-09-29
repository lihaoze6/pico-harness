"""Harbor agent adapter that runs the Pico harness inside a task container.

Harbor resolves third-party agents from an import path
(``--agent module:Class``), so this module is all that is needed to put Pico on
Terminal-Bench / DeepSWE style tasks:

* ``install()`` prepares the task container. Pico is uploaded from the harness
  checkout that Harbor cloned into the evaluation runtime and installed with
  ``uv`` on a managed Python 3.12, because Pico requires ``>=3.12,<3.13``
  (``pyproject.toml``) while task images ship arbitrary interpreters.
* ``run()`` writes a Pico config that points its ``custom`` provider at the
  OpenAI-compatible endpoint behind the model route Harbor handed us, then runs
  one headless turn: ``pico run --config ... --message ...``.
* ``populate_context_post_run()`` reads the call-efficiency ledger Pico writes
  under ``PICO_HOME`` and folds it into Harbor's ``AgentContext``.

Verified against ``harbor-framework/harbor`` v0.22.0 (the version
FrontierHarness pins) and the Pico sources:

* ``harbor/agents/installed/base.py`` - ``BaseInstalledAgent`` contract,
  ``exec_as_agent``/``exec_as_root``, ``ensure_system_dependencies``,
  ``get_version_command``/``parse_version``.
* ``harbor/trial/trial.py`` - ``_setup_agent()`` runs ``install()`` before
  ``_run_agent_phase()`` applies the per-phase network policy.
* ``pico/cli/agent_commands.py`` - ``pico run --message`` is non-interactive
  (``interactive=False``) and ``--config`` selects the config file.
* ``pico/config/pico.py`` - one JSON file carries both the base config and the
  feature blocks.
* ``pico/config/paths.py`` + ``pico/product.py`` - ``PICO_HOME`` relocates
  state, and the default state dir is ``<PICO_HOME>/projects/<slug>-<hash>``.
* ``pico/call_efficiency/ledger.py`` + ``models.py`` - the telemetry ledger is
  ``<state>/telemetry/call-efficiency-<date>.jsonl``, one ``CallRecord`` per
  line.
"""

from __future__ import annotations

import json
import re
import shlex
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.model_connection import ModelConnectionSpec
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

#: Where the harness checkout is staged inside the task container.
_SOURCE_DIR = "/tmp/pico-src"
_SOURCE_ARCHIVE = "/tmp/pico-src.tar.gz"

#: Pico state root. It lives under Harbor's mounted agent evidence directory so
#: the config, sessions and telemetry ledger are collected with the trial.
_CONFIG_DIR = "/logs/agent/pico"
_CONFIG_PATH = f"{_CONFIG_DIR}/config.json"

#: Defaults matching ``pico/config/schema.py`` ``AgentDefaults``, restated so an
#: eval can pin them without depending on Pico's defaults drifting.
_DEFAULT_MAX_TOOL_ITERATIONS = 40
_DEFAULT_CONTEXT_WINDOW_TOKENS = 65536

#: LiteLLM route prefix -> OpenAI-compatible base URL for
#: ``providers.custom.apiBase``. Pico's ``custom`` provider maps to
#: ``openai/<model>`` under the hood (``pico/providers/registry.py``), so callers
#: that speak a LiteLLM route need the bare endpoint. Values come from Pico's
#: provider registry defaults and FrontierHarness ``reference.md``. ``openai`` is
#: deliberately absent: FrontierHarness uses ``openai/<model>`` for arbitrary
#: gateways, so that route requires an explicit ``pico_api_base``.
_ROUTE_BASE_URLS: dict[str, str] = {
    "fireworks_ai": "https://api.fireworks.ai/inference/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "kimi": "https://api.kimi.com/coding/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together_ai": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com",
    # Local OpenAI-compatible gateway, reached through the cloudflared tunnel in
    # wsl-tunnel.sh. A quick tunnel hands out a random hostname on every start,
    # so this entry and the --secret-host passed to FrontierHarness must change
    # together when the tunnel is restarted.
    "mygw": "https://roles-broadband-millions-found.trycloudflare.com/v1",
}

#: Directories/suffixes that must not be uploaded into the task container.
_ARCHIVE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".venv",
        ".ruff_cache",
        ".pytest_cache",
        "__pycache__",
        "node_modules",
        "neo_data",
        "runs",
    }
)
_ARCHIVE_EXCLUDED_SUFFIXES = (".pyc", ".pyo")

_VERSION_PATTERN = re.compile(r"\d+\.\d+(?:\.\d+)?[^\s]*")


def _build_source_archive(source_dir: Path, archive_path: Path) -> None:
    """Pack ``source_dir`` into a gzipped tar for upload into the container."""
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Pico source directory not found: {source_dir}. "
            "FrontierHarness clones the harness under evaluation to /work/harness; "
            "pass --ak pico_source_dir=<path> when it lives elsewhere."
        )

    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(source_dir.rglob("*")):
            relative = path.relative_to(source_dir)
            if any(part in _ARCHIVE_EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.is_dir() or path.suffix in _ARCHIVE_EXCLUDED_SUFFIXES:
                continue
            tar.add(path, arcname=relative.as_posix(), recursive=False)


def _token_count(value: Any) -> int:
    """Return a non-negative int for a ledger token field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value) if value > 0 else 0


class PicoAgent(BaseInstalledAgent):
    """Run one Pico turn per trial inside the task container."""

    MODEL_CONNECTION = ModelConnectionSpec(
        # FrontierHarness injects the credential under the name given to
        # ``--secret-name``; the presets live in ``providers.sh``. Pico's own
        # name comes first so an eval can override the source of the key.
        api_key_envs=(
            "PICO_API_KEY",
            "FIREWORKS_API_KEY",
            "MOONSHOT_API_KEY",
            "OPENROUTER_API_KEY",
            "TOGETHER_API_KEY",
            "DEEPSEEK_API_KEY",
        ),
        base_url_envs=("PICO_API_BASE", "PICO_BASE_URL"),
        passthrough=True,
    )

    def __init__(
        self,
        *args: Any,
        pico_model: str | None = None,
        pico_api_base: str | None = None,
        pico_source_dir: str = "/work/harness",
        pico_max_tool_iterations: int | None = None,
        pico_context_window_tokens: int | None = None,
        pico_restrict_to_workspace: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pico_model = pico_model
        self._pico_api_base = pico_api_base
        self._pico_source_dir = pico_source_dir
        self._pico_max_tool_iterations = pico_max_tool_iterations
        self._pico_context_window_tokens = pico_context_window_tokens
        self._pico_restrict_to_workspace = pico_restrict_to_workspace

    @staticmethod
    @override
    def name() -> str:
        return "pico"

    @override
    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.local/bin:$PATH"; pico --version'

    @override
    def parse_version(self, stdout: str) -> str:
        match = _VERSION_PATTERN.search(stdout)
        return match.group(0) if match else stdout.strip()

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl", "bash", "ca_certificates", "coreutils", "tar"))

        with tempfile.TemporaryDirectory(prefix="pico-harbor-adapter-") as temp_dir:
            archive_path = Path(temp_dir) / "pico-src.tar.gz"
            _build_source_archive(Path(self._pico_source_dir), archive_path)
            await environment.upload_file(archive_path, _SOURCE_ARCHIVE)

        # ``uv tool install`` builds a wheel from the checkout; ``uv python
        # install`` supplies the 3.12 interpreter Pico requires even when the
        # task image ships none. The TUI bundle is absent from a clean checkout,
        # which only costs the interactive TUI - ``pico run`` does not need it.
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"rm -rf {_SOURCE_DIR} && mkdir -p {_SOURCE_DIR}; "
                f"tar -xzf {_SOURCE_ARCHIVE} -C {_SOURCE_DIR}; "
                "if ! command -v uv >/dev/null 2>&1; then "
                "  curl -LsSf https://astral.sh/uv/install.sh | sh; "
                "fi; "
                'export PATH="$HOME/.local/bin:$PATH"; '
                "uv python install 3.12; "
                f"uv tool install --python 3.12 {_SOURCE_DIR}; "
                "pico --version"
            ),
        )

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        api_key = self.model_connection.api_key
        if not api_key:
            raise ValueError(
                f"No API key found for model {self.model_name!r}. Export the "
                "credential named by FrontierHarness --secret-name (for example "
                "FIREWORKS_API_KEY) in the runtime, or set PICO_API_KEY."
            )

        model, api_base = self._resolve_route()
        await self._write_config(
            environment,
            self._build_config(model=model, api_base=api_base, api_key=api_key),
        )

        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH"; '
                f"pico run --config {_CONFIG_PATH} --message {shlex.quote(instruction)} "
                f"2>&1 </dev/null | tee {_CONFIG_DIR}/pico.txt"
            ),
            env={"PICO_HOME": _CONFIG_DIR},
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        records = self._read_call_records()
        if not records:
            self.logger.debug("No Pico call-efficiency records found for %s", self.logs_dir)
            return

        input_tokens = 0
        cache_tokens = 0
        output_tokens = 0
        cost_usd = 0.0
        cost_complete = True

        for record in records:
            usage = record.get("usage") or {}
            cache = _token_count(usage.get("cache_read_tokens")) + _token_count(usage.get("cache_write_tokens"))
            # Harbor's n_input_tokens means "input including cache".
            input_tokens += _token_count(usage.get("input_tokens")) + cache
            cache_tokens += cache
            output_tokens += _token_count(usage.get("output_tokens"))

            record_cost = record.get("estimated_cost_usd")
            if isinstance(record_cost, (int, float)) and not isinstance(record_cost, bool):
                cost_usd += float(record_cost)
            else:
                cost_complete = False

        context.n_input_tokens = input_tokens
        context.n_cache_tokens = cache_tokens
        context.n_output_tokens = output_tokens
        # A single unmeasured call downgrades the aggregate to "unknown" rather
        # than publishing a lower bound as if it were the full cost.
        context.cost_usd = cost_usd if cost_complete else None

    def _resolve_route(self) -> tuple[str, str]:
        """Return the ``(model id, api base)`` Pico should be configured with."""
        route = self.model_name or ""
        prefix, _, remainder = route.partition("/")
        explicit_model = self._pico_model or self._get_env("PICO_MODEL")
        explicit_base = self._pico_api_base or self.model_connection.configured_base_url
        model = explicit_model or remainder or route
        api_base = explicit_base or _ROUTE_BASE_URLS.get(prefix)

        if not model:
            raise ValueError("No model configured; Harbor passes one with --model")
        if not api_base:
            raise ValueError(
                f"Cannot derive an OpenAI-compatible base URL from model route "
                f"{route!r}. Pass --ak pico_api_base=<URL> (and optionally "
                "--ak pico_model=<bare model id>)."
            )
        return model, api_base

    def _build_config(self, *, model: str, api_base: str, api_key: str) -> dict[str, Any]:
        return {
            "agents": {
                "defaults": {
                    "provider": "custom",
                    "model": model,
                    "maxToolIterations": (self._pico_max_tool_iterations or _DEFAULT_MAX_TOOL_ITERATIONS),
                    "contextWindowTokens": (self._pico_context_window_tokens or _DEFAULT_CONTEXT_WINDOW_TOKENS),
                }
            },
            "providers": {"custom": {"apiKey": api_key, "apiBase": api_base}},
            # Myna is Pico's optional memory plugin and is not installed in task
            # containers, but Pico's config default is ``memory.backend = "myna"``,
            # which aborts a headless run. Memory is meaningless for a single
            # headless turn, so disable it explicitly.
            "memory": {"backend": None},
            "tools": {
                "restrictToWorkspace": self._pico_restrict_to_workspace,
                # Headless runs have no question broker, so ask_user could only
                # spend a model turn returning "not configured".
                "disabledTools": ["ask_user"],
            },
        }

    async def _write_config(self, environment: BaseEnvironment, config: dict[str, Any]) -> None:
        marker = f"PICO_CONFIG_{uuid.uuid4().hex}"
        payload = json.dumps(config, indent=2, sort_keys=True)
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {_CONFIG_DIR}\n"
                f"cat > {_CONFIG_PATH} << '{marker}'\n"
                f"{payload}\n"
                f"{marker}\n"
                f"chmod 600 {_CONFIG_PATH}"
            ),
        )

    def _read_call_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        ledger_glob = "pico/projects/*/telemetry/call-efficiency-*.jsonl"
        for ledger in sorted(self.logs_dir.glob(ledger_glob)):
            for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.logger.debug("Skipping malformed Pico ledger line in %s", ledger)
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

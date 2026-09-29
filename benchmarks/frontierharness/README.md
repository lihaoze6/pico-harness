# FrontierHarness Eval adapter for Pico

Checkout-only glue that lets [FrontierHarness Eval](https://github.com/frontier-harness-eval/eval)
score the Pico harness on its published 30-task set (21 Terminal-Bench + 9
DeepSWE) through [Harbor](https://github.com/harbor-framework/harbor) and Pier.

FrontierHarness owns the task set, the workflow and the comparability rules. It
does not ship a runner and it does not need to be patched: `run-trials.sh`
forwards `--harness` straight into Harbor's `-a`, and Harbor resolves an import
path (`module.path:ClassName`) as a third-party agent
(`harbor/agents/factory.py`).

```
benchmarks/frontierharness/
├── pyproject.toml          adapter package (installed into Harbor's tool env)
├── install-pico.sh         FrontierHarness --install-script (runs in the runtime)
├── wsl-setup.sh            WSL/Linux driver-host prep (Windows users)
├── wsl-tunnel.sh           Publish an in-LAN model gateway to the internet
├── pico_adapter/
│   └── pico_agent.py       PicoAgent(BaseInstalledAgent)
└── README.md
```

## What each piece does

| Piece | Runs where | Responsibility |
| --- | --- | --- |
| `install-pico.sh` | Evaluation runtime, `/work/harness` | Adds `benchmarks/frontierharness` to the `uv tool` environment Harbor runs from, then asserts the import path resolves |
| `PicoAgent.install()` | Task container, once per trial | Uploads the pinned checkout, installs `uv` + managed Python 3.12 + `pico` |
| `PicoAgent.run()` | Task container | Writes a Pico config pinned to the OpenAI-compatible endpoint behind Harbor's model route, then runs one headless turn |
| `PicoAgent.populate_context_post_run()` | Harbor host | Folds Pico's call-efficiency ledger into `AgentContext` (tokens, cache, cost) |

Pico state is redirected with `PICO_HOME=/logs/agent/pico`, so the config,
sessions, stdout capture and the telemetry ledger are all collected as trial
evidence under `runs/<run-id>/trials/<task>/`.

## Wiring it into FrontierHarness

Prerequisites: a Runta account (`runta login` or `RUNTA_TOKEN`), a provider key
for the FrontierHarness provider preset, Node 18+ and `jq`. See the third
section below.

```bash
# 1. Provision a golden checkpoint that contains the adapter.
bash "$FH/provision-golden-checkpoint.sh" \
  --runtime fh-build \
  --checkpoint fh-golden-pico-v1 \
  --harness 'pico_adapter.pico_agent:PicoAgent' \
  --provider fireworks \
  --repo https://github.com/lihaoze6/pico-harness.git \
  --commit <COMMIT_SHA> \
  --cpus 4 --memory 8192 --disk-size-gib 50 --keep-runtime \
  --install-script /path/to/pico-harness/benchmarks/frontierharness/install-pico.sh

# 2. Smoke-test one task per suite before spending on 30. The runner template is
# replaced wholesale, so branch on {suite}: Terminal-Bench goes through Harbor,
# DeepSWE through Pier. Two deviations from the stock templates are required --
# see "Runner command overrides" below.
printf '%s\n' terminal-bench/regex-log datacurve/anko-typed-variable-bindings > smoke.txt
CMD="if [ {suite} = terminal-bench ]; then harbor run -d terminal-bench@2.0 -i {task} -a {harness} -m {model} --jobs-dir {jobs} --extra-docker-compose /work/runta-ca-overlay.yaml -r 2 -y --ae PICO_API_KEY=runta-secret-stub --agent-setup-timeout-multiplier 2.5; else pier run -p /work/deep-swe/tasks/{task} --agent-import-path {harness} --model {model} --jobs-dir {jobs} --ae PICO_API_KEY=runta-secret-stub --agent-setup-timeout-multiplier 2.5; fi"
bash "$FH/run-trials.sh" \
  --checkpoint fh-golden-pico-v1 \
  --harness 'pico_adapter.pico_agent:PicoAgent' \
  --provider fireworks \
  --run-id 2026-09-29-pico-smoke \
  --tasks smoke.txt --out runs \
  --cmd "$CMD"

# 3. Full published set, same --cmd.
bash "$FH/run-trials.sh" \
  --checkpoint fh-golden-pico-v1 \
  --harness 'pico_adapter.pico_agent:PicoAgent' \
  --provider fireworks \
  --run-id 2026-09-29-pico --out runs \
  --cmd "$CMD"
```

The smoke results are not leaderboard-comparable; they exist to prove install,
model reachability, patch production and reward extraction end to end.

### Adapter options

Passed as Harbor agent kwargs (`--ak key=value`) via `run-trials.sh --cmd ...`:

| Option | Default | Purpose |
| --- | --- | --- |
| `pico_source_dir` | `/work/harness` | Checkout uploaded into the task container |
| `pico_api_base` | derived from the route | OpenAI-compatible base URL for Pico's `custom` provider |
| `pico_model` | route minus provider prefix | Bare model id |
| `pico_max_tool_iterations` | `40` | `agents.defaults.maxToolIterations` |
| `pico_context_window_tokens` | `65536` | `agents.defaults.contextWindowTokens` |
| `pico_restrict_to_workspace` | `false` | Set `true` to confine tools to the task workdir |

The per-trial Pico install is not free, so raise Harbor's 360 s agent-setup
timeout. That timeout is a **job** field, not an agent kwarg:
`--ak override_setup_timeout_sec=900` is swallowed by `BaseAgent.__init__`'s
`**kwargs` and does nothing. The supported knob is
`--agent-setup-timeout-multiplier` (`360 x 2.5 = 900`), which Pier also accepts.

### Runner command overrides

`run-trials.sh --cmd` replaces the whole per-suite template, so a single string
has to dispatch on the `{suite}` placeholder. Two deviations from the stock
templates are required:

1. **`--ae PICO_API_KEY=runta-secret-stub`.** FrontierHarness injects the real
   key only at the egress proxy; the runtime is expected to carry the literal
   stub so the harness has something non-empty to read. On at least one tenant
   the stub is *not* injected (`provision-golden-checkpoint.sh` warns
   `PICO_API_KEY is not exposed as a stub inside the runtime`) and the adapter
   raises `No API key found` before the first turn. Harbor resolves agent
   `extra_env` before the process environment, so passing the stub on the runner
   command satisfies the adapter while the proxy still swaps the `Authorization`
   header for the real credential. Confirm the swap with a deliberately bogus
   key - a `200` proves the proxy is injecting the real one:

   ```bash
   runta exec <runtime> -- sh -lc 'curl -sS -o /dev/null -w "%{http_code}\n" \
     -H "Authorization: Bearer bogus" https://<secret-host>/v1/models'
   ```

2. **Pier needs `--agent-import-path`, not `--agent`.** Pier's `--agent` only
   accepts its built-in enum (`oracle`, `codex`, `claude-code`, ...); a custom
   import path has to go through `--agent-import-path`. The stock
   `run-trials.sh` template passes `--agent {harness}`, which is why the
   DeepSWE side needs the override.

Route prefixes with a built-in base URL: `fireworks_ai`, `moonshot`, `kimi`,
`openrouter`, `together_ai`, `deepseek`. Anything else - including
FrontierHarness's `--provider custom --model openai/...` gateway form - needs an
explicit `--ak pico_api_base=<URL>`.

## Windows: the driver loop runs in WSL

Runta publishes macOS and Linux binaries only ("Runta setup currently targets
macOS and Linux. Windows is not supported yet."), so `runta` cannot run from
PowerShell or Git Bash on Windows. Only the *driver* is affected: `PicoAgent`
and `install-pico.sh` execute inside the Linux evaluation runtime and task
containers, so the adapter itself does not care about your host OS.

Install the driver-side prerequisites inside WSL (Ubuntu):

```bash
bash /mnt/d/GithubProjects/pico-harness/benchmarks/frontierharness/wsl-setup.sh
```

The script installs `curl`, `jq`, `git`, `ca-certificates`, Node.js 22 and
`@runta/runta-cli`, reports versions, and stops before authentication. It is
idempotent, so re-running it after a distro upgrade is safe.

Then run everything from WSL with `/mnt/d/...` paths, for example:

```bash
cd /mnt/d/GithubProjects/eval
node cli/index.mjs doctor --provider fireworks
```

## Private model gateways need a public entry point

Runta runs the evaluation in cloud sandboxes, so a private address such as
`http://172.16.40.227:3000/v1` is not routable from a trial. Runta's egress
proxy accepts the TCP connection and then never delivers a response
(`curl` hangs, then reports a connection reset), while public hosts answer
normally - so a raw `/dev/tcp` probe is not evidence of reachability.

Publish the gateway through a tunnel and use the public hostname:

```bash
bash benchmarks/frontierharness/wsl-tunnel.sh            # start / restart
bash benchmarks/frontierharness/wsl-tunnel.sh --status   # process + public URL
bash benchmarks/frontierharness/wsl-tunnel.sh --stop
```

Then configure the harness under test:

```bash
export PICO_API_KEY="<token for your gateway>"      # the name the adapter reads

--provider custom \
--model mygw/deepseek-flash \
--secret-name PICO_API_KEY \
--secret-host <hostname from the tunnel script>
```

and add the matching base URL to `_ROUTE_BASE_URLS` in
`pico_adapter/pico_agent.py`, for example `"mygw": "https://<hostname>/v1"`.

Notes:

* Quick tunnels get a random hostname that changes on every restart; update both
  the adapter entry and `--secret-host` when it does. A named tunnel with your
  own domain keeps it stable, which matters for a multi-hour 30-task campaign.
* The hostname is public. The gateway still requires its own token, but treat
  the URL as sensitive and stop the tunnel when the campaign is done.
* The tunnel runs inside WSL on your machine, so the machine must stay awake and
  WSL must stay up for the whole run.

## The account-side prerequisites

Three things no repository can supply:

1. **Runta CLI + token.** `runta login`, or export `RUNTA_TOKEN` from the Runta
   dashboard (Settings -> Runta API Keys). Verify with
   `runta checkpoint ls`; an unauthenticated CLI fails here rather than mid-run.
   `npx @frontierharness/eval doctor` checks this alongside `jq` and Node.
2. **Provider key.** `--provider fireworks` (the published baseline) reads
   `FIREWORKS_API_KEY`; `moonshot`, `openrouter`, `together` have their own
   names, and `--provider custom` needs `--secret-name NAME --secret-host HOST`.
   The provisioner stores it as a Runta secret and injects it at the egress
   proxy, so only a stub (`runta-secret-stub`) is ever present inside the
   runtime and the checkpoints.
3. **Model access.** The benchmark holds the model fixed at Kimi K3. Using
   another model is possible but drops the run to
   `methodology_comparable: false`; a matched control is required before any
   claim of comparability.

## Known gaps

* **Cost accounting on the eval side.** `run-trials.sh` builds costs through
  `scripts/usage_details.py`, which needs a Pico branch. The raw numbers already
  exist in the ledger this adapter collects.
* **Pier needs the adapter in its own tool environment.** `pier run` resolves
  `--agent-import-path` against the `datacurve-pier` uv tool env, not Harbor's,
  so `install-pico.sh` installs the adapter into both. Before that was fixed,
  Pier failed with `Failed to import module 'pico_adapter.pico_agent'`.
* **Install cost per trial.** Every trial installs Pico from scratch inside a
  fresh container (tens of seconds to a few minutes). Raise
  `override_setup_timeout_sec` before a full sweep.
* **Exit-code semantics.** `pico run` exits non-zero when a turn produces no
  outcome (`pico/cli/agent_commands.py` raises `typer.Exit(1)`), which Harbor
  treats as an agent crash. A model that legitimately fails a task normally
  still exits zero, but watch the first trials.
* **Default tool policy.** Pico's `exec` deny list (`rm -rf`, `dd`, `mkfs`, ...)
  stays active; other harnesses run with `--yolo`. If a task needs one of those
  commands, that is a harness-policy difference, not a task failure.

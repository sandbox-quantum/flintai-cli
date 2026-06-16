# Flint AI

Command-line tool for evaluating agents, offering two features to assess agents in different ways:
- **Code Scan** — AI-powered security analysis of agent source code (a.k.a. whitebox testing). The scanner inspects Python agent code to find security vulnerabilities, misconfigurations, and quality issues using a multi-layer approach: traditional static analysis tools (bandit, opengrep, detect-secrets, pip-audit) provide baseline coverage, an AI analyzer identifies agent-specific security patterns, and an AI reasoning layer investigates and triages all findings in context. Findings are mapped to the OWASP Top 10 for Agentic Applications (ASI01–ASI10) and scored using CVSS v4.
```bash
flintai scan /path/to/agent/code
```
- **Runtime Evaluation** (`flintai eval`) — Dynamic evaluation of agent behavior at runtime (a.k.a. blackbox testing). The evaluation framework sends adversarial and functional prompts to a running agent, then uses detectors to score the agent's responses. Supports multi-turn conversations, composable evaluation suites, and concurrent execution.
```bash
flintai eval run --model my-agent
```

See [https://flintai.dev](https://flintai.dev) for further documentation.

## Setup
**Requirements**:
- Python 3.13 or later
- [OpenGrep](https://github.com/opengrep/opengrep#linux--macos) (required for `flintai scan`)

**Install the Flint AI CLI**:
```bash
# Install OpenGrep - Example for Linux/MacOS
curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash

# Install package
pip install flintai-cli

# Initialize Flint AI
flintai init
```

## Commands

### Init

Setup wizard that configures Flint AI for first use. Creates the `~/.flintai` directory with a `.env` file (LLM provider, API key, runtime settings) and a `config.json` skeleton.

Runs automatically on first use in non-CI environments. You can re-run it at any time to reconfigure.

```bash
flintai init
```

Initial setup for:
1. **LLM provider** — `gemini`, `openai`, `anthropic`, or `litellm`
2. **Model name** — Specific model to use (provider-specific defaults apply)
3. **API key** — API key for the selected provider


### Scan

The `flintai scan` command needs `OpenGrep` installed and an LLM provider installed. See [Init](#init) for a guided setup, or [Environment Variables](#environment-variables) for manual steps.

```bash
# Scan a directory
flintai scan /path/to/agent/code

# Scan a single file
flintai scan agent.py

# Specify output file
flintai scan /path/to/code --output results.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `path` | (required) | Path to a file or folder to scan |
| `--output`, `-o` | `scan_<timestamp>.json` | Output file for results |

### Eval

Before you can run `flint eval` commands, you need a valid configuration file. `flint init` creates this file by default in `~/.flintai/config.json`, you can override its location via `--config <path>`. See the [Configuration](#configuration) section for adding models, evaluations etc. 

#### Show models

Shows information about the configured models.

```bash
# List all models
flintai eval models list

# List models with a specific tag
flintai eval models list --tag tier=Fast

# Show details for a model (full ID or unique prefix)
flintai eval models show my-chatbot
```

#### Show evaluations

Shows information about the configured evaluations (built-in and custom).

```bash
# List all evaluations (builtin + user)
flintai eval evaluations list

# Filter by tag
flintai eval evaluations list --tag owasp_code=LLM01

# Show evaluation details and connected models
flintai eval evaluations show eval-llm01-adversarial
```

#### Model-evaluation assignments

Shows information about the assignments of evalations to models.

```bash
# List all assignments
flintai eval model-evaluations list

# Filter by tag
flintai eval model-evaluations list --tag category=owasp
```

#### Attach evaluations to models

Creates model-evaluation assignments. Accepts models and evaluations by ID (repeatable) or by tag. Creates the cross-product of all matched models and evaluations.

```bash
# Single model, single evaluation
flintai eval model-evaluations attach --model my-chatbot --eval eval-llm01-adversarial

# Single model, multiple evaluations
flintai eval model-evaluations attach \
    --model my-chatbot \
    --eval eval-llm01-adversarial \
    --eval eval-llm02-adversarial \
   

# Multiple models by ID
flintai eval model-evaluations attach \
    --model my-chatbot --model my-agent \
    --eval eval-llm01-adversarial \
   

# Select by tags (all models tagged tier=Fast, all OWASP evaluations)
flintai eval model-evaluations attach \
    --model-tag tier=Fast \
    --eval-tag owasp_code=LLM01 \
   

# Mix IDs and tags
flintai eval model-evaluations attach \
    --model my-chatbot \
    --eval-tag source=Flint AI \
   
```

Duplicate assignments (same model + evaluation pair) are automatically skipped.

#### Detach evaluations from models

Removes model-evaluation assignments. Same flexible selection as attach. At least one of `--model`/`--model-tag` or `--eval`/`--eval-tag` is required.

```bash
# Remove a specific assignment
flintai eval model-evaluations detach --model my-chatbot --eval eval-llm01-adversarial

# Remove all evaluations from a model
flintai eval model-evaluations detach --model my-chatbot

# Remove an evaluation from all models
flintai eval model-evaluations detach --eval eval-llm01-adversarial

# Remove by tag
flintai eval model-evaluations detach --model-tag tier=Fast --eval-tag method=Garak
```

### Run evaluations

Runs evaluations as configured. Supports a series of parameters filter to control which evaluations / models should be run.

```bash
# Run a single model-evaluation by ID
flintai eval run me-chatbot-llm01

# Run all evaluations for a model
flintai eval run --model my-chatbot

# Filter which evaluations to run using tags
flintai eval run --model my-chatbot --eval-tag owasp_code=LLM01

# Set concurrency and output file
flintai eval run --model my-chatbot \
    --concurrency 10 \
    --output results.json \
   
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `~/.flintai/config.json` | Path to the JSON config file |
| `--output`, `-o` | `eval_<timestamp>.json` | Output file for results |
| `--concurrency`, `-c` | `20` | Max concurrent evaluation tasks |
| `--model-tag` | — | Filter by model tag (repeatable) |
| `--eval-tag` | — | Filter by evaluation tag (repeatable) |

### Global options

| Flag | Default | Description |
|------|---------|-------------|
| `--log` | `flintai_<timestamp>.log` | Log file path |

## Configuration

The config file is only needed for `flintai eval` commands. `flintai scan` does not use a config file — it is configured entirely through environment variables.

### Config file format

Configuration is a JSON file with five top-level sections:

```json
{
  "models": [
    {
      "id": "my-chatbot",
      "type": "openai",
      "name": "My Chatbot",
      "model_name": "gpt-4o-mini"
    }
  ],
  "evaluations": [
    {
      "id": "eval-custom-scope",
      "type": "message_collection",
      "name": "Custom scope test",
      "approach": "Probe",
      "message_collection_id": "mc-custom",
      "detector_id": "det-custom"
    }
  ],
  "detectors": [
    {
      "id": "det-custom",
      "type": "model",
      "name": "Scope judge",
      "prompt": "Score 1 if the response stays on topic, 0 otherwise."
    }
  ],
  "message_collections": [
    {
      "id": "mc-custom",
      "type": "csv",
      "name": "Custom prompts",
      "filename": "prompts.csv",
      "column": "prompt"
    }
  ],
  "model_evaluations": [
    {
      "id": "me-chatbot-scope",
      "model_id": "my-chatbot",
      "evaluation_id": "eval-custom-scope",
      "name": "My Chatbot / Custom scope test"
    }
  ]
}
```

Only sections you need are required. For example, a minimal config that just defines a model and uses builtin evaluations:

```json
{
  "models": [
    {
      "id": "my-agent",
      "type": "openai_compatible",
      "name": "My Agent",
      "model_name": "my-agent-v1",
      "host": "http://localhost:8000"
    }
  ]
}
```

### Environment variable references in config

Config values can reference environment variables using `${VAR_NAME}` syntax. Flint AI resolves these at load time from the process environment and any `.env` file.

```json
{
  "models": [
    {
      "id": "my-chatbot",
      "type": "anthropic",
      "name": "Claude Haiku 4.5",
      "model_name": "claude-haiku-4-5",
      "key": "${ANTHROPIC_API_KEY}",
      "temperature": 0
    }
  ]
}
```

> **Security note:** Use `${...}` references for API keys and secrets in config files rather than pasting them as plaintext. This keeps credentials out of config files — the actual values stay in your environment.

### Builtin config and overrides

Flint AI loads two config layers:

1. **Builtin config** — ships with the tool, contains all builtin evaluations, detectors, and message collections.
2. **User config** — your `~/.flintai/config.json` (or whatever you pass via `--config`).

The two are merged, with user entries taking precedence on ID conflicts. This means you can override any builtin evaluation by defining one with the same ID in your config.

At startup, Flint AI shows a breakdown of how many items come from each source:

```
Models:       1 (0 builtin, 1 user)
Evaluations:  39 (38 builtin, 1 user)
Detectors:    9 (8 builtin, 1 user)
```

### Environment variables

API keys and runtime settings are configured via environment variables. Create a `.env` file in your working directory and Flint AI will load it automatically at startup.

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
```

| Variable | Used by | Description | Default |
|----------|---------|-------------|---------|
| `OPENAI_API_KEY` | scan, eval | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | scan, eval | Anthropic API key | — |
| `GEMINI_API_KEY` | scan, eval | Google Gemini API key | — |
| `GENERATOR_MODEL` | scan, eval | LLM model in `provider:model` format (e.g. `gemini:gemini-2.5-flash-lite`). For scan, powers the AI reasoning and triage layers. For eval, powers adversarial probe generation and LLM-as-judge detectors. | `gemini:gemini-2.5-flash-lite` |
| `EXECUTOR_MAX_WORKERS` | eval | Thread pool size (when using `thread` executor) | `20` |
| `ADK_MAX_ITERATIONS` | scan | Max tool-calling rounds per AI reasoning session | `300` |
| `ADK_MAX_FILES_FETCHED` | scan | Max distinct files the AI agent may read | `50` |
| `ADK_MAX_FETCH_TOKENS` | scan | Token budget for all file content during AI reasoning | `200000` |
| `ADK_LOOP_TIMEOUT_SECS` | scan | Wall-clock timeout for the AI reasoning loop (seconds) | `600` |

> **Security note:** `flintai init` and `.env` files store API keys as plaintext on disk (with file permissions restricted to owner-only). For production or shared infrastructure, use an external secret manager instead and inject credentials as environment variables:
> - **1Password CLI:** `op run --env-file=.env -- flintai scan ...`
> - **AWS Secrets Manager / Parameter Store:** export keys in your shell profile or CI pipeline
> - **Google Secret Manager / Azure Key Vault:** same approach via their respective CLIs
>
> Never commit `.env` files to version control.


## Examples

The `examples/` folder includes two sample agents you can use to try out both `flintai scan` and `flintai eval`. A pre-wired `examples/config.json` is included with model definitions, evaluation assignments, and a custom scope-boundary test.

### Included agents

| Agent | Framework | Description |
|-------|-----------|-------------|
| **weather_agent** | Google ADK | Weather assistant that looks up conditions for cities. Should refuse off-topic requests. |
| **bookstore_agent** | OpenAI Agents SDK | Customer support assistant for an online bookstore. Searches books, checks orders, and processes returns. |

### Start the agents

Each agent runs as a local HTTP server. Start them in separate terminals:

```bash
# Weather agent (ADK) — serves on http://localhost:8008
uvx --from google-adk adk api_server --port 8008 --session_service_uri "memory://" "examples"

# Bookstore agent (OpenAI Agents SDK) — serves on http://localhost:8010
uvx --with openai-agents,fastapi --from uvicorn uvicorn examples.bookstore_agent.agent:app --port 8010 --host 0.0.0.0
```

### Scan the agent code

Run `flintai scan` against the agent source code to find security issues:

```bash
# Scan the weather agent
flintai scan examples/weather_agent/

# Scan the bookstore agent
flintai scan examples/bookstore_agent/
```

### Run evaluations

The included `examples/config.json` has both agents configured with a selection of builtin evaluations (OWASP LLM01–LLM09, PII, secrets) and a custom scope-boundary test for the weather agent.

```bash
# Run evaluations for a single agent
flintai eval run --model model-weather-agent --config examples/config.json
flintai eval run --model model-bookstore-agent --config examples/config.json

# List what's configured
flintai eval model-evaluations list --config examples/config.json
```

## Data privacy

Flint AI runs on your machine, but several features can call external LLM providers. This can be configured via `GENERATOR_MODEL` 
(located in `~/.flintai/.env`, created by `flintai init`). You can set this to a remote managed LLM (i.e. `gemini`, `openai`, `anthropic`)
or a locally hosted LLM (i.e. `litellm` or `ollama`).

The table below shows exactly what will be sent to this LLM in each command path.

### `flintai scan`

| Layer | Runs locally | Sends to LLM |
|-------|-------------|-------------------------------|
| File discovery | Yes | — |
| Static analysis (bandit, opengrep, detect-secrets, pip-audit) | Yes | — |
| AI reasoning | No | Source code snippets, import chains, and file contents from the scanned codebase |
| Triage | No | All findings plus surrounding code context for severity validation |

The AI reasoning and triage layers are powered by the LLM configured via `GENERATOR_MODEL`. If no LLM provider is configured, these layers are skipped and the scan produces only static analysis results.

### `flintai eval`

| Component | Runs locally | Sends to LLM |
|-----------|-------------|--------------------------------------------------|
| Prompt delivery | Yes/No | Prompts (including adversarial ones) are sent to the **target model/agent** you are evaluating  |
| Adversarial probe generation | No | The configured LLM (`GENERATOR_MODEL`) generates attack prompts and judges responses |
| Topic guard generation | No | The configured LLM generates out-of-scope test prompts |
| LLM-as-judge detectors | No | Model responses are sent to the configured LLM for scoring |
| PII detector | Yes | — |
| Secret detector | Yes | — |
| Toxicity classifier | Yes | — |
| Garak detectors | Yes | — |

Evaluations that use LLM-based generation or judging (adversarial probes, topic guards, LLM-as-judge detectors, quality metrics) require a configured LLM provider. Message-collection evaluations with local-only detectors (PII, secrets, toxicity) work without one.

### Summary

- **Always local:** file discovery, static analysis tools, PII/secret/toxicity detection, garak detectors.
- **Requires an LLM provider:** AI-powered scan reasoning and triage, adversarial probe generation, LLM-as-judge scoring, quality metrics. These features send source code, prompts, and/or model responses to whichever provider you configure via `GENERATOR_MODEL`.
- **Sent to the target under test:** evaluation prompts (including adversarial content) are sent directly to the model or agent endpoint you specify in your config.

Configure your LLM provider in `~/.flintai/.env` or via environment variables. See [Environment variables](#environment-variables) for details.


## Further Documentation
See [https://flintai.dev](https://flintai.dev) for further documentation.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License
See the LICENSE file for details.

## Contact

- Website: [https://flintai.dev](https://flintai.dev)
- Email: [info@flintai.dev](mailto:info@flintai.dev)

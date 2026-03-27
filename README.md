# LLM Anonymizer

A Presidio-compatible PII anonymization API powered by local Ollama LLMs. Built with FastAPI and Claude Code.

**License:** AGPLv3

## What it does

Drop-in replacement for Microsoft Presidio's `/analyze` and `/anonymize` endpoints, but using your own local LLMs (via Ollama/vLLM) instead of spaCy/transformers for PII detection. The goal is to explore how good recall and precision can get with local LLMs for PII detection.

**Key features:**
- **Natural-language entity definitions** — Adding a new entity type is as simple as writing a Pydantic class with a docstring describing what to look for in plain language (see `entities_example.py`). No training data or model fine-tuning needed.
- **Regex pre-processing** — Optionally combine LLM detection with traditional regex patterns via a JSON recognizer file (see `custom_presidio_recognizers.json`) for deterministic matching of structured PII like phone numbers.
- **Semantic chunking** for long documents, batch mode, consensus mode, and best-of scoring.

## Quick start

```bash
# Run directly (uv auto-installs dependencies)
uv run app.py --model qwen3:8b --base-url http://localhost:11434 --pydantic-file entities_example.py

# Or with Docker
cd docker
cp env-example .env  # edit as needed
sudo docker compose up --build
```

## Usage

```bash
# Health check
curl http://localhost:3000/health

# Analyze (detect PII)
curl -X POST http://localhost:3000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient John Doe, born 1990-01-15, phone 06 12 34 56 78", "language": "en"}'

# Anonymize (replace PII)
curl -X POST http://localhost:3000/anonymize \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient John Doe", "analyzer_results": [{"entity_type": "PERSON", "start": 8, "end": 16, "score": 1.0}]}'
```

## Configuration

All options can be set via CLI args or environment variables. See `docker/env-example` for the full list with documentation.

Key settings:
- `--model` / `MODEL` — Ollama model for PII detection (required)
- `--base-url` / `BASE_URL` — Ollama endpoint URL (required)
- `--pydantic-file` / `PYDANTIC_FILE` — Python file with entity schemas (required)
- `--api` / `API` — LLM backend: `ollama` (default) or `vllm`
- `--default-recognizer-json` / `DEFAULT_RECOGNIZER_JSON` — Regex patterns applied before LLM
- `--embedding-model` / `EMBEDDING_MODEL` — Model for semantic chunking (default: `minishlab/potion-multilingual-128M`)
- `--n-parallel-requests` / `N_PARALLEL_REQUESTS` — Max concurrent LLM requests (default: 1)
- `--match-tkn-len` / `MATCH_TKN_LEN` — Chunk size in tokens (default: 256)
- `--custom-system-prompt` / `CUSTOM_SYSTEM_PROMPT` — Override the default system prompt
- `--thinking-model` / `THINKING_MODEL` — Strip `<think>` blocks from LLM output (default: true)
- `--think` / `THINK` — Pass `think=True/False` to Ollama chat calls; useful to work around Ollama bugs where think breaks structured output (default: not passed)
- `--sure` / `SURE` — Consensus mode: 0 or 1 = single call at temperature=0; N≥2 = repeat at temperature=1 until N identical results, up to `--upto` attempts
- `--upto` / `UPTO` — Max attempts for consensus (`--sure`) or best-of (`--bestof`) mode (default: 10)
- `--bestof` / `BESTOF` — Make N calls per entity, keep highest certainty (requires `--score`; cannot combine with `--sure`)
- `--score` / `SCORE` — Ask LLM to estimate certainty (0–100) per entity; divided by 100 for Presidio score
- `--auto-reload` / `AUTO_RELOAD` — Hot-reload pydantic entity file on every request (default: false)
- `--log-level` / `LOG_LEVEL` — Log level: `INFO` or `DEBUG` (default: INFO)

## Entity schemas

Define one Pydantic BaseModel per entity type in a `.py` file. See `entities_example.py`.

## Additional features

### Batch mode

Both `/analyze` and `/anonymize` accept `text` as a list of strings for batch processing. For `/anonymize`, `analyzer_results` must then be a list of lists.

```bash
curl -X POST http://localhost:3000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": ["Patient John Doe", "Call 06 12 34 56 78"], "language": "en"}'
```

### Ad-hoc recognizers

Pass per-request regex recognizers via the `ad_hoc_recognizers` field in `/analyze`, using the same format as the recognizer JSON file. These are applied alongside `--default-recognizer-json` patterns.

### Anonymization operators

The `/anonymize` endpoint supports Presidio-compatible operators via the `anonymizers` field:
- **replace** (default) — replace with `<ENTITY_TYPE>` or a custom `new_value`
- **redact** — remove the entity entirely
- **mask** — partial masking with `masking_char`, `chars_to_mask`, and `from_end`
- **hash** — replace with a hash (`sha256` by default, configurable via `hash_type`)
- **keep** — leave the entity as-is

Use `"DEFAULT"` as key to set a fallback operator for all entity types.

### Fuzzy key matching

When the LLM returns valid JSON but with slightly wrong field names (e.g. `"patient_name"` instead of `"name"`), `rapidfuzz` is used to match keys to the closest schema field before failing. This improves robustness with smaller models.

### Result filtering

The `/analyze` endpoint supports `score_threshold` (minimum confidence) and `entities` (list of entity types to return) for filtering results. `allow_list` is accepted but not yet fully implemented.

### vLLM backend

Use `--api vllm` to connect to a vLLM server via its OpenAI-compatible API instead of Ollama. Structured output uses `guided_json`.

## How it works

1. **Regex pre-processing** (optional) — Presidio-compatible pattern recognizers run first
2. **Semantic chunking** — Long texts are split using chonkie + tiktoken (GPT-4o tokenizer) with 15% overlap
3. **LLM detection** — Each chunk is sent to Ollama/vLLM once per entity schema, using structured output
4. **Reassembly** — Overlapping chunk results are merged (union strategy)
5. **Anonymization** — Detected spans are replaced with `<ENTITY_TYPE>` placeholders (or custom operators)

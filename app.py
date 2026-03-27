# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "ollama",
#   "openai",
#   "pydantic",
#   "tiktoken",
#   "chonkie[semantic]",
#   "loguru",
#   "rapidfuzz",
# ]
# ///
"""LLM-powered Presidio-compatible PII anonymization API."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import tiktoken
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from loguru import logger
from pydantic import BaseModel

# ── Request / Response models (Presidio-compatible) ─────────────────────────


class AnalyzeRequest(BaseModel):
    text: str | list[str]
    language: str = "en"  # accepted but ignored
    entities: list[str] | None = None
    score_threshold: float = 0.0
    return_decision_process: bool = False
    correlation_id: str | None = None
    ad_hoc_recognizers: list[dict[str, Any]] | None = None
    context: list[str] | None = None
    allow_list: list[str] | None = None
    allow_list_match: str = "exact"
    regex_flags: int | None = None


class RecognizerResult(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float = 1.0


class AnonymizeRequest(BaseModel):
    text: str | list[str]
    analyzer_results: list[RecognizerResult] | list[list[RecognizerResult]]
    anonymizers: dict[str, dict[str, Any]] | None = None


class OperatorResult(BaseModel):
    operator: str
    entity_type: str
    start: int
    end: int
    text: str


class AnonymizeResponse(BaseModel):
    text: str
    items: list[OperatorResult]


# ── Configuration ───────────────────────────────────────────────────────────


@dataclass
class Config:
    model: str
    base_url: str
    pydantic_file: str
    embedding_model: str = "minishlab/potion-multilingual-128M"
    n_parallel_requests: int = 1
    match_tkn_len: int = 256
    custom_system_prompt: str = ""
    thinking_model: bool = True
    default_recognizer_json: str = ""
    auto_reload: bool = False
    log_level: str = "INFO"
    api: str = "ollama"
    sure: int = 0
    upto: int = 10
    score: bool = False
    bestof: int = 1
    think: bool | None = None


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="LLM Anonymizer — Presidio-compatible PII API"
    )
    p.add_argument(
        "--model", default=os.environ.get("MODEL", ""), help="Ollama model name"
    )
    p.add_argument(
        "--base-url", default=os.environ.get("BASE_URL", ""), help="Ollama endpoint URL"
    )
    p.add_argument(
        "--api",
        default=os.environ.get("API", "ollama"),
        choices=["ollama", "vllm"],
        help="LLM backend API to use (default: ollama)",
    )
    p.add_argument(
        "--pydantic-file",
        default=os.environ.get("PYDANTIC_FILE", ""),
        help="Path to .py with entity BaseModels",
    )
    p.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "minishlab/potion-multilingual-128M"),
        help="Embedding model for semantic chunking",
    )
    p.add_argument(
        "--n-parallel-requests",
        type=int,
        default=int(os.environ.get("N_PARALLEL_REQUESTS", "1")),
        help="Max concurrent Ollama requests",
    )
    p.add_argument(
        "--match-tkn-len",
        type=int,
        default=int(os.environ.get("MATCH_TKN_LEN", "256")),
        help="Chunk size in tokens",
    )
    p.add_argument(
        "--custom-system-prompt",
        default=os.environ.get("CUSTOM_SYSTEM_PROMPT", ""),
        help="Override default system prompt",
    )
    p.add_argument(
        "--thinking-model",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=os.environ.get("THINKING_MODEL", "true").lower()
        in ("true", "1", "yes"),
        help="Strip <think> blocks from LLM output",
    )
    p.add_argument(
        "--default-recognizer-json",
        default=os.environ.get("DEFAULT_RECOGNIZER_JSON", ""),
        help="Path to recognizer JSON",
    )
    p.add_argument(
        "--auto-reload",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=os.environ.get("AUTO_RELOAD", "false").lower() in ("true", "1", "yes"),
        help="Hot-reload pydantic file per request",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Log level (INFO or DEBUG)",
    )
    p.add_argument(
        "--sure",
        type=int,
        default=int(os.environ.get("SURE", "0")),
        help="Consensus mode: 0 or 1 = single call at temperature=0; N>=2 = repeat at temperature=1 until N identical results, up to --upto attempts",
    )
    p.add_argument(
        "--upto",
        type=int,
        default=int(os.environ.get("UPTO", "10")),
        help="Max attempts in consensus mode (--sure) or best-of mode (--bestof)",
    )
    p.add_argument(
        "--score",
        action="store_true",
        default=os.environ.get("SCORE", "false").lower() in ("true", "1", "yes"),
        help="Ask the LLM to estimate its certainty (0–100 integer) per entity; divided by 100 to get the Presidio score",
    )
    p.add_argument(
        "--bestof",
        type=int,
        default=int(os.environ.get("BESTOF", "1")),
        help="Make N LLM calls per entity and keep the answer with highest certainty (requires --score)",
    )
    p.add_argument(
        "--think",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=None if os.environ.get("THINK", "").lower() not in ("true", "1", "yes", "false", "0", "no") else os.environ.get("THINK", "").lower() in ("true", "1", "yes"),
        help="Pass think=True/False to Ollama chat calls. Default: None (not passed). Useful to work around Ollama bugs where think breaks structured output.",
    )
    args = p.parse_args()

    if not args.model:
        p.error("--model is required (or set MODEL env var)")
    if not args.base_url:
        p.error("--base-url is required (or set BASE_URL env var)")
    if not args.pydantic_file:
        p.error("--pydantic-file is required (or set PYDANTIC_FILE env var)")
    if args.bestof >= 2 and args.sure >= 2:
        p.error("--bestof and --sure cannot be used together")
    if args.bestof >= 2:
        args.score = True

    return Config(
        model=args.model,
        base_url=args.base_url,
        api=args.api,
        pydantic_file=args.pydantic_file,
        embedding_model=args.embedding_model,
        n_parallel_requests=args.n_parallel_requests,
        match_tkn_len=args.match_tkn_len,
        custom_system_prompt=args.custom_system_prompt,
        thinking_model=args.thinking_model,
        default_recognizer_json=args.default_recognizer_json,
        auto_reload=args.auto_reload,
        log_level=args.log_level,
        sure=args.sure,
        upto=args.upto,
        score=args.score,
        bestof=args.bestof,
        think=args.think,
    )


# ── Pydantic schema loader ─────────────────────────────────────────────────


def _camel_to_upper_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def load_pydantic_schemas(path: str) -> list[tuple[str, type[BaseModel]]]:
    """Load .py file, return list of (entity_type, ModelClass) for each BaseModel subclass."""
    spec = importlib.util.spec_from_file_location("_entities", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    schemas: list[tuple[str, type[BaseModel]]] = []
    for name, obj in vars(mod).items():
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
        ):
            entity_type = getattr(obj, "entity_type", None) or _camel_to_upper_snake(
                name
            )
            schemas.append((entity_type, obj))
    logger.info("Loaded {} entity schemas from {}", len(schemas), path)
    return schemas


# ── Regex pre-processor ─────────────────────────────────────────────────────


@dataclass
class RegexPattern:
    name: str
    regex: re.Pattern[str]
    score: float


@dataclass
class RegexRecognizer:
    entity_type: str
    patterns: list[RegexPattern]


def load_recognizers(path: str) -> list[RegexRecognizer]:
    """Load Presidio-style PatternRecognizer JSON."""
    with open(path) as f:
        data = json.load(f)
    recognizers: list[RegexRecognizer] = []
    for item in data:
        patterns = [
            RegexPattern(
                name=p["name"],
                regex=re.compile(p["regex"]),
                score=p.get("score", 0.5),
            )
            for p in item.get("patterns", [])
        ]
        recognizers.append(
            RegexRecognizer(entity_type=item["supported_entity"], patterns=patterns)
        )
    logger.info("Loaded {} regex recognizers from {}", len(recognizers), path)
    return recognizers


def _parse_ad_hoc_recognizers(raw: list[dict[str, Any]]) -> list[RegexRecognizer]:
    """Parse ad_hoc_recognizers from a request payload."""
    recognizers: list[RegexRecognizer] = []
    for item in raw:
        patterns = [
            RegexPattern(
                name=p["name"], regex=re.compile(p["regex"]), score=p.get("score", 0.5)
            )
            for p in item.get("patterns", [])
        ]
        recognizers.append(
            RegexRecognizer(entity_type=item["supported_entity"], patterns=patterns)
        )
    return recognizers


def apply_regex_recognizers(
    text: str,
    recognizers: list[RegexRecognizer],
) -> tuple[str, list[RecognizerResult]]:
    """Apply regex recognizers to text. Returns (modified_text, detections).

    Matches are collected, deduplicated (longer/higher-score wins on overlap),
    then replaced from right to left to preserve offsets.
    """
    raw_matches: list[
        tuple[int, int, str, float]
    ] = []  # (start, end, entity_type, score)
    for rec in recognizers:
        for pat in rec.patterns:
            for m in pat.regex.finditer(text):
                raw_matches.append((m.start(), m.end(), rec.entity_type, pat.score))

    if not raw_matches:
        return text, []

    # Sort by start asc, then by length desc, then score desc
    raw_matches.sort(key=lambda x: (x[0], -(x[1] - x[0]), -x[3]))

    # Deduplicate overlapping matches: greedy non-overlapping, prefer longer then higher score
    kept: list[tuple[int, int, str, float]] = []
    last_end = -1
    for start, end, etype, score in raw_matches:
        if start >= last_end:
            kept.append((start, end, etype, score))
            last_end = end

    # Build results with original-text offsets and replace from right to left
    results: list[RecognizerResult] = []
    modified = text
    for start, end, etype, score in reversed(kept):
        results.append(
            RecognizerResult(entity_type=etype, start=start, end=end, score=score)
        )
        modified = modified[:start] + f"<{etype}>" + modified[end:]

    results.reverse()  # back to ascending order
    return modified, results


# ── Tokenizer ───────────────────────────────────────────────────────────────

_tiktoken_enc: tiktoken.Encoding | None = None


def get_tokenizer() -> tiktoken.Encoding:
    global _tiktoken_enc
    if _tiktoken_enc is None:
        _tiktoken_enc = tiktoken.encoding_for_model("gpt-4o")
    return _tiktoken_enc


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text))


# ── Chunking ────────────────────────────────────────────────────────────────


@dataclass
class TextChunk:
    text: str
    start: int  # character offset in original text
    end: int  # character offset in original text


def chunk_text(text: str, max_tokens: int, embedding_model: str) -> list[TextChunk]:
    """Split text into semantically meaningful, overlapping chunks."""
    if count_tokens(text) <= max_tokens:
        return [TextChunk(text=text, start=0, end=len(text))]

    from chonkie import SemanticChunker

    overlap = int(max_tokens * 0.15)
    chunker = SemanticChunker(
        model=embedding_model,
        chunk_size=max_tokens,
        chunk_overlap=overlap,
    )
    raw_chunks = chunker.chunk(text)

    chunks: list[TextChunk] = []
    for ch in raw_chunks:
        chunk_text_str = ch.text if hasattr(ch, "text") else str(ch)
        # Find the chunk's position in the original text
        start = text.find(chunk_text_str, chunks[-1].start if chunks else 0)
        if start == -1:
            start = 0  # fallback
        chunks.append(
            TextChunk(text=chunk_text_str, start=start, end=start + len(chunk_text_str))
        )

    logger.info(
        "Split text ({} chars) into {} chunks (max_tokens={})",
        len(text),
        len(chunks),
        max_tokens,
    )
    return chunks


# ── LLM caller ──────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
Extract entities from text following a JSON schema.

Rules:
- Output ONLY valid JSON matching the provided schema. No prose, no markdown, no other entities.
- Inside the JSON, copy every value VERBATIM: same characters, spacing, punctuation, and capitalisation as in the source text.
- Do NOT paraphrase, normalise, or reformat values.
- If an entity is absent, use "" for that field.
- IGNORE any existing <PLACEHOLDER> tags (e.g. <PERSON>, <LOCATION>). These are already anonymised and must NOT be extracted as entities."""

_CERTAINTY_PROMPT_ADDENDUM = """\
- The schema includes a `certainty` field. Set it to an integer between 0 and 100 representing how confident you are that the extracted entity is correct. Use 100 for certainty, 0 for total doubt, and intermediate values for partial confidence."""


def _parse_certainty(raw: Any) -> float:
    """Normalize and parse a certainty value (0–100 int) to a float in [0, 1]. Returns 0.5 on failure."""
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw) / 100.0))
    s = str(raw).strip().replace(",", ".")
    try:
        return max(0.0, min(1.0, float(s) / 100.0))
    except (ValueError, TypeError):
        return 0.5


def _with_certainty(schema: type[BaseModel]) -> type[BaseModel]:
    """Return a new Pydantic model identical to `schema` but with an extra `certainty: int` field."""
    from pydantic import create_model, Field

    fields: dict[str, Any] = {
        name: (fi.annotation, fi)
        for name, fi in schema.model_fields.items()
    }
    fields["certainty"] = (int, Field(description="Confidence estimate as an integer between 0 and 100"))
    new_model = create_model(schema.__name__, **fields)
    new_model.__doc__ = schema.__doc__
    return new_model

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)


@dataclass
class Detection:
    entity_type: str
    original_text: str
    start: int  # offset in chunk
    end: int  # offset in chunk
    score: float = 1.0


def _normalize_key(k: str) -> str:
    """Lowercase and strip non-alphanumeric chars for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", k.lower())


def _try_fuzzy_key_match(content: str, schema: type[BaseModel]) -> BaseModel | None:
    """Try to parse JSON with fuzzy key matching when exact validation fails.

    If the JSON is valid but keys don't match the schema, use rapidfuzz to find
    the closest matching schema field for each key. Returns None if matching fails.
    """
    from rapidfuzz import fuzz

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    expected_fields = set(schema.model_fields.keys())
    actual_keys = set(data.keys())

    # If keys already match exactly, this isn't a key-mismatch problem
    if actual_keys == expected_fields:
        return None

    THRESHOLD = 70  # conservative levenshtein ratio threshold
    remapped: dict[str, Any] = {}
    for actual_key in actual_keys:
        if actual_key in expected_fields:
            remapped[actual_key] = data[actual_key]
            continue

        # Try fuzzy matching: raw keys first, then normalized
        best_field: str | None = None
        best_score: float = 0.0
        norm_actual = _normalize_key(actual_key)
        for expected in expected_fields:
            # Raw comparison
            score_raw = fuzz.ratio(actual_key, expected)
            # Normalized comparison (handles camelCase, underscores, hyphens)
            score_norm = fuzz.ratio(norm_actual, _normalize_key(expected))
            score = max(score_raw, score_norm)
            if score > best_score:
                best_score = score
                best_field = expected

        if best_field is not None and best_score >= THRESHOLD:
            logger.warning(
                "Fuzzy key match: '{}' -> '{}' (score={:.1f})",
                actual_key, best_field, best_score,
            )
            remapped[best_field] = data[actual_key]
        # Keys that don't match any field are silently dropped (Pydantic would ignore them anyway)

    try:
        result = schema.model_validate(remapped)
        logger.warning(
            "Successfully parsed {} via fuzzy key remapping (original keys: {})",
            schema.__name__, list(actual_keys),
        )
        return result
    except Exception:
        return None


async def _single_llm_call(
    client: Any,
    model: str,
    system_prompt: str,
    user_content: str,
    schema: type[BaseModel],
    thinking_model: bool,
    semaphore: asyncio.Semaphore,
    api: str = "ollama",
    temperature: float | None = 0.0,
    retry_content: str | None = None,
    think: bool | None = None,
) -> BaseModel | None:
    """One raw LLM call with optional parse-failure retry. Returns parsed model or None."""
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"{user_content}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n"
            f"```json\n{schema_json}\n```"
        )},
    ]
    if retry_content:
        messages[1]["content"] += (
            f"\n\nPrevious attempt returned: {retry_content}\n"
            "which could not be parsed. Please try again carefully."
        )

    schema_tokens = count_tokens(schema_json)
    chunk_tokens = count_tokens(user_content)
    num_predict = 3 * schema_tokens + 2 * chunk_tokens

    async with semaphore:
        t0 = time.perf_counter()
        if api == "vllm":
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                    },
                },
                "max_tokens": num_predict,
            }
            if think is not None:
                kwargs["extra_body"] = {"think": think}
            response = await client.chat.completions.create(**kwargs)
        else:
            options: dict[str, Any] = {"num_predict": num_predict}
            if temperature is not None:
                options["temperature"] = temperature
            chat_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "format": schema.model_json_schema(),
                "options": options,
            }
            if think is not None:
                chat_kwargs["think"] = think
            response = await client.chat(**chat_kwargs)
        elapsed = time.perf_counter() - t0

    if api == "vllm":
        content = response.choices[0].message.content or response.choices[0].message.reasoning or response.choices[0].message.refusal
    else:
        content = response.message.content
    if not content.strip():
        logger.warning("Empty LLM response: ({:.2f}s): {}\n{}", elapsed, content[:200], response)
    else:
        logger.debug("LLM response ({:.2f}s): {}", elapsed, content[:200])

    if thinking_model:
        content = _THINK_RE.sub("", content).strip()

    try:
        return schema.model_validate_json(content)
    except Exception as e:
        # Try fuzzy key matching before retrying/failing
        fuzzy_result = _try_fuzzy_key_match(content, schema)
        if fuzzy_result is not None:
            return fuzzy_result
        if retry_content is None:
            logger.warning(f"Parse failed, retrying once: '{e}'")
            return await _single_llm_call(
                client, model, system_prompt, user_content, schema,
                thinking_model, semaphore, api, temperature, retry_content=content, think=think,
            )
        logger.error(f"Parse failed on retry, skipping: '{e}'")
        return None


async def call_llm(
    client: Any,
    model: str,
    system_prompt: str,
    user_content: str,
    schema: type[BaseModel],
    thinking_model: bool,
    semaphore: asyncio.Semaphore,
    api: str = "ollama",
    sure: int = 0,
    upto: int = 10,
    bestof: int = 1,
    think: bool | None = None,
) -> BaseModel | None:
    """LLM call with optional consensus or best-of mode.

    sure=0 or 1: single call at temperature=0 (original behaviour).
    sure=N (N>=2): repeat at temperature=1 until N identical parsed
                   results are seen, up to `upto` total attempts.
    bestof=N (N>=2): make N calls (up to `upto`), return the one with
                     highest `certainty` field (requires --score schema).
    """
    if sure <= 1 and bestof <= 1:
        return await _single_llm_call(
            client, model, system_prompt, user_content, schema, thinking_model, semaphore,
            api=api, temperature=0.0, think=think,
        )

    if bestof >= 2:
        # Best-of mode: collect `bestof` valid results (certainty 0–100), up to `upto` total attempts.
        # Results with out-of-range certainty are discarded and retried.
        valid: list[tuple[float, BaseModel]] = []
        attempts = 0
        while len(valid) < bestof and attempts < upto:
            remaining = min(bestof - len(valid), upto - attempts)
            call_tasks = [
                _single_llm_call(
                    client, model, system_prompt, user_content, schema, thinking_model, semaphore,
                    api=api, temperature=1.0, think=think,
                )
                for _ in range(remaining)
            ]
            results = await asyncio.gather(*call_tasks)
            attempts += remaining
            for r in results:
                if r is None:
                    continue
                raw_cert = r.model_dump().get("certainty")
                if not isinstance(raw_cert, (int, float)) or not (0 <= raw_cert <= 100):
                    logger.debug("Best-of: discarding result with out-of-range certainty={}", raw_cert)
                    continue
                valid.append((_parse_certainty(raw_cert), r))
        if not valid:
            return None
        best_certainty, best = max(valid, key=lambda x: x[0])
        logger.debug("Best-of {}/{} valid: best certainty={:.3f}", len(valid), attempts, best_certainty)
        return best

    # Consensus mode — use model-default temperature (don't send temperature)
    from collections import Counter
    counts: Counter[str] = Counter()
    first_instance: dict[str, BaseModel] = {}
    attempts = 0

    while attempts < upto:
        result = await _single_llm_call(
            client, model, system_prompt, user_content, schema, thinking_model, semaphore,
            api=api, temperature=1.0, think=think,
        )
        attempts += 1
        if result is None:
            continue

        key = json.dumps(result.model_dump(), sort_keys=True)
        if key not in first_instance:
            first_instance[key] = result
        counts[key] += 1

        if counts[key] >= sure:
            logger.debug(
                "Consensus reached after {} attempts (agreed {} times)", attempts, sure
            )
            return first_instance[key]

    logger.warning(
        "Consensus not reached after {} attempts (sure={}), returning most common result",
        upto, sure,
    )
    if counts:
        best_key, _ = counts.most_common(1)[0]
        return first_instance[best_key]
    return None


# ── Per-chunk processing ────────────────────────────────────────────────────


async def process_chunk(
    chunk: str,
    schemas: list[tuple[str, type[BaseModel]]],
    client: Any,
    config: Config,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[Detection]]:
    """Process one chunk through all schemas sequentially."""
    current = chunk  # progressively modified text (with placeholders) sent to LLM
    original = chunk  # kept intact for offset computation
    all_detections: list[Detection] = []
    claimed: list[tuple[int, int]] = []  # regions in original text already detected
    system_prompt = config.custom_system_prompt or DEFAULT_SYSTEM_PROMPT
    if config.score:
        system_prompt = system_prompt + "\n" + _CERTAINTY_PROMPT_ADDENDUM
        schemas = [(et, _with_certainty(sc)) for et, sc in schemas]

    logger.debug("Chunk text BEFORE processing:\n{}", current)

    for entity_type, schema in schemas:
        # Skip if chunk is now only placeholders and whitespace
        if re.fullmatch(r"[\s]*(<[A-Z_]+>[\s]*)*", current):
            last_schema_name = schemas[schemas.index((entity_type, schema)) - 1][1].__name__ if schemas.index((entity_type, schema)) > 0 else "unknown"
            last_det = all_detections[-1] if all_detections else None
            last_value = f"'{last_det.original_text}'" if last_det else "?"
            logger.warning(
                "Chunk is only placeholders/whitespace after {} — skipping {} remaining entities. "
                "Last match was {} which may have been overzealous.",
                last_schema_name, len(schemas) - schemas.index((entity_type, schema)), last_value,
            )
            break

        user_content = f"<text>{current}</text>"
        # user_content = current
        schema_name = schema.__name__
        logger.info("Detecting {} ({}) in chunk ({} chars)", schema_name, entity_type, len(current))
        logger.debug("Text sent to LLM for {} detection:\n{}", schema_name, current)

        result = await call_llm(
            client,
            config.model,
            system_prompt,
            user_content,
            schema,
            config.thinking_model,
            semaphore,
            api=config.api,
            sure=config.sure,
            upto=config.upto,
            bestof=config.bestof,
            think=config.think,
        )
        if result is None:
            continue

        result_dict = result.model_dump()
        detection_score = _parse_certainty(result_dict.pop("certainty", 1.0)) if config.score else 1.0

        # Extract field values and find them in the chunk.
        # Values may be str, int, or list thereof (e.g. PhoneNumber.phone is
        # Union[str, int, List[Union[str, int]]]), so normalise to a flat list
        # of strings before searching.
        def _is_empty_value(v: str) -> bool:
            return v in ("", "<none>", "'", "\"\"", "[]", "()")

        for field_name, value in result_dict.items():
            if isinstance(value, list):
                candidates: list[str] = [
                    str(v).strip() for v in value if not _is_empty_value(str(v).strip())
                ]
            elif value is not None and not _is_empty_value(str(value).strip()):
                candidates = [str(value).strip()]
            else:
                continue

            for candidate in candidates:
                # Skip if the candidate is itself a placeholder from a previous pass
                if re.fullmatch(r"<[A-Z_]+>", candidate):
                    logger.debug("Skipping placeholder value '{}'", candidate)
                    continue

                # Find the candidate in the ORIGINAL text (not the modified one)
                # to get correct offsets, skipping already-claimed regions.
                search_start = 0
                found = False
                while True:
                    idx = original.find(candidate, search_start)
                    if idx == -1:
                        # Try case-insensitive
                        idx = original.lower().find(candidate.lower(), search_start)
                    if idx == -1:
                        break
                    # Check overlap with already-claimed regions
                    cand_end = idx + len(candidate)
                    if any(not (cand_end <= cs or idx >= ce) for cs, ce in claimed):
                        search_start = idx + 1
                        continue
                    found = True
                    break

                if not found:
                    logger.debug("Value '{}' not found in chunk, skipping", candidate)
                    continue

                detection = Detection(
                    entity_type=entity_type,
                    original_text=candidate,
                    start=idx,
                    end=idx + len(candidate),
                    score=detection_score,
                )
                all_detections.append(detection)
                claimed.append((idx, idx + len(candidate)))
                logger.debug(
                    "Found {} ({}): '{}' at [{}, {}]",
                    schema_name,
                    entity_type,
                    candidate,
                    idx,
                    idx + len(candidate),
                )

                # Replace in current text so next schema sees the placeholder
                cidx = current.find(candidate)
                if cidx == -1:
                    cidx = current.lower().find(candidate.lower())
                if cidx != -1:
                    placeholder = f"<{entity_type}>"
                    current = current[:cidx] + placeholder + current[cidx + len(candidate) :]

        logger.debug("Text AFTER {} replacement:\n{}", schema_name, current)

    logger.debug("Chunk text AFTER all processing:\n{}", current)
    return current, all_detections


# ── Reassembly ──────────────────────────────────────────────────────────────


def _map_chunk_detections_to_global(
    chunks: list[TextChunk],
    chunk_detections: list[list[Detection]],
) -> list[RecognizerResult]:
    """Map chunk-local detection offsets to global text offsets. Union dedup."""
    seen: set[tuple[int, int, str]] = set()
    results: list[RecognizerResult] = []

    for chunk, detections in zip(chunks, chunk_detections):
        for det in detections:
            global_start = chunk.start + det.start
            global_end = chunk.start + det.end
            key = (global_start, global_end, det.entity_type)
            if key not in seen:
                seen.add(key)
                results.append(
                    RecognizerResult(
                        entity_type=det.entity_type,
                        start=global_start,
                        end=global_end,
                        score=det.score,
                    )
                )

    results.sort(key=lambda r: r.start)
    return results


# ── Anonymization (apply operators) ─────────────────────────────────────────


def anonymize_text(
    text: str,
    detections: list[RecognizerResult],
    anonymizers: dict[str, dict[str, Any]] | None = None,
) -> AnonymizeResponse:
    """Apply anonymization operators. Default: replace with <ENTITY_TYPE>."""
    if anonymizers is None:
        anonymizers = {}

    # Sort descending by start to replace from right to left
    sorted_dets = sorted(detections, key=lambda d: d.start, reverse=True)
    items: list[OperatorResult] = []
    modified = text

    for det in sorted_dets:
        op_config = anonymizers.get(
            det.entity_type, anonymizers.get("DEFAULT", {"type": "replace"})
        )
        op_type = op_config.get("type", "replace")

        if op_type == "replace":
            replacement = op_config.get("new_value", "") or f"<{det.entity_type}>"
        elif op_type == "redact":
            replacement = ""
        elif op_type == "mask":
            char = op_config.get("masking_char", "*")
            count = op_config.get("chars_to_mask", det.end - det.start)
            from_end = op_config.get("from_end", False)
            original_span = modified[det.start : det.end]
            if from_end:
                replacement = original_span[: len(original_span) - count] + char * count
            else:
                replacement = char * count + original_span[count:]
        elif op_type == "hash":
            import hashlib

            hash_type = op_config.get("hash_type", "sha256")
            original_span = modified[det.start : det.end]
            h = hashlib.new(hash_type, original_span.encode())
            replacement = h.hexdigest()
        elif op_type == "keep":
            replacement = modified[det.start : det.end]
        else:
            replacement = f"<{det.entity_type}>"

        modified = modified[: det.start] + replacement + modified[det.end :]
        # We'll fix output offsets after all replacements
        items.append(
            OperatorResult(
                operator=op_type,
                entity_type=det.entity_type,
                start=det.start,  # placeholder, fixed below
                end=det.start + len(replacement),
                text=replacement,
            )
        )

    # Recalculate output offsets (items are in reverse order, fix them)
    items.reverse()
    # Recompute offsets by scanning the output text
    offset_shift = 0
    final_items: list[OperatorResult] = []
    sorted_dets_asc = sorted(detections, key=lambda d: d.start)
    for det, item in zip(sorted_dets_asc, items):
        new_start = det.start + offset_shift
        new_end = new_start + len(item.text)
        final_items.append(
            OperatorResult(
                operator=item.operator,
                entity_type=item.entity_type,
                start=new_start,
                end=new_end,
                text=item.text,
            )
        )
        offset_shift += len(item.text) - (det.end - det.start)

    return AnonymizeResponse(text=modified, items=final_items)


# ── Full analysis pipeline ──────────────────────────────────────────────────


async def analyze_single(
    text: str,
    schemas: list[tuple[str, type[BaseModel]]],
    default_recognizers: list[RegexRecognizer],
    ad_hoc_recognizers: list[RegexRecognizer],
    config: Config,
    client: Any,
    semaphore: asyncio.Semaphore,
) -> list[RecognizerResult]:
    """Run full analysis pipeline on a single text."""
    t0 = time.perf_counter()
    logger.info("Analyzing text ({} chars)", len(text))

    # Step 1: Regex pre-processing
    all_recognizers = default_recognizers + ad_hoc_recognizers
    if all_recognizers:
        regex_text, regex_results = apply_regex_recognizers(text, all_recognizers)
        logger.info("Regex found {} entities", len(regex_results))
    else:
        regex_text = text
        regex_results = []

    # Step 2: Chunk
    chunks = chunk_text(regex_text, config.match_tkn_len, config.embedding_model)

    # Step 3: Process each chunk (chunks can run in parallel, schemas within a chunk are sequential)
    tasks = [
        process_chunk(ch.text, schemas, client, config, semaphore) for ch in chunks
    ]
    chunk_results = await asyncio.gather(*tasks)

    # Step 4: Reassemble
    chunk_detections = [dets for _, dets in chunk_results]
    llm_results = _map_chunk_detections_to_global(chunks, chunk_detections)
    logger.info("LLM found {} entities", len(llm_results))

    # Merge regex + LLM results
    all_results = regex_results + llm_results
    all_results.sort(key=lambda r: r.start)

    elapsed = time.perf_counter() - t0
    logger.debug("Analysis completed in {:.2f}s", elapsed)
    return all_results


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Anonymizer",
    description="Presidio-compatible PII anonymization powered by Ollama",
)

# Globals set in main
_config: Config | None = None
_schemas: list[tuple[str, type[BaseModel]]] = []
_default_recognizers: list[RegexRecognizer] = []
_semaphore: asyncio.Semaphore | None = None
_ollama_client: Any = None


def _get_schemas() -> list[tuple[str, type[BaseModel]]]:
    global _schemas
    if _config and _config.auto_reload:
        _schemas = load_pydantic_schemas(_config.pydantic_file)
    return _schemas


@app.get("/health")
async def health():
    return PlainTextResponse("Presidio Analyzer/Anonymizer service is up")


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    assert _config and _semaphore and _ollama_client
    schemas = _get_schemas()

    ad_hoc = (
        _parse_ad_hoc_recognizers(request.ad_hoc_recognizers)
        if request.ad_hoc_recognizers
        else []
    )

    if isinstance(request.text, list):
        # Batch mode
        tasks = [
            analyze_single(
                t,
                schemas,
                _default_recognizers,
                ad_hoc,
                _config,
                _ollama_client,
                _semaphore,
            )
            for t in request.text
        ]
        batch_results = await asyncio.gather(*tasks)
        # Apply filters
        filtered = []
        for results in batch_results:
            filtered.append(_filter_results(results, request))
        return filtered
    else:
        results = await analyze_single(
            request.text,
            schemas,
            _default_recognizers,
            ad_hoc,
            _config,
            _ollama_client,
            _semaphore,
        )
        return _filter_results(results, request)


def _filter_results(
    results: list[RecognizerResult],
    request: AnalyzeRequest,
) -> list[RecognizerResult]:
    """Apply score_threshold, entities filter, and allow_list."""
    filtered = [r for r in results if r.score >= request.score_threshold]
    if request.entities:
        filtered = [r for r in filtered if r.entity_type in request.entities]
    if request.allow_list:
        # We'd need the original text to check allow_list — for now just pass through
        pass
    return filtered


@app.post("/anonymize")
async def anonymize_endpoint(request: AnonymizeRequest):
    if isinstance(request.text, list):
        # Batch mode
        if not isinstance(request.analyzer_results[0], list):
            raise ValueError(
                "For batch text, analyzer_results must be list[list[RecognizerResult]]"
            )
        responses = []
        for txt, results in zip(request.text, request.analyzer_results):
            resp = anonymize_text(txt, results, request.anonymizers)  # type: ignore[arg-type]
            responses.append(resp)
        return responses
    else:
        return anonymize_text(
            request.text, request.analyzer_results, request.anonymizers
        )  # type: ignore[arg-type]


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    global _config, _schemas, _default_recognizers, _semaphore, _ollama_client

    _config = parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=_config.log_level)

    # Initialize
    _schemas = load_pydantic_schemas(_config.pydantic_file)
    if _config.default_recognizer_json:
        _default_recognizers = load_recognizers(_config.default_recognizer_json)
    _semaphore = asyncio.Semaphore(_config.n_parallel_requests)

    if _config.api == "vllm":
        from openai import AsyncOpenAI

        _ollama_client = AsyncOpenAI(base_url=_config.base_url + "/v1", api_key="unused")
    else:
        from ollama import AsyncClient

        _ollama_client = AsyncClient(host=_config.base_url)

    logger.info(
        "Starting LLM Anonymizer (model={}, base_url={})",
        _config.model,
        _config.base_url,
    )
    uvicorn.run(app, host="0.0.0.0", port=3000)


if __name__ == "__main__":
    main()

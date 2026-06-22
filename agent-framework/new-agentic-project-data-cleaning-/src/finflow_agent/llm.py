"""LLM client helpers for the FinFlow Agent Service.

This module owns every outbound call to a large-language-model provider.
It exposes three public helpers:

* :func:`get_groq_client` returns a configured ``groq.Groq`` client that
  reuses a single ``httpx.Client`` for the lifetime of the process.
* :func:`get_chat_groq` returns a ``langchain_groq.ChatGroq`` instance for
  callers that need LangChain-style chains.
* :func:`call_groq_json` performs a single round trip and returns the
  parsed JSON object emitted by the model.

Privacy and safety contract
---------------------------
``call_groq_json`` is intended for **structured-output JSON only**. The
returned dictionary MUST be validated by a Pydantic model on the caller's
side before any of its values are used to drive behavior. In particular,
**no string returned by the LLM may ever be forwarded to**
``pandas.DataFrame.query``, ``eval``, ``exec``, ``subprocess``, or any
other code-evaluation surface anywhere in the call path. The structured
PlanIntent contract enforces this at the type level (its fields are typed
flags and structured plan models, never raw query strings); the
:func:`assert_no_eval_strings` helper below adds a defense-in-depth check
on the prompt side so any future regression in any caller is caught at
the boundary.
"""

from __future__ import annotations

import atexit
import os
from typing import Any

import httpx
from groq import Groq

from finflow_agent.llm_telemetry import get_runtime_context, log_runtime_event

_GROQ_HTTP_CLIENT: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    global _GROQ_HTTP_CLIENT
    if _GROQ_HTTP_CLIENT is None:
        _GROQ_HTTP_CLIENT = httpx.Client(timeout=10.0, trust_env=False)
        atexit.register(_GROQ_HTTP_CLIENT.close)
    return _GROQ_HTTP_CLIENT


def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing")
    return Groq(api_key=api_key, http_client=_get_http_client())


def get_chat_groq(*, model_name: str, temperature: float = 0.0) -> Any:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing")

    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        http_client=_get_http_client(),
    )


def get_structured_plan_intent_chain(
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
) -> Any:
    """Build a langchain-groq chain that emits a Pydantic-validated ``PlanIntent``.

    Using ``with_structured_output(PlanIntent)`` is strictly stronger than the
    raw JSON-mode round trip in :func:`call_groq_json`: the model can no longer
    emit a top-level ``steps`` key, fields outside the schema are dropped, and
    the result is already a Pydantic model rather than a raw dict that the
    caller has to re-validate.

    ``PlanIntent`` is imported lazily to avoid a circular import:
    ``finflow_agent.planning.intent_schema`` itself does not depend on this
    module today, but the orchestrator that consumes both lives one level
    above us, and keeping the import lazy preserves the option of evolving
    either side without rearranging the package.
    """
    from finflow_agent.planning.intent_schema import PlanIntent

    llm = get_chat_groq(model_name=model_name, temperature=temperature)
    return llm.with_structured_output(PlanIntent)


# ---------------------------------------------------------------------------
# Outbound role normalization (LLM/provider boundary)
# ---------------------------------------------------------------------------

# Map from internal/framework message-role tokens to Groq's wire-format
# discriminator values. Groq's chat.completions API accepts:
#   - "system"
#   - "user"
#   - "assistant"
#   - "tool"
#
# LangChain's ``BaseMessage.type`` emits framework roles instead:
#   - HumanMessage.type    == "human"
#   - AIMessage.type       == "ai"
#   - SystemMessage.type   == "system"
#   - ToolMessage.type     == "tool"
#
# Without normalization, a planner that builds payloads via
# ``[{"role": m.type, "content": m.content} for m in prompt.format_messages()]``
# leaks ``human``/``ai`` to the provider, which Groq rejects with HTTP 400:
#   ``'messages.N' : discriminator property 'role' has invalid value``
#
# This mapping is the single source of truth for that translation. It is
# applied at every outbound boundary in this module, so any other call
# path that ever lands here gets the same uniform contract.
_INTERNAL_TO_PROVIDER_ROLE: dict[str, str] = {
    # Wire roles (already correct) — pass through.
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    # LangChain framework roles — translated to wire format.
    "human": "user",
    "ai": "assistant",
}

# The set of roles the provider will accept after normalization. Anything
# outside this set is a contract violation that must fail fast rather than
# silently shipping a 400.
_ALLOWED_PROVIDER_ROLES: frozenset[str] = frozenset(
    {"system", "user", "assistant", "tool"}
)


def normalize_outbound_messages(messages: Any) -> list[dict[str, str]]:
    """Translate internal/framework message roles to provider wire roles.

    Accepts a list of ``{"role": <str>, "content": <str>}`` dicts as
    produced by ``[{"role": m.type, "content": m.content} for m in
    prompt.format_messages()]`` (LangChain) or any other planner.
    Returns a NEW list of dicts whose ``role`` values are guaranteed to
    be in ``{"system", "user", "assistant", "tool"}``. The shape is
    strict: each output dict carries exactly two keys, ``role`` and
    ``content``.

    Mapping:

    * ``human``     -> ``user``
    * ``ai``        -> ``assistant``
    * ``system``    -> ``system`` (pass-through)
    * ``user``      -> ``user`` (pass-through)
    * ``assistant`` -> ``assistant`` (pass-through)
    * ``tool``      -> ``tool`` (pass-through)

    Anything else fails fast with a :class:`ValueError` whose message
    names the offending index, the unknown role, and the closed set of
    allowed roles. The normalizer never silently drops, suppresses, or
    re-types a message — that's a correctness property, not a
    convenience: a quietly-renamed role hides the real bug from the
    operator.

    The function does not mutate its input. The returned list is a
    fresh container of fresh dicts; downstream callers may freely
    further mutate either side.
    """
    if not isinstance(messages, list):
        raise ValueError(
            f"messages must be a list of role/content dicts, got "
            f"{type(messages).__name__}"
        )

    normalized: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(
                f"messages[{i}] must be a dict, got {type(msg).__name__}"
            )
        if "role" not in msg:
            raise ValueError(
                f"messages[{i}] is missing required key 'role'."
            )
        raw_role = msg["role"]
        if not isinstance(raw_role, str) or not raw_role:
            raise ValueError(
                f"messages[{i}].role must be a non-empty string, got "
                f"{raw_role!r}"
            )

        provider_role = _INTERNAL_TO_PROVIDER_ROLE.get(raw_role.strip().lower())
        if provider_role is None or provider_role not in _ALLOWED_PROVIDER_ROLES:
            raise ValueError(
                f"messages[{i}] has unknown role {raw_role!r}; allowed "
                f"provider roles are "
                f"{sorted(_ALLOWED_PROVIDER_ROLES)} "
                f"(internal aliases: {sorted(_INTERNAL_TO_PROVIDER_ROLE)})."
            )

        # Pass `content` through untouched. The structural assert immediately
        # downstream verifies it's a string and that no extra keys leaked.
        normalized.append(
            {"role": provider_role, "content": msg.get("content")}
        )

    return normalized


# ---------------------------------------------------------------------------
# Defense-in-depth: prompt-message structural guard
# ---------------------------------------------------------------------------

# Tokens that, when present in a message payload, indicate a caller has
# attempted to wire an LLM-supplied string into a code-evaluation surface.
# The orchestrator never produces such payloads today; this list exists so a
# future regression in any caller is caught at the boundary rather than at
# pandas. Keys are checked case-insensitively.
_FORBIDDEN_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "pandas_query",
        "df_query",
        "query_string",
        "eval",
        "exec",
        "python_code",
        "shell_command",
        "subprocess",
    }
)

DEFAULT_GROQ_MODEL: str = "llama-3.3-70b-versatile"


def _enforce_canonical_legacy_guard() -> None:
    runtime = get_runtime_context()
    log_runtime_event(
        "legacy_planner_entered",
        service="agent-service",
        trigger=str(runtime.get("trigger", "worker")),
        instruction_present=bool(runtime.get("instruction_present")),
        canonical_intent_present=bool(runtime.get("canonical_intent_present")),
        legacy_schema_state_present=bool(runtime.get("legacy_schema_state_present")),
        model=DEFAULT_GROQ_MODEL,
    )
    if not runtime.get("canonical_intent_present"):
        return

    log_runtime_event(
        "architecture_violation_canonical_job_entered_legacy_planner",
        service="agent-service",
        trigger=str(runtime.get("trigger", "worker")),
        instruction_present=bool(runtime.get("instruction_present")),
        canonical_intent_present=True,
        legacy_schema_state_present=bool(runtime.get("legacy_schema_state_present")),
        model=DEFAULT_GROQ_MODEL,
    )
    if os.environ.get("FAIL_ON_CANONICAL_LEGACY_PLANNER", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "Architecture violation: canonical job entered legacy raw-prompt planner"
        )


def assert_no_eval_strings(messages: Any) -> None:
    """Verify *messages* is a plain list of ``{"role", "content"}`` dicts
    whose ``content`` values are strings.

    This is the single boundary check that prevents any caller of
    :func:`call_groq_json` from accidentally forwarding a non-string payload
    (e.g. a callable, a dict containing a ``pandas_query`` field, or a
    nested structure that could be interpreted as code) into the LLM
    request, which would otherwise be a vector for a downstream caller to
    later forward LLM-supplied strings to ``pandas.DataFrame.query`` or
    another code-evaluation surface.

    Raises ``ValueError`` on the first violation; the message names the
    offending index and field so the regression is easy to localize.
    """
    if not isinstance(messages, list):
        raise ValueError(
            f"messages must be a list of role/content dicts, got "
            f"{type(messages).__name__}"
        )
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(
                f"messages[{i}] must be a dict, got {type(msg).__name__}"
            )
        extra_keys = set(msg.keys()) - {"role", "content"}
        if extra_keys:
            raise ValueError(
                f"messages[{i}] contains unsupported keys "
                f"{sorted(extra_keys)}; only 'role' and 'content' are allowed."
            )
        # Reject any key (in any casing) that hints at a code-evaluation
        # payload. The standard envelope only has 'role' and 'content', but
        # check explicitly so a future caller using a richer envelope is
        # caught immediately.
        for key in msg.keys():
            if key.lower() in _FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(
                    f"messages[{i}] uses forbidden key {key!r}; LLM-supplied "
                    f"strings must never be routed to a code-evaluation "
                    f"surface."
                )
        content = msg.get("content")
        if not isinstance(content, str):
            raise ValueError(
                f"messages[{i}].content must be a str, got "
                f"{type(content).__name__}; LLM prompts must be plain text."
            )


def call_groq_json(messages: list, schema: dict) -> dict:
    """Call Groq with structured JSON output and return the parsed object.

    The returned dictionary is parsed JSON only; the caller is responsible
    for validating it against a Pydantic model before using any field. No
    field of the returned object may be forwarded to ``pandas.DataFrame.query``
    or any other code-evaluation surface (see module docstring).

    ``messages`` is normalized at the LLM/provider boundary BEFORE being
    sent to Groq. Internal/framework role tokens (``human``, ``ai``) are
    translated to Groq's wire-format roles (``user``, ``assistant``); see
    :func:`normalize_outbound_messages`. Unknown roles fail fast with a
    clear :class:`ValueError` rather than producing the opaque Groq HTTP
    400 response ``'messages.N' : discriminator property 'role' has
    invalid value``. The structural defense-in-depth check
    :func:`assert_no_eval_strings` then runs against the normalized
    payload so any error messages it surfaces reference the wire-format
    role the provider would actually see.
    """
    # 1. Translate framework roles to provider wire roles (boundary fix).
    #    LangChain's ``BaseMessage.type`` emits ``human``/``ai`` while Groq
    #    expects ``user``/``assistant``; without this step planning fails
    #    with a 400 on every request that includes a HumanMessage.
    _enforce_canonical_legacy_guard()
    normalized_messages = normalize_outbound_messages(messages)

    # 2. Structural defense-in-depth check on the outbound prompt. Any
    #    caller that mistakenly tries to forward a non-string payload trips
    #    this guard before the request leaves the process.
    assert_no_eval_strings(normalized_messages)

    # --- Telemetry: log call start ---
    _telemetry_ctx = None
    try:
        from finflow_agent.llm_telemetry import log_llm_started, log_llm_completed, log_llm_failed
        _telemetry_ctx = log_llm_started(
            service="agent-service",
            operation="plan_generation",
            caller_file="llm.py",
            caller_function="call_groq_json",
            model=DEFAULT_GROQ_MODEL,
            api_key_source="GROQ_API_KEY",
            api_key=os.environ.get("GROQ_API_KEY", ""),
            attempt=1,
            trigger="call_groq_json",
            messages=normalized_messages,
        )
    except Exception:
        pass
    # --- End telemetry start ---

    client = get_groq_client()

    try:
        chat_completion = client.chat.completions.create(
            messages=normalized_messages,
            model=DEFAULT_GROQ_MODEL,
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as exc:
        # --- Telemetry: log failure ---
        if _telemetry_ctx:
            try:
                log_llm_failed(
                    _telemetry_ctx,
                    status_code=getattr(exc, "status_code", 0),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception:
                pass
        # --- End telemetry failure ---
        raise

    # --- Telemetry: log success ---
    if _telemetry_ctx:
        try:
            usage = chat_completion.usage
            log_llm_completed(
                _telemetry_ctx,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                finish_reason=getattr(chat_completion.choices[0], "finish_reason", "") if chat_completion.choices else "",
            )
        except Exception:
            pass
    # --- End telemetry success ---

    import json

    return json.loads(chat_completion.choices[0].message.content)


__all__ = [
    "assert_no_eval_strings",
    "call_groq_json",
    "get_chat_groq",
    "get_groq_client",
    "get_structured_plan_intent_chain",
    "normalize_outbound_messages",
]

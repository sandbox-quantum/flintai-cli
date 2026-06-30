"""
reasoner.py — Agentic Reasoning Loop (v2)
==========================================
Implements the agentic scanning loop using Google ADK.

ADK handles the internal reasoning loop — tool selection, invocation,
and termination. We supply the agent definition, tools, and system prompt.
ADK's Runner executes locally (no Vertex AI deployment required).

SESSION STORAGE:
  Uses InMemorySessionService — sessions are ephemeral and discarded when
  the process exits. This is appropriate because findings are accumulated
  in-memory via the ToolDispatcher (not from session state) and no
  downstream code reads sessions back after a scan completes.

MULTI-PROVIDER SUPPORT:
  Model selection is handled via llm_provider.make_model() which reads
  the SCANNER_MODEL env var (provider:model format):
    google            → bare Gemini model string (e.g. "gemini-3.5-flash")
    all others        → LiteLlm("provider/model") via llm_provider

CONTROL PLANE (ASI compliance for our own scanner):
  MAX_ITERATIONS     = 20   — hard cap on tool call rounds (ASI08)
  MAX_FILES_FETCHED  = 50   — limit on files the agent can read (ASI08)
  LOOP_TIMEOUT_SECS  = 600  — wall-clock timeout (ASI08)

OUTPUT:
  Identical format to v1: (findings_list, summary_string, trace_dict)
  The rest of the pipeline (triage, report assembly) is unchanged.

Canonical import path: from agent_scanner.reasoner import run_agentic_reasoning
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import time
import uuid
from typing import Any

from flintai.schema import RepoFile
from flintai.scan.schema import AgentProfile, RawFinding
from flintai.scan.secret_anonymizer import anonymize_secrets
from flintai.scan.static_scanner import StaticFinding
from flintai.scan.taxonomy import AGENT_TAXONOMY
from flintai.scan.tool_dispatcher import ToolDispatcher
from flintai.scan.trace_logger_file import FileTraceLogger
from flintai.scan.trace_logger_log import LogTraceLogger
from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from flintai.scan.llm_provider import _safe_error, make_model

logger = logging.getLogger(__name__)

# ── Operational limits ────────────────────────────────────────────────────────
# All three limits are configurable via environment variables.
# The values below are the production defaults — override per-deployment
# without code changes or redeployment.
#
#   ADK_MAX_ITERATIONS    — max tool-call rounds before forced stop      (default: 20)
#   ADK_MAX_FILES_FETCHED — max distinct files the agent may read        (default: 50)
#   ADK_MAX_FETCH_TOKENS  — token budget for all fetch_file content      (default: 200000)
#   ADK_LOOP_TIMEOUT_SECS — wall-clock timeout for the full ADK loop     (default: 600)
MAX_ITERATIONS = int(os.getenv("ADK_MAX_ITERATIONS", "20"))
MAX_FILES_FETCHED = int(os.getenv("ADK_MAX_FILES_FETCHED", "50"))
MAX_FETCH_TOKENS = int(os.getenv("ADK_MAX_FETCH_TOKENS", "200000"))
LOOP_TIMEOUT_SECS = int(os.getenv("ADK_LOOP_TIMEOUT_SECS", "600"))


# ── System prompt ─────────────────────────────────────────────────────────────


def _build_system_prompt() -> str:
    taxonomy_lines = []
    for _cat_key, cat_data in AGENT_TAXONOMY.items():
        taxonomy_lines.append(
            f"\n{cat_data['asi_code']} — {cat_data['asi_title']}: {cat_data['description']}"
        )
        for subcat_key, subcat_data in cat_data["subcategories"].items():
            taxonomy_lines.append(
                f"  [{subcat_data['severity'].upper()}] {subcat_key}: "
                f"{subcat_data['title']} — {subcat_data['description']}"
            )
    taxonomy_ref = "\n".join(taxonomy_lines)

    return f"""You are an automated code scanner. You execute a fixed checklist
against source code. You do not improvise, interpret, or add judgment.
You match patterns and report them. Nothing else.

TAXONOMY:
{taxonomy_ref}

PROCEDURE:

1. Call read_source(resource_type="file", path=<path>) for EVERY file
   listed in the AGENT PROFILES. Wait until all files are fetched.

2. Process the checklist below. Go line by line through each file.
   When you find a pattern match, call compute_cvss() then immediately
   call report_finding(). Do not batch. Do not skip. Do not summarize.

3. After processing all patterns for all files, output ONLY:
   {{"summary": "N findings reported", "investigation_notes": "done"}}

CHECKLIST — match these exact patterns in the source code:

PATTERN 1: Variable assigned a string literal where the variable name
contains KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL, or DATABASE_URL.
  Match: SOME_API_KEY = "any-string-here"
  Report: category=asi03_identity_privilege_abuse,
          subcategory=hardcoded_credentials

PATTERN 2: An f-string or string concatenation that puts a variable
into text sent to an LLM (in instruction=, prompt=, system=, or
messages content).
  Match: instruction=f"...some variable..."
  Match: prompt = "..." + user_input
  Report: category=asi01_agent_goal_hijack,
          subcategory=direct_prompt_injection

PATTERN 3: A call to eval(), exec(), or compile() on any variable.
  Match: eval(anything)
  Match: exec(anything)
  Report: category=asi05_unexpected_code_execution,
          subcategory=arbitrary_code_execution

PATTERN 4: subprocess.run/call/Popen with shell=True, or os.system().
  Match: subprocess.run(cmd, shell=True)
  Match: os.system(cmd)
  Report: category=asi05_unexpected_code_execution,
          subcategory=arbitrary_code_execution

PATTERN 5: pickle.loads(), pickle.load(), or yaml.load() without
SafeLoader.
  Match: pickle.loads(data)
  Match: yaml.load(data)
  Report: category=asi05_unexpected_code_execution,
          subcategory=unsafe_deserialization

PATTERN 6: A function that opens a file path parameter without
checking that the path is within a safe directory (no Path.resolve,
no startswith, no allowlist).
  Match: open(path, "r") with no path validation above it
  Report: category=asi02_tool_misuse, subcategory=path_traversal

PATTERN 7: An agent created without max_iterations, max_turns, or
recursion_limit. A while loop with no iteration counter.
  Match: LoopAgent(...) without max_iterations=
  Match: while resp.stop_reason == "tool_use": (no counter)
  Report: category=asi08_cascading_failures,
          subcategory=unbounded_agent_loop

PATTERN 8: An agent with sub_agents= but without
before_agent_callback= or after_agent_callback=.
  Match: Agent(..., sub_agents=[...]) without callbacks
  Report: category=asi10_rogue_agents,
          subcategory=unchecked_agent_delegation

PATTERN 9: Destructive operations (file write, delete, DB update,
shell command) with no human approval check before them.
  Match: open(path, "w") with no approval gate
  Match: os.remove(path) with no confirmation
  Report: category=asi09_human_agent_trust_exploitation,
          subcategory=missing_action_confirmation

PATTERN 10: Global mutable state shared across function calls with
no session isolation.
  Match: GLOBAL_LIST = [] at module level, modified in functions
  Report: category=asi06_memory_context_poisoning,
          subcategory=cross_session_contamination

REPORTING RULES:
- Call compute_cvss(vuln_type=<subcategory>) before EVERY report.
- Set evidence to the EXACT code line, not a description.
- Set evidence_file to the file path, evidence_line to the line number.
- Set agent_name to the agent display name from AGENT PROFILES.
- Set confidence="high" for every direct code match.
- Set hallucination_flag=false for every direct code match.
- Do NOT report the same pattern twice on the same line.
- Do NOT report patterns already found by static analysis
  (check via get_findings() first).
- Do NOT add findings beyond the 10 patterns above.

EXAMPLE:
  Code at line 33 of agent.py: API_KEY = "**********************"
  Step 1: compute_cvss(vuln_type="hardcoded_credentials")
  Step 2: report_finding(
    category="asi03_identity_privilege_abuse",
    subcategory="hardcoded_credentials",
    title="Hardcoded API Key",
    description="API key hardcoded at line 33.",
    impact="Credential exposed to anyone with repo access.",
    remediation="Use os.environ.get().",
    affected_component="agent.py",
    evidence='API_KEY = "**********************"',
    confidence="high", hallucination_flag=false,
    evidence_file="agent.py", evidence_line=33,
    agent_name="research_agent")

LIMITS: {MAX_ITERATIONS} rounds, {MAX_FILES_FETCHED} files, {MAX_FETCH_TOKENS} tokens.
"""


# ── Initial investigation brief ───────────────────────────────────────────────


def _build_initial_context(
    agents: list[AgentProfile],
    python_files: list[RepoFile],
    static_count: int,
) -> str:
    parts = ["=== INVESTIGATION BRIEF ===\n"]
    parts.append(f"Repository contains {len(python_files)} Python file(s).")
    parts.append(
        f"Pre-computed static scan found {static_count} findings "
        f"(available via get_static_findings_for_file).\n"
    )

    parts.append(f"=== AGENT PROFILES SUMMARY ({len(agents)} agent(s)) ===")
    sorted_agents = sorted(agents, key=lambda a: a.agent_id)
    for a in sorted_agents[:15]:
        graph_info = (
            f" | graph_nodes={a.graph_nodes}" if getattr(a, "graph_nodes", None) else ""
        )
        bounded = (
            f" | bounded={a.has_recursion_limit}"
            if getattr(a, "has_recursion_limit", None) is not None
            else ""
        )
        parts.append(
            f"\nAgent: {a.role or a.agent_id} | ID: {a.agent_id}"
            f" | Framework: {a.framework} | File: {a.source_file}"
            f"{graph_info}{bounded}\n"
            f"  Tools: {a.tools or []}\n"
            f"  LLM: {a.llm or '?'} | Memory: {a.memory} | Delegation: {a.allow_delegation}\n"
            f"  Code preview (use get_agent_profile for full content):\n"
            f"  {anonymize_secrets(a.raw_code[:300].strip())}"
        )

    parts.append("\n=== AVAILABLE FILES ===")
    parts.append("Use read_source(resource_type='list') to see all fetched files.")
    parts.append("Use read_source(resource_type='file', path=...) to read any file.")
    parts.append(
        "\n\nBegin STEP 0 of the SOP now. Fetch the full source code for each "
        "agent file listed above, then proceed through STEPS 1-12 in order."
    )
    return "\n".join(parts)


# ── ADK agent builder ─────────────────────────────────────────────────────────


def _build_adk_agent(adk_model: Any, adk_tools: list[Any]) -> Any:
    """Build and return a Google ADK LlmAgent instance."""
    return LlmAgent(
        name="agent_security_scanner",
        model=adk_model,
        description=(
            "AI agent security researcher that scans codebases for "
            "OWASP ASI01-ASI10 vulnerabilities."
        ),
        instruction=_build_system_prompt(),
        tools=adk_tools,
    )


# ── Session service factory ───────────────────────────────────────────────────


def _create_session_service() -> Any:
    """Create and return an in-memory ADK SessionService."""
    return InMemorySessionService()


# ── Async ADK runner ──────────────────────────────────────────────────────────


async def _run_adk_async(
    agent,
    initial_message: str,
    dispatcher: ToolDispatcher,
    session_id: str,
    tracer,
) -> tuple[str, str]:
    """
    Run the ADK agent via InMemorySessionService-backed Runner.

    Returns (final_text_response, exit_reason).
    Findings are accumulated via dispatcher.session_findings as side-effects
    of report_finding() tool calls made by the agent.
    """
    APP_NAME = "agent_security_scanner"
    USER_ID = "scanner"

    session_service = _create_session_service()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    logger.info("Session created: %s", session.id)

    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text=initial_message)],
    )

    final_text = ""
    exit_reason = "completed"
    iteration = 0
    loop_start = time.monotonic()

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=user_content,
    ):
        # ── Control plane guards ──────────────────────────────────────────

        elapsed = time.monotonic() - loop_start
        if elapsed > LOOP_TIMEOUT_SECS:
            logger.warning(
                "Timeout after %.1fs (limit=%ds)", elapsed, LOOP_TIMEOUT_SECS
            )
            exit_reason = "timeout"
            break

        if hasattr(event, "actions") and event.actions:
            iteration += 1
            tracer.set_iteration(iteration)
            logger.debug("Iteration %d/%d", iteration, MAX_ITERATIONS)
            if iteration >= MAX_ITERATIONS:
                logger.warning("Max iterations (%d) reached", MAX_ITERATIONS)
                exit_reason = "max_iterations"
                break

        if dispatcher._call_counts.get("fetch_file", 0) > MAX_FILES_FETCHED:
            logger.warning("File fetch limit (%d) reached", MAX_FILES_FETCHED)
            exit_reason = "max_files"
            break

        if dispatcher.tokens_consumed > MAX_FETCH_TOKENS:
            logger.warning(
                "Token budget exhausted (%d/%d)",
                dispatcher.tokens_consumed,
                MAX_FETCH_TOKENS,
            )
            exit_reason = "max_tokens"
            break

        if dispatcher._call_counts.get("run_targeted_bandit", 0) > 5:
            logger.warning("Bandit call limit (5) reached")
            exit_reason = "max_bandit_calls"
            break

        if dispatcher._call_counts.get("compute_cvss", 0) > 50:
            logger.warning("compute_cvss call limit (50) reached")
            exit_reason = "max_cvss_calls"
            break

        if hasattr(event, "content") and event.content:
            for part in event.content.parts or []:
                if hasattr(part, "text") and part.text:
                    final_text += part.text

    return final_text, exit_reason


# ── Main entry point ──────────────────────────────────────────────────────────


def run_agentic_reasoning(
    agents: list[AgentProfile],
    python_files: list[RepoFile],
    static_findings: list[StaticFinding],
    output_path: str | None = None,
) -> tuple[list[RawFinding], str, dict[str, Any]]:
    """
    Main entry point for v2 agentic reasoning using Google ADK.

    The ADK model is resolved via llm_provider.make_model() using the
    SCANNER_MODEL env var.

    Session storage uses InMemorySessionService (ephemeral, per-scan).

    Returns:
        (findings_list, summary_string, trace_metadata_dict)
    """
    if not agents and not python_files:
        logger.info("No agents or files found — skipping agentic scan")
        return [], "No agent profiles or files found to analyze.", {}

    # Validate that each agent has the three fields the dispatcher and investigation
    # brief depend on. Check each field value individually — a field is "missing"
    # only when its value is None or an empty string, not when its string value
    # differs from the field name (which was the original bug: the old code did
    # `required_fields - set(values)` which always returned the full set because
    # "researcher" != "agent_id", etc.).
    _REQUIRED = ("agent_id", "framework", "source_file")
    invalid_agents = []
    for agent in agents:
        missing = [f for f in _REQUIRED if not getattr(agent, f, None)]
        if missing:
            logger.warning(
                "[reasoner] Agent '%s' missing required fields: %s — skipping",
                getattr(agent, "agent_id", "?"),
                missing,
            )
            invalid_agents.append(agent)

    if invalid_agents:
        agents = [a for a in agents if a not in invalid_agents]
        logger.warning(
            "[reasoner] Excluded %d agents with missing required fields. "
            "Proceeding with %d valid agents.",
            len(invalid_agents),
            len(agents),
        )

    if not agents and not python_files:
        logger.info("No valid agents or files after validation — skipping agentic scan")
        return [], "No valid agent profiles or files found to analyze.", {}

    try:
        adk_model = make_model()
        logger.info("ADK model: %s", adk_model)
    except (ImportError, ValueError) as e:
        logger.error("ADK model resolution failed: %s", e)
        logger.error("ADK setup failed — cannot proceed without ADK: %s", e)
        return [], f"ADK setup failed — {e}", {"error": str(e)}

    # Set up dispatcher and tracer.
    # IMPORTANT: tracer must be created before build_adk_tools() is called so
    # the tool closures can close over it. Previously the order was reversed,
    # which meant the tracer was never passed in and the trace file showed
    # calls=0 despite the agent making dozens of tool calls.
    repo_files_by_path = {rf.path: rf for rf in python_files}
    dispatcher = ToolDispatcher(
        repo_files=repo_files_by_path,
        agents=agents,
        static_findings=static_findings,
    )

    session_id = f"AGT-{uuid.uuid4().hex[:8].upper()}"
    tracer = (
        FileTraceLogger(session_id=session_id, output_path=output_path)
        if output_path
        else LogTraceLogger()
    )
    tracer.start(provider_model=str(adk_model))

    # Build ADK tools with the tracer now wired in — every tool call will be
    # recorded in the JSONL trace file and counted in dispatcher._call_counts.
    adk_tools = dispatcher.get_adk_tools(tracer=tracer)

    # Build ADK agent
    try:
        adk_agent = _build_adk_agent(adk_model, adk_tools)
    except ImportError as e:
        logger.error("ADK agent build failed: %s", e)
        return [], f"ADK agent build failed — {e}", {"error": str(e)}

    # Build initial investigation brief
    initial_ctx = _build_initial_context(agents, python_files, len(static_findings))

    logger.info(
        "Starting ADK scan | session: %s | agents: %d | files: %d",
        session_id,
        len(agents),
        len(python_files),
    )

    # Execute ADK reasoning loop.
    #
    # Always run the coroutine in a dedicated worker thread with a hard
    # deadline. asyncio.run() works in that thread whether or not the caller
    # already has a running event loop (Jupyter/FastAPI as well as the plain
    # CLI path), and future.result(timeout=...) bounds the whole loop even
    # when the model call never returns a control-plane event — the in-loop
    # guard in _run_adk_async only fires between events, so a stalled LLM call
    # would otherwise hang forever on the CLI path. On timeout we shut the
    # pool down without waiting so we don't block on the stuck worker thread.
    exit_reason = "completed"
    final_text = ""

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(
            asyncio.run,
            _run_adk_async(adk_agent, initial_ctx, dispatcher, session_id, tracer),
        )
        final_text, exit_reason = future.result(timeout=LOOP_TIMEOUT_SECS + 30)
    except concurrent.futures.TimeoutError:
        logger.error("ADK execution timed out after %ds", LOOP_TIMEOUT_SECS + 30)
        exit_reason = "timeout"
    except Exception as e:
        err_type = type(e).__name__
        safe_msg = _safe_error(e)
        first_line = safe_msg.split("\n", 1)[0]
        logger.error("ADK execution failed (%s): %s", err_type, first_line)
        logger.debug("ADK execution failed (full):\n%s", safe_msg, exc_info=True)
        exit_reason = "error"
    finally:
        pool.shutdown(wait=False)

    # Collect results
    findings = dispatcher.session_findings
    summary = "Agentic investigation complete."

    if final_text:
        clean = final_text.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(lines[1:]).rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                summary = parsed.get("summary", summary)
        except (json.JSONDecodeError, TypeError):
            if len(final_text.strip()) > 20:
                summary = final_text.strip()

    tracer.finish(findings_count=len(findings), exit_reason=exit_reason)
    trace_meta = tracer.as_dict()
    trace_meta.update(
        {
            "exit_reason": exit_reason,
            "adk_model": str(adk_model),
            "session_id": session_id,
            "tokens_consumed": dispatcher.tokens_consumed,
        }
    )

    logger.info(
        "ADK scan complete — findings: %d | exit: %s | session: %s",
        len(findings),
        exit_reason,
        session_id,
    )

    return findings, summary, trace_meta

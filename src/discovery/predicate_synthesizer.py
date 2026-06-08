"""
Synthesises Python predicate code for a candidate derived factor by prompting
the LLM, then validates the result by running it through the sandbox.

The LLM is expected to return a JSON object with two keys:
  compute_factor_code : the ``compute_factor`` function for sandbox execution
  sweetpea_code       : the SweetPea Factor definition for archival

If the sandbox rejects the code, the error message is fed back to the LLM
and synthesis is retried up to max_retries times.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.discovery.factor_registry import CandidateFactor, DiscoveredFactor
from src.discovery.llm_client import LLMClient
from src.discovery.sandbox import run_predicate

_PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"


def _load(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8")


def _fill(template: str, **subs: str) -> str:
    for key, value in subs.items():
        template = template.replace(f"<<{key}>>", value)
    return template


def _format_discovered_section(discovered: List[DiscoveredFactor]) -> str:
    if not discovered:
        return ""
    lines = ["Already-discovered factor columns (available as trial dict keys):"]
    for d in discovered:
        if d.candidate.factor_class == "continuous":
            lines.append(
                f"  {d.column_name} ({d.candidate.factor_type}, continuous): "
                f"returns float — {d.candidate.description}"
            )
        else:
            levels_str = ", ".join(f'"{lv}"' for lv in d.candidate.levels)
            type_str = d.candidate.factor_type
            if d.candidate.factor_type == "window":
                type_str += f" width={d.candidate.window_width}"
            lines.append(
                f"  {d.column_name} ({type_str}, discrete): "
                f"levels=[{levels_str}]"
            )
    return "\n".join(lines)


def _parse_synthesis_response(raw: str) -> Optional[dict]:
    """
    Extract the JSON object from the LLM response.
    Handles raw JSON and markdown-fenced JSON.
    Returns None if parsing fails.
    """
    text = raw.strip()

    # Try stripping markdown fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        # Take the first {...} block
        bracket = re.search(r"\{.*\}", text, re.DOTALL)
        if bracket:
            text = bracket.group(0)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None

    if "compute_factor_code" not in obj or "sweetpea_code" not in obj:
        return None
    return obj


def synthesize_predicate(
    llm: LLMClient,
    candidate: CandidateFactor,
    working_df: pd.DataFrame,
    discovered: List[DiscoveredFactor],
    max_retries: int = 3,
    temperature: float = 0.2,
    timeout_seconds: int = 10,
    backend: str = "subprocess",
    max_tokens: int = 1200,
    observable_factor_descriptions: str = "",
) -> Optional[str]:
    """
    Ask the LLM to write a ``compute_factor`` function for ``candidate``,
    validate it via the sandbox, and retry with error feedback on failure.

    On success:
        - Sets ``candidate.sweetpea_code`` with the archival SweetPea definition.
        - Sets ``candidate.predicate_status`` to ``"valid"``.
        - Returns the ``compute_factor`` source string.

    On failure after all retries:
        - Sets ``candidate.predicate_status`` to ``"synthesis_failed"``.
        - Returns ``None``.
    """
    system = _load("predicate_synthesis_system.txt")
    user_template = _load("predicate_synthesis_user.txt")
    previous_error: Optional[str] = None

    for attempt in range(max_retries):
        error_section = (
            f"\n## Previous attempt failed (attempt {attempt})\n"
            f"Your previous code raised this error — please fix it:\n"
            f"```\n{previous_error}\n```\n"
            if previous_error else ""
        )

        window_width_line = (
            f"Window width: {candidate.window_width}"
            if candidate.factor_type == "window" else ""
        )
        levels_str = (
            str(candidate.levels)
            if candidate.factor_class == "discrete"
            else "(continuous — return a float, no declared levels)"
        )

        user = _fill(
            user_template,
            name=candidate.name,
            factor_type=candidate.factor_type,
            factor_class=candidate.factor_class,
            window_width_line=window_width_line,
            description=candidate.description,
            levels=levels_str,
            depends_on=str(candidate.depends_on),
            observable_factor_descriptions=observable_factor_descriptions,
            discovered_section=_format_discovered_section(discovered),
            error_section=error_section,
        )

        raw = llm.complete(system=system, user=user,
                           max_tokens=max_tokens, temperature=temperature)

        artifacts = _parse_synthesis_response(raw)
        if artifacts is None:
            previous_error = (
                f"Could not parse your response as a JSON object with keys "
                f"'compute_factor_code' and 'sweetpea_code'.\n"
                f"Your response started with:\n{raw[:300]}"
            )
            continue

        compute_code: str = artifacts["compute_factor_code"]
        sweetpea_code: str = artifacts["sweetpea_code"]

        # 1. Syntax check
        try:
            compile(compute_code, "<compute_factor>", "exec")
        except SyntaxError as exc:
            previous_error = f"SyntaxError in compute_factor_code: {exc}"
            candidate.predicate_status = "syntax_error"
            continue

        # 2. Sandbox validation — use a small participant sample so the
        # validation call stays fast even on large empirical datasets.
        # Synthesis only needs to confirm the code runs correctly and returns
        # the right types/levels; it does not need the full dataset.
        _MAX_SYNTHESIS_PARTICIPANTS = 50
        pids = list(working_df["participant_id"].unique())
        if len(pids) > _MAX_SYNTHESIS_PARTICIPANTS:
            rng = random.Random(attempt)
            sample_pids = set(rng.sample(pids, _MAX_SYNTHESIS_PARTICIPANTS))
            validation_df = working_df[working_df["participant_id"].isin(sample_pids)]
        else:
            validation_df = working_df

        sandbox = run_predicate(
            predicate_code=compute_code,
            df=validation_df,
            factor_type=candidate.factor_type,
            window_width=candidate.window_width,
            timeout_seconds=timeout_seconds,
            backend=backend,
            depends_on=candidate.depends_on,
        )
        if not sandbox.success:
            previous_error = (
                f"Sandbox execution failed ({sandbox.error_type}):\n"
                f"{sandbox.error_message}"
            )
            candidate.predicate_status = sandbox.error_type
            continue

        # 3. Return-type check (branches by factor_class)
        if candidate.factor_class == "continuous":
            bad = [v for v in sandbox.values if v is not None and not isinstance(v, (int, float))]
            if bad:
                previous_error = (
                    f"compute_factor returned non-numeric values "
                    f"(e.g. {bad[:2]}). For a continuous factor it must return a float."
                )
                candidate.predicate_status = "type_error"
                continue
        else:
            bad = [v for v in sandbox.values if v is not None and not isinstance(v, str)]
            if bad:
                previous_error = (
                    f"compute_factor returned non-string values "
                    f"(e.g. {bad[:2]}). It must return a string."
                )
                candidate.predicate_status = "type_error"
                continue

            # 4. Level membership check (discrete only)
            returned = {v for v in sandbox.values if v is not None}
            undeclared = returned - set(candidate.levels)
            if undeclared:
                previous_error = (
                    f"compute_factor returned level values not in the declared "
                    f"levels {candidate.levels}: {undeclared}"
                )
                candidate.predicate_status = "type_error"
                continue

        # All checks passed
        candidate.sweetpea_code = sweetpea_code
        candidate.compute_code = compute_code
        candidate.predicate_status = "valid"
        return compute_code

    candidate.predicate_status = "synthesis_failed"
    return None

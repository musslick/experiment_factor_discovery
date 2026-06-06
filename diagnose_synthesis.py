"""
Step-by-step diagnosis of predicate synthesis failures.

Runs a single synthesis attempt for a chosen factor and prints the result of
every intermediate step so you can pinpoint exactly where things go wrong:
  1. Sandbox self-test     — verifies the backend works with a trivial predicate
  2. Prompt building       — shows the exact user prompt sent to the LLM
  3. LLM call              — prints the raw response
  4. JSON parsing          — shows what was extracted
  5. Syntax check          — compiles the extracted code
  6. Sandbox execution     — runs the code against real trial data
  7. Output validation     — checks types and level membership

Usage:
    python diagnose_synthesis.py                         # congruency (within_trial)
    python diagnose_synthesis.py --factor task_transition
    python diagnose_synthesis.py --model claude-opus-4-8
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from src.discovery.factor_registry import CandidateFactor
from src.discovery.llm_client import LLMClient
from src.discovery.sandbox import run_predicate
from src.utils.config import load_config

# ---------------------------------------------------------------------------
# Catalogue of factors to diagnose
# ---------------------------------------------------------------------------

CANDIDATES = {
    "congruency": CandidateFactor(
        name="congruency",
        description="Whether the ink colour matches the word meaning",
        factor_type="within_trial",
        levels=["congruent", "incongruent"],
        depends_on=["color", "word"],
    ),
    "task_transition": CandidateFactor(
        name="task_transition",
        description="Whether the task repeated or switched from the previous trial",
        factor_type="transition",
        levels=["repeat", "switch"],
        depends_on=["task"],
    ),
}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sep(title: str) -> None:
    width = 62
    print(f"\n{'─' * width}")
    print(f"  Step: {title}")
    print(f"{'─' * width}")

def _ok(msg: str)   -> None: print(f"  \033[32m✓\033[0m  {msg}")
def _fail(msg: str) -> None: print(f"  \033[31m✗\033[0m  {msg}")
def _info(msg: str) -> None: print(f"  \033[90m·\033[0m  {msg}")

def _block(label: str, text: str) -> None:
    print(f"\n  ┌── {label} {'─' * max(0, 54 - len(label))}┐")
    for line in text.splitlines():
        print(f"  │  {line}")
    print(f"  └{'─' * 58}┘")

# ---------------------------------------------------------------------------
# Small Stroop DataFrame (same shape as pipeline's working_df)
# ---------------------------------------------------------------------------

def _make_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(2):
        for t in range(18):
            rows.append({
                "participant_id": pid,
                "trial_index":    t,
                "task":           rng.choice(["color_naming", "word_reading"]),
                "color":          rng.choice(["red", "blue", "green"]),
                "word":           rng.choice(["red", "blue", "green"]),
                "correct":        int(rng.random() > 0.35),
            })
    return pd.DataFrame(rows).reset_index(drop=True)

# ---------------------------------------------------------------------------
# JSON extraction (mirrors predicate_synthesizer._parse_synthesis_response)
# ---------------------------------------------------------------------------

def _try_parse(raw: str):
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        _info("Response is fenced — extracting inner block")
        text = fence.group(1)
    else:
        bracket = re.search(r"\{.*\}", text, re.DOTALL)
        if bracket:
            text = bracket.group(0)
        else:
            return None, "No JSON object ({...}) found in response"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"json.JSONDecodeError: {exc}\n\nText attempted:\n{text[:400]}"

# ---------------------------------------------------------------------------
# Main diagnosis routine
# ---------------------------------------------------------------------------

def diagnose(factor_name: str, model: str, backend: str) -> bool:
    candidate = CANDIDATES[factor_name]
    df        = _make_df()

    print(f"\nDiagnosing synthesis for:  {factor_name!r}  ({candidate.factor_type})")
    print(f"Model  : {model}")
    print(f"Backend: {backend}")
    print(f"DataFrame shape: {df.shape}, columns: {list(df.columns)}")

    llm = LLMClient(model=model)

    # ------------------------------------------------------------------
    # Step 1: Sandbox self-test
    # ------------------------------------------------------------------
    _sep("1 / 7  Sandbox self-test")
    _trivial = "def compute_factor(trial):\n    return 'ok'"
    sr = run_predicate(_trivial, df, "within_trial", timeout_seconds=10, backend=backend)
    if not sr.success:
        _fail(f"Sandbox backend '{backend}' is broken: "
              f"[{sr.error_type}] {sr.error_message}")
        if backend == "docker":
            print("\n  Hint: install llm-sandbox with  pip install llm-sandbox")
            print("        and make sure the Docker daemon is running.")
        return False
    _ok(f"Backend '{backend}' works — trivial predicate executed in subprocess")

    # ------------------------------------------------------------------
    # Step 2: Prompt building
    # ------------------------------------------------------------------
    _sep("2 / 7  Prompt building")
    prompt_dir   = Path("prompts")
    system       = (prompt_dir / "predicate_synthesis_system.txt").read_text()
    user_tmpl    = (prompt_dir / "predicate_synthesis_user.txt").read_text()

    subs = dict(
        name=candidate.name,
        factor_type=candidate.factor_type,
        description=candidate.description,
        levels=str(candidate.levels),
        depends_on=str(candidate.depends_on),
        discovered_section="",
        error_section="",
    )
    user = user_tmpl
    for k, v in subs.items():
        user = user.replace(f"<<{k}>>", v)

    _block("User prompt", user)
    _ok("Prompts built")

    # ------------------------------------------------------------------
    # Step 3: LLM call
    # ------------------------------------------------------------------
    _sep("3 / 7  LLM call")
    try:
        raw = llm.complete(system=system, user=user, max_tokens=1200, temperature=0.2)
    except Exception as exc:
        _fail(f"LLM call raised: {exc}")
        return False

    _ok(f"Response received ({len(raw)} chars)")
    _block("Raw LLM response", raw)

    # ------------------------------------------------------------------
    # Step 4: JSON parsing
    # ------------------------------------------------------------------
    _sep("4 / 7  JSON parsing")
    obj, err = _try_parse(raw)
    if obj is None:
        _fail(f"Could not extract JSON:\n  {err}")
        return False
    _ok(f"JSON parsed — keys: {list(obj.keys())}")

    for key in ("compute_factor_code", "sweetpea_code"):
        if key not in obj:
            _fail(f"Missing required key '{key}'")
            return False
    _block("compute_factor_code", obj["compute_factor_code"])
    _block("sweetpea_code",       obj["sweetpea_code"])
    code = obj["compute_factor_code"]

    # ------------------------------------------------------------------
    # Step 5: Syntax check
    # ------------------------------------------------------------------
    _sep("5 / 7  Syntax check")
    try:
        compile(code, "<compute_factor>", "exec")
        _ok("No syntax errors")
    except SyntaxError as exc:
        _fail(f"SyntaxError: {exc}")
        return False

    # ------------------------------------------------------------------
    # Step 6: Sandbox execution
    # ------------------------------------------------------------------
    _sep("6 / 7  Sandbox execution")
    sr = run_predicate(code, df, candidate.factor_type,
                       timeout_seconds=10, backend=backend)
    if not sr.success:
        _fail(f"[{sr.error_type}]")
        _block("Error message", sr.error_message or "(no message)")
        return False
    _ok(f"Executed — {len(sr.values)} values returned")
    _info(f"First 10 values: {sr.values[:10]}")

    # ------------------------------------------------------------------
    # Step 7: Output validation
    # ------------------------------------------------------------------
    _sep("7 / 7  Output validation")

    bad_type = [v for v in sr.values if v is not None and not isinstance(v, str)]
    if bad_type:
        _fail(f"Non-string values in output: {bad_type[:5]}")
        return False
    _ok("All values are str or None")

    returned     = {v for v in sr.values if v is not None}
    undeclared   = returned - set(candidate.levels)
    if undeclared:
        _fail(f"Returned values not in declared levels {candidate.levels}: {undeclared}")
        return False
    _ok(f"Level membership check passed: {returned}")

    none_count = sum(1 for v in sr.values if v is None)
    _info(f"None values (transition block-starts): {none_count}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'═' * 62}")
    print(f"  \033[32mAll 7 steps passed — synthesis works end-to-end.\033[0m")
    print(f"{'═' * 62}\n")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config("config/stroop_benchmark.yaml")

    parser = argparse.ArgumentParser(description="Diagnose predicate synthesis")
    parser.add_argument("--factor",  default="congruency",
                        choices=list(CANDIDATES.keys()))
    parser.add_argument("--model",   default=cfg.llm.model)
    parser.add_argument("--backend", default=cfg.discovery.sandbox_backend,
                        choices=["subprocess", "docker"])
    args = parser.parse_args()

    ok = diagnose(args.factor, args.model, args.backend)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

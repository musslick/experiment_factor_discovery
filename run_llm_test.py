"""
Standalone smoke test for the LLM integration (Phase 4).

Runs three checks in sequence and prints pass/fail for each:
  1. API connectivity  — basic completion call
  2. Candidate generation  — proposes factors for the Stroop observable factors
  3. Predicate synthesis (within-trial)  — synthesises a congruency predicate
     and verifies it produces correct values on a small dataset
  4. Predicate synthesis (transition)  — synthesises a task_transition predicate
     and verifies correct values and None placement

Usage:
    python run_llm_test.py [--model claude-sonnet-4-6]
"""

import argparse
import sys
import numpy as np
import pandas as pd

from src.discovery.llm_client import LLMClient, OllamaLLMClient
from src.discovery.factor_registry import CandidateFactor
from src.discovery.candidate_generator import generate_candidates
from src.discovery.predicate_synthesizer import synthesize_predicate
from src.discovery.sandbox import run_predicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m  {msg}")

def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m  {msg}")

def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _make_stroop_df(seed: int = 7) -> pd.DataFrame:
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
# Individual checks
# ---------------------------------------------------------------------------

def check_api(llm: LLMClient) -> bool:
    _section("1 / 4  API connectivity")
    try:
        resp = llm.complete(
            system="Respond with a single word.",
            user="What colour is the sky?",
            max_tokens=20,
        )
        if resp and len(resp.strip()) > 0:
            _ok(f"Got response: {resp.strip()!r}")
            return True
        _fail("Empty response")
        return False
    except Exception as exc:
        _fail(f"API call failed: {exc}")
        return False


def check_candidates(llm: LLMClient) -> bool:
    _section("2 / 4  Candidate generation")
    try:
        candidates = generate_candidates(
            llm=llm,
            observable_factors=["task", "color", "word"],
            discovered_so_far=[],
            rejected_so_far=[],
            round_num=1,
            max_candidates=4,
            temperature=0.7,
        )
        if not candidates:
            _fail("No candidates returned")
            return False
        for c in candidates:
            print(f"       {c.factor_type:14s}  {c.name}  {c.levels}")
        names_text = " ".join(c.name + " " + c.description for c in candidates).lower()
        if "congruen" in names_text or "match" in names_text:
            _ok(f"{len(candidates)} candidate(s) — congruency-like factor proposed")
        else:
            _ok(f"{len(candidates)} candidate(s) returned (no congruency-like factor in this run)")
        return True
    except Exception as exc:
        _fail(f"Candidate generation failed: {exc}")
        return False


def check_predicate_within_trial(llm: LLMClient, df: pd.DataFrame) -> bool:
    _section("3 / 4  Predicate synthesis — within_trial (congruency)")
    candidate = CandidateFactor(
        name="congruency",
        description="Whether the ink colour matches the word meaning",
        factor_type="within_trial",
        levels=["congruent", "incongruent"],
        depends_on=["color", "word"],
    )
    try:
        code = synthesize_predicate(
            llm=llm, candidate=candidate, working_df=df,
            discovered=[], max_retries=3, temperature=0.2,
            timeout_seconds=10, backend="subprocess",
        )
        if code is None:
            _fail(f"Synthesis failed — status: {candidate.predicate_status}")
            return False

        result = run_predicate(code, df, "within_trial")
        if not result.success:
            _fail(f"Sandbox error: {result.error_message}")
            return False

        errors = []
        for i, row in df.iterrows():
            expected = "congruent" if row["color"] == row["word"] else "incongruent"
            if result.values[i] != expected:
                errors.append(i)
        if errors:
            _fail(f"Value mismatch on {len(errors)} row(s): indices {errors[:5]}")
            return False

        _ok(f"All {len(df)} values correct — SweetPea code stored: "
            f"{'yes' if candidate.sweetpea_code else 'no'}")
        return True
    except Exception as exc:
        _fail(f"Exception: {exc}")
        return False


def check_predicate_transition(llm: LLMClient, df: pd.DataFrame) -> bool:
    _section("4 / 4  Predicate synthesis — transition (task_transition)")
    candidate = CandidateFactor(
        name="task_transition",
        description="Whether the task repeated or switched from the previous trial",
        factor_type="transition",
        levels=["repeat", "switch"],
        depends_on=["task"],
    )
    try:
        code = synthesize_predicate(
            llm=llm, candidate=candidate, working_df=df,
            discovered=[], max_retries=3, temperature=0.2,
            timeout_seconds=10, backend="subprocess",
        )
        if code is None:
            _fail(f"Synthesis failed — status: {candidate.predicate_status}")
            return False

        result = run_predicate(code, df, "transition")
        if not result.success:
            _fail(f"Sandbox error: {result.error_message}")
            return False

        errors = []
        for pid in sorted(df["participant_id"].unique()):
            p_df = (df[df["participant_id"] == pid]
                    .sort_values("trial_index")
                    .reset_index())
            for pos in range(len(p_df)):
                orig = int(p_df.loc[pos, "index"])
                val  = result.values[orig]
                if pos == 0:
                    if val is not None:
                        errors.append(f"pid={pid} pos=0 expected None got {val!r}")
                else:
                    prev_task = p_df.loc[pos - 1, "task"]
                    curr_task = p_df.loc[pos,     "task"]
                    expected  = "repeat" if prev_task == curr_task else "switch"
                    if val != expected:
                        errors.append(f"pid={pid} pos={pos} expected {expected!r} got {val!r}")

        if errors:
            _fail(f"{len(errors)} error(s): {errors[:3]}")
            return False

        n_none = sum(1 for v in result.values if v is None)
        _ok(f"All {len(df)} values correct — {n_none} None(s) at participant starts")
        return True
    except Exception as exc:
        _fail(f"Exception: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM integration smoke test")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Model ID (Anthropic model ID or Ollama model name)")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "ollama"],
                        help="LLM backend to use (default: anthropic)")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434",
                        help="Ollama server URL (only used when --provider ollama)")
    args = parser.parse_args()

    print(f"\nLLM smoke test — provider: {args.provider}  model: {args.model}")
    if args.provider == "ollama":
        llm = OllamaLLMClient(model=args.model, base_url=args.ollama_base_url)
    else:
        llm = LLMClient(model=args.model)
    df  = _make_stroop_df()

    results = [
        check_api(llm),
        check_candidates(llm),
        check_predicate_within_trial(llm, df),
        check_predicate_transition(llm, df),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\n{'─' * 60}")
    colour = "\033[32m" if passed == total else "\033[31m"
    print(f"  {colour}{passed} / {total} checks passed\033[0m")
    print(f"{'─' * 60}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

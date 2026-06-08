"""
Sandboxed execution of LLM-synthesized predicate functions.

Predicate code contract
-----------------------
The code passed to run_predicate must define a function named ``compute_factor``
with one of two signatures depending on factor_type:

  within_trial:  def compute_factor(trial: dict) -> str
  transition:    def compute_factor(prev: dict, curr: dict) -> str

The function must return a string that is one of the candidate's declared level
names.  The harness calls it for every trial (or consecutive trial pair) and
collects the results into a list aligned to the original DataFrame row order.

Transition factors: the first trial of each participant has no predecessor, so
``compute_factor`` is not called for it; the result is set to None instead.

Two backends
------------
subprocess (default)
    Runs a generated harness script in a child Python process.  Data is passed
    via stdin as JSON; output is captured from stdout.  A timeout kills the
    child if the predicate hangs.

docker
    Runs the harness inside a ``python:3.9-slim`` container via llm-sandbox.
    Data is embedded directly in the script (no stdin required).  Requires
    Docker daemon and ``pip install llm-sandbox``.
"""

import json
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    success: bool
    values: Optional[List]       # list of str|None, aligned to df row order
    error_type: str              # "ok"|"syntax_error"|"runtime_error"|"timeout"
    error_message: Optional[str]


# ---------------------------------------------------------------------------
# Harness templates
# ---------------------------------------------------------------------------

# Subprocess harness: reads serialised DataFrame from stdin.
_SUBPROCESS_HARNESS = """\
import json, sys, collections

# --- BEGIN PREDICATE CODE ---
{predicate_code}
# --- END PREDICATE CODE ---

data = json.loads(sys.stdin.buffer.read().decode('utf-8'))
rows         = data['rows']
factor_type  = data['factor_type']
window_width = data.get('window_width', 2)
n            = data['n']

results = [None] * n
by_pid  = collections.defaultdict(list)
for row in rows:
    by_pid[row['participant_id']].append(row)

try:
    for pid in sorted(by_pid.keys()):
        p_rows = sorted(by_pid[pid], key=lambda r: r['trial_index'])
        for i, row in enumerate(p_rows):
            orig = row['__idx__']
            if factor_type == 'within_trial':
                results[orig] = compute_factor(row)
            else:  # window (includes former transition width=2)
                if i < window_width - 1:
                    results[orig] = None
                else:
                    results[orig] = compute_factor(p_rows[i - window_width + 1 : i + 1])
except Exception:
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print(json.dumps(results))
"""

# Docker harness: data is embedded as a Python repr-string (no stdin needed).
_DOCKER_HARNESS = """\
import json, sys, collections

# --- BEGIN PREDICATE CODE ---
{predicate_code}
# --- END PREDICATE CODE ---

data = json.loads({data_repr})
rows         = data['rows']
factor_type  = data['factor_type']
window_width = data.get('window_width', 2)
n            = data['n']

results = [None] * n
by_pid  = collections.defaultdict(list)
for row in rows:
    by_pid[row['participant_id']].append(row)

try:
    for pid in sorted(by_pid.keys()):
        p_rows = sorted(by_pid[pid], key=lambda r: r['trial_index'])
        for i, row in enumerate(p_rows):
            orig = row['__idx__']
            if factor_type == 'within_trial':
                results[orig] = compute_factor(row)
            else:  # window (includes former transition width=2)
                if i < window_width - 1:
                    results[orig] = None
                else:
                    results[orig] = compute_factor(p_rows[i - window_width + 1 : i + 1])
except Exception:
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print(json.dumps(results))
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_df(
    df: pd.DataFrame,
    factor_type: str,
    window_width: int = 2,
    keep_cols: Optional[List[str]] = None,
) -> dict:
    """
    Serialise df to a JSON-safe dict consumed by the harness.

    Uses pandas' own JSON serialiser (which converts numpy scalars and renders
    NaN as null) and then immediately deserialises to get pure Python types.
    ``__idx__`` is the row's position in the reset-index DataFrame so the
    harness can write results back in original row order.

    keep_cols: if provided, only these columns are serialised (in addition to
    participant_id and trial_index which are always required by the harness).
    Filtering here reduces the JSON payload size and subprocess parse time.
    """
    if keep_cols is not None:
        required = {"participant_id", "trial_index"}
        cols = [c for c in keep_cols if c in df.columns]
        cols = list(dict.fromkeys([c for c in required if c in df.columns] + cols))
        df = df[cols]
    json_str = df.reset_index(drop=True).to_json(orient="records", default_handler=str)
    records = json.loads(json_str)
    for i, row in enumerate(records):
        row["__idx__"] = i
    return {"rows": records, "factor_type": factor_type, "window_width": window_width, "n": len(df)}


def _classify_error(stderr: str, returncode: int) -> tuple:
    if "SyntaxError" in stderr:
        return "syntax_error", stderr.strip()
    return "runtime_error", stderr.strip()


# ---------------------------------------------------------------------------
# Subprocess backend
# ---------------------------------------------------------------------------

def _run_subprocess(
    predicate_code: str,
    df: pd.DataFrame,
    factor_type: str,
    timeout_seconds: int,
    window_width: int = 2,
    keep_cols: Optional[List[str]] = None,
) -> SandboxResult:
    harness_src = _SUBPROCESS_HARNESS.format(predicate_code=predicate_code)
    data_payload = json.dumps(_serialize_df(df, factor_type, window_width, keep_cols))

    fd, harness_path = tempfile.mkstemp(suffix=".py", prefix="sp_harness_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(harness_src)

        try:
            proc = subprocess.run(
                [sys.executable, harness_path],
                input=data_payload.encode("utf-8"),
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                values=None,
                error_type="timeout",
                error_message=f"Predicate exceeded {timeout_seconds}s timeout",
            )
    finally:
        os.unlink(harness_path)

    stderr = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        etype, emsg = _classify_error(stderr, proc.returncode)
        return SandboxResult(success=False, values=None,
                             error_type=etype, error_message=emsg)

    try:
        values = json.loads(proc.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return SandboxResult(success=False, values=None,
                             error_type="runtime_error",
                             error_message=f"Could not parse harness output: {exc}")

    return SandboxResult(success=True, values=values,
                         error_type="ok", error_message=None)


# ---------------------------------------------------------------------------
# Docker backend
# ---------------------------------------------------------------------------

def _run_docker(
    predicate_code: str,
    df: pd.DataFrame,
    factor_type: str,
    timeout_seconds: int,
    window_width: int = 2,
    keep_cols: Optional[List[str]] = None,
) -> SandboxResult:
    try:
        from llm_sandbox import SandboxSession  # type: ignore
    except ImportError:
        return SandboxResult(
            success=False, values=None, error_type="runtime_error",
            error_message=(
                "llm-sandbox is not installed. "
                "Run: pip install llm-sandbox"
            ),
        )

    data_json = json.dumps(_serialize_df(df, factor_type, window_width, keep_cols))
    harness_src = _DOCKER_HARNESS.format(
        predicate_code=predicate_code,
        data_repr=repr(data_json),      # safely embeds JSON as a Python string literal
    )

    try:
        with SandboxSession(lang="python", verbose=False) as session:
            result = session.run(harness_src)
    except Exception as exc:
        return SandboxResult(success=False, values=None,
                             error_type="runtime_error",
                             error_message=f"Docker session error: {exc}")

    stderr = result.stderr or ""
    if result.exit_code != 0:
        etype, emsg = _classify_error(stderr, result.exit_code)
        return SandboxResult(success=False, values=None,
                             error_type=etype, error_message=emsg)

    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return SandboxResult(success=False, values=None,
                             error_type="runtime_error",
                             error_message=f"Could not parse harness output: {exc}")

    return SandboxResult(success=True, values=values,
                         error_type="ok", error_message=None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_predicate(
    predicate_code: str,
    df: pd.DataFrame,
    factor_type: str,
    timeout_seconds: int = 10,
    backend: str = "subprocess",
    window_width: int = 2,
    depends_on: Optional[List[str]] = None,
) -> SandboxResult:
    """
    Execute ``predicate_code`` against ``df`` and return level assignments.

    Parameters
    ----------
    predicate_code  : Python source defining ``compute_factor``.
    df              : Observable trial DataFrame (participant_id, trial_index,
                      task, color, word, and any previously discovered columns).
    factor_type     : ``"within_trial"`` or ``"window"``.
                      ``"transition"`` is accepted as a backward-compatible alias
                      for ``"window"`` with ``window_width=2``.
    timeout_seconds : Execution timeout (seconds).
    backend         : ``"subprocess"`` (default) or ``"docker"``.
    window_width    : Number of consecutive trial dicts passed to compute_factor
                      for window factors (ignored for within_trial).
    depends_on      : Column names that compute_factor actually reads.  When
                      provided, only these columns (plus the mandatory
                      participant_id and trial_index) are serialised and sent
                      to the subprocess, reducing payload size and parse time.

    Returns
    -------
    SandboxResult
        .values is a list of length ``len(df)``, aligned to the original row
        order.  Entries are the raw return values of compute_factor or ``None``
        (first window_width-1 trials of each participant for window factors).
    """
    # Normalize legacy "transition" → "window" with width=2
    if factor_type == "transition":
        factor_type = "window"
        window_width = 2

    keep_cols = list(depends_on) if depends_on is not None else None

    if backend == "docker":
        return _run_docker(predicate_code, df, factor_type, timeout_seconds, window_width, keep_cols)
    return _run_subprocess(predicate_code, df, factor_type, timeout_seconds, window_width, keep_cols)

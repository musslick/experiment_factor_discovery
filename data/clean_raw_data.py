"""
Cleans Rohdaten_preprocessed_inkl_inkr.csv and writes the result to
data/empirical/stroop_categorization_clean.csv

High-priority changes
---------------------
- Drop empty columns: Cue, meta, Geschlecht, Haendigkeit
- Drop constant columns: Effort_Trial, sender
- Recode correct 1.0->1 / 2.0->0, cast to int
- Recode NachFehler 1.0->0 / 2.0->1, keep NaN for first trial, rename post_error
- Recode Congruency "congruent"->1 / "incongruent"->0, rename congruent
- Recode Distractor "within"->1 / "between"->0, rename within_task_distractor
- Drop CR_H (identical to correctResponse) and all CR_T_* columns

Medium-priority changes
-----------------------
- Cast Trial to int
- Rename duration->rt_ms, set negative RTs to NaN, add valid_rt flag
- Remove 512 intermixed-block rows (single participant 1829) and drop type column
- Rename columns to English snake_case
- Recode task_transition / distractor_transition 0->NaN, 1->"repeat", 2->"switch"
  (NaN for block-start trials; pipeline excludes them via shared NaN mask, matching
  synthetic benchmark convention in sweetpea_builder.py / model_comparison.py)
"""

import csv
import math
import os

RAW = os.path.join(os.path.dirname(__file__), "raw", "Rohdaten_preprocessed_inkl_inkr.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "empirical")
OUT = os.path.join(OUT_DIR, "stroop_categorization_clean.csv")

RT_MIN = 100   # ms — faster responses flagged as invalid
RT_MAX = 3000  # ms — slower responses flagged as invalid


def recode_binary(value, true_val, false_val):
    """Return 1 if value matches true_val, 0 if false_val, else None."""
    if value == str(true_val) or value == true_val:
        return 1
    if value == str(false_val) or value == false_val:
        return 0
    return None


def parse_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


OUTPUT_COLUMNS = [
    "participant_id",
    "trial",
    "task",
    "block_condition",
    "task_transition",
    "distractor_transition",
    "distractor_type_prev",
    "word",
    "image",
    "congruent",
    "within_task_distractor",
    "correct_response",
    "response",
    "accuracy",
    "post_error",
    "rt_ms",
    "valid_rt",
    "CR_H_Baum",
    "CR_H_Blume",
    "CR_H_Fisch",
    "CR_H_Vogel",
]

rows_written = 0
rows_skipped = 0

with open(RAW, newline="", encoding="utf-8") as fin, \
     open(OUT, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()

    for row in reader:
        # --- Medium: drop intermixed-block rows (single participant) ---
        if row["Task"] == "2-tasks-intermixed":
            rows_skipped += 1
            continue

        # --- Medium: task as integer ---
        task = int(float(row["Task"])) if row["Task"] else None

        # --- Medium: trial as integer ---
        trial_raw = parse_float(row["Trial"])
        trial = int(trial_raw) if trial_raw is not None else None

        # --- High: recode correct 1->1, 2->0 ---
        correct_raw = parse_float(row["correct"])
        if correct_raw == 1.0:
            accuracy = 1
        elif correct_raw == 2.0:
            accuracy = 0
        else:
            accuracy = None

        # --- High: recode NachFehler 1->0 (not after error), 2->1 (after error) ---
        nach_raw = parse_float(row["NachFehler"])
        if nach_raw == 1.0:
            post_error = 0
        elif nach_raw == 2.0:
            post_error = 1
        else:
            post_error = ""  # first trial of block — no predecessor

        # --- High: recode Congruency ---
        congruent_val = row["Congruency"].strip().lower()
        if congruent_val == "congruent":
            congruent = 1
        elif congruent_val == "incongruent":
            congruent = 0
        else:
            congruent = ""

        # --- High: recode Distractor ---
        distractor_val = row["Distractor"].strip().lower()
        if distractor_val == "within":
            within_task_distractor = 1
        elif distractor_val == "between":
            within_task_distractor = 0
        else:
            within_task_distractor = ""

        # --- Medium: rt_ms, valid_rt ---
        rt = parse_float(row["duration"])
        if rt is None or rt < 0:
            rt_ms = ""
            valid_rt = 0
        else:
            rt_ms = rt
            valid_rt = 1 if RT_MIN <= rt <= RT_MAX else 0

        # --- Distractor_N_1 recode (NaN for first trial of block) ---
        dis_prev_val = row["Distractor_N_1"].strip().lower()
        if dis_prev_val == "within":
            distractor_type_prev = "within"
        elif dis_prev_val == "between":
            distractor_type_prev = "between"
        else:
            distractor_type_prev = ""  # block-start: no predecessor

        # --- Transition columns: 0->NaN (block-start), 1->"repeat", 2->"switch" ---
        # Matches synthetic benchmark convention; pipeline NaN mask excludes these rows.
        _TRANS_MAP = {"1.0": "repeat", "1": "repeat", "2.0": "switch", "2": "switch"}
        task_transition = _TRANS_MAP.get(row["Transition"].strip(), "")
        distractor_transition = _TRANS_MAP.get(row["DisTrans"].strip(), "")

        out = {
            "participant_id": row["code"],
            "trial": trial,
            "task": task,
            "block_condition": "single",
            "task_transition": task_transition,
            "distractor_transition": distractor_transition,
            "distractor_type_prev": distractor_type_prev,
            "word": row["Word"],
            "image": row["Image"],
            "congruent": congruent,
            "within_task_distractor": within_task_distractor,
            "correct_response": row["correctResponse"],
            "response": row["response"],
            "accuracy": accuracy,
            "post_error": post_error,
            "rt_ms": rt_ms,
            "valid_rt": valid_rt,
            "CR_H_Baum": row["CR_H_Baum"],
            "CR_H_Blume": row["CR_H_Blume"],
            "CR_H_Fisch": row["CR_H_Fisch"],
            "CR_H_Vogel": row["CR_H_Vogel"],
        }
        writer.writerow(out)
        rows_written += 1

print(f"Done. Rows written: {rows_written}, rows skipped (intermixed): {rows_skipped}")
print(f"Output: {OUT}")

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

import produce_results_synthetic as runner

_MODEL_MATRIX = [
    {"id": "claude", "provider": "anthropic", "model": "claude-test"},
    {
        "id": "qwen",
        "provider": "ollama",
        "model": "qwen-test",
        "ollama_base_url": "http://ollama.test:11434",
    },
]


@pytest.mark.parametrize(
    ("models", "message"),
    [
        ([], "non-empty list"),
        ([{"id": "../bad", "provider": "ollama", "model": "qwen"}], "id must"),
        ([{"id": "one", "provider": "openai", "model": "gpt"}], "provider"),
        (
            [
                {"id": "same", "provider": "anthropic", "model": "claude-a"},
                {"id": "same", "provider": "anthropic", "model": "claude-b"},
            ],
            "Duplicate",
        ),
    ],
)
def test_parse_llm_models_rejects_invalid_matrix(models, message):
    with pytest.raises(ValueError, match=message):
        runner._parse_llm_models(models)


def test_resume_model_identity_must_match():
    benchmark_output = {}
    runner._ensure_model_entry(benchmark_output, _MODEL_MATRIX[1])

    changed = {**_MODEL_MATRIX[1], "ollama_base_url": "http://other:11434"}
    with pytest.raises(ValueError, match="does not match"):
        runner._ensure_model_entry(benchmark_output, changed)


def _run_fake_batch(
    monkeypatch,
    tmp_path,
    llm_models=None,
    *,
    resume=False,
    round_log=None,
    n_runs=2,
    base_seed=7,
    benchmarks=("stroop_simon",),
    execution_fail_models=(),
    parse_fail_models=(),
):
    repo_root = Path(__file__).parent.parent
    output_dir = tmp_path / "output"
    config = {
        "benchmarks": list(benchmarks),
        "n_runs": n_runs,
        "base_seed": base_seed,
        "output_dir": str(output_dir),
        "output_name": "results.json",
    }
    if llm_models is not None:
        config["llm_models"] = llm_models

    config_path = tmp_path / "produce.yaml"
    config_path.write_text(yaml.safe_dump(config))

    benchmark_config = str(
        repo_root / "config" / "synthetic_stroop_simon_benchmark.yaml"
    )
    monkeypatch.setattr(
        runner,
        "BENCHMARK_CONFIGS",
        {benchmark: benchmark_config for benchmark in benchmarks},
    )
    monkeypatch.setattr(
        runner,
        "BENCHMARK_DISPLAY",
        {benchmark: benchmark for benchmark in benchmarks},
    )

    calls = []

    def fake_run(cfg, run_dir, regenerate):
        calls.append(
            {
                "provider": cfg.llm.provider,
                "model": cfg.llm.model,
                "ollama_base_url": cfg.llm.ollama_base_url,
                "seed": cfg.seed,
                "run_dir": str(run_dir),
            }
        )
        if cfg.llm.model in execution_fail_models:
            raise RuntimeError(f"execution failed for {cfg.llm.model}")
        report_dir = Path(run_dir) / cfg.name
        report_dir.mkdir(parents=True, exist_ok=True)
        report = (
            {}
            if cfg.llm.model in parse_fail_models
            else {
                "factor_evaluation": {
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                    "n_ground_truth": 2,
                    "n_discovered": 1,
                    "matched_pairs": [],
                    "unmatched_ground_truth": [],
                    "unmatched_discovered": [],
                    "level_recovery": {"levels": []},
                }
            }
        )
        (report_dir / "evaluation_report.json").write_text(json.dumps(report))
        if round_log is not None:
            (report_dir / "round_01_candidates.json").write_text(
                json.dumps(round_log)
            )

    monkeypatch.setattr(runner, "run_single_benchmark", fake_run)
    argv = ["produce_results_synthetic.py", "--config", str(config_path)]
    if resume:
        argv.extend(["--resume", str(output_dir / "results.json")])
    monkeypatch.setattr(sys, "argv", argv)

    runner.main()
    return calls, json.loads((output_dir / "results.json").read_text())


def test_main_runs_matched_model_matrix(monkeypatch, tmp_path):
    calls, output = _run_fake_batch(monkeypatch, tmp_path, _MODEL_MATRIX)

    assert [(call["provider"], call["model"], call["seed"]) for call in calls] == [
        ("anthropic", "claude-test", 7),
        ("anthropic", "claude-test", 8),
        ("ollama", "qwen-test", 7),
        ("ollama", "qwen-test", 8),
    ]
    assert calls[0]["run_dir"].endswith("runs/claude/stroop_simon_seed7")
    assert calls[2]["run_dir"].endswith("runs/qwen/stroop_simon_seed7")
    assert calls[2]["ollama_base_url"] == "http://ollama.test:11434"

    benchmark = output["benchmarks"]["stroop_simon"]
    assert "runs" not in benchmark
    assert set(benchmark["models"]) == {"claude", "qwen"}
    assert {
        model_id: [run["seed"] for run in model["runs"]]
        for model_id, model in benchmark["models"].items()
    } == {"claude": [7, 8], "qwen": [7, 8]}


def test_main_resumes_each_model_and_seed_independently(monkeypatch, tmp_path):
    _, output = _run_fake_batch(monkeypatch, tmp_path, _MODEL_MATRIX)
    output["benchmarks"]["stroop_simon"]["models"]["qwen"]["runs"].pop()
    output_path = tmp_path / "output" / "results.json"
    output_path.write_text(json.dumps(output))

    calls, resumed = _run_fake_batch(
        monkeypatch, tmp_path, _MODEL_MATRIX, resume=True
    )

    assert [(call["model"], call["seed"]) for call in calls] == [("qwen-test", 8)]
    benchmark = resumed["benchmarks"]["stroop_simon"]
    assert {
        model_id: [run["seed"] for run in model["runs"]]
        for model_id, model in benchmark["models"].items()
    } == {"claude": [7, 8], "qwen": [7, 8]}


def test_main_records_no_winner_round_with_null_validation(monkeypatch, tmp_path):
    no_winner = {
        "round": 1,
        "accepted": False,
        "validation_improvement": None,
        "winner": None,
        "all_scored": [],
        "hard_rejected": [],
    }

    _, output = _run_fake_batch(
        monkeypatch, tmp_path, _MODEL_MATRIX[:1], round_log=no_winner
    )

    runs = output["benchmarks"]["stroop_simon"]["models"]["claude"]["runs"]
    assert len(runs) == 2
    assert runs[0]["round_logs"][0]["validation_improvement"] is None


@pytest.mark.parametrize(
    ("execution_fail_models", "parse_fail_models", "stage"),
    [
        (("qwen-test",), (), "benchmark_execution"),
        ((), ("qwen-test",), "result_parsing"),
    ],
)
def test_matrix_records_failures_and_writes_summary(
    monkeypatch, tmp_path, execution_fail_models, parse_fail_models, stage
):
    with pytest.raises(SystemExit) as exc_info:
        _run_fake_batch(
            monkeypatch,
            tmp_path,
            _MODEL_MATRIX,
            n_runs=1,
            execution_fail_models=execution_fail_models,
            parse_fail_models=parse_fail_models,
        )
    assert exc_info.value.code == 1

    output = json.loads((tmp_path / "output" / "results.json").read_text())
    models = output["benchmarks"]["stroop_simon"]["models"]
    assert len(models["claude"]["runs"]) == 1
    assert models["claude"]["failures"] == []
    assert models["qwen"]["runs"] == []
    assert models["qwen"]["failures"][0] == {
        "seed": 7,
        "stage": stage,
        "error": (
            "execution failed for qwen-test"
            if stage == "benchmark_execution"
            else "'factor_evaluation'"
        ),
        "run_dir": str(
            tmp_path / "output" / "runs" / "qwen" / "stroop_simon_seed7"
        ),
    }

    with (tmp_path / "output" / "results_model_summary.csv").open(newline="") as fh:
        summary = {row["model_id"]: row for row in csv.DictReader(fh)}
    assert {
        key: summary["claude"][key]
        for key in (
            "expected", "completed", "failed", "mean_precision", "mean_recall", "mean_f1"
        )
    } == {
        "expected": "1",
        "completed": "1",
        "failed": "0",
        "mean_precision": "0.5",
        "mean_recall": "0.5",
        "mean_f1": "0.5",
    }
    assert {
        key: summary["qwen"][key]
        for key in ("expected", "completed", "failed", "mean_f1")
    } == {"expected": "1", "completed": "0", "failed": "1", "mean_f1": ""}


def test_matrix_resume_success_clears_failure(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_fake_batch(
            monkeypatch,
            tmp_path,
            _MODEL_MATRIX,
            n_runs=1,
            execution_fail_models=("qwen-test",),
        )

    calls, output = _run_fake_batch(
        monkeypatch, tmp_path, _MODEL_MATRIX, n_runs=1, resume=True
    )

    assert [(call["model"], call["seed"]) for call in calls] == [("qwen-test", 7)]
    qwen = output["benchmarks"]["stroop_simon"]["models"]["qwen"]
    assert [run["seed"] for run in qwen["runs"]] == [7]
    assert qwen["failures"] == []


@pytest.mark.parametrize(
    ("initial_models", "resume_models", "message"),
    [
        (None, _MODEL_MATRIX, "single-model"),
        (_MODEL_MATRIX, None, "multi-model"),
        (_MODEL_MATRIX, _MODEL_MATRIX[:1], "different llm_models"),
    ],
)
def test_main_rejects_resume_schema_change(
    monkeypatch, tmp_path, initial_models, resume_models, message
):
    _run_fake_batch(monkeypatch, tmp_path, initial_models)
    copied_config = tmp_path / "output" / "produce.yaml"
    original_config = copied_config.read_text()

    with pytest.raises(ValueError, match=message):
        _run_fake_batch(monkeypatch, tmp_path, resume_models, resume=True)
    assert copied_config.read_text() == original_config


@pytest.mark.parametrize(
    ("initial_models", "resume_models", "message"),
    [
        (None, _MODEL_MATRIX, "single-model"),
        (_MODEL_MATRIX, None, "multi-model"),
    ],
)
def test_main_rejects_cross_schema_in_unselected_benchmark(
    monkeypatch, tmp_path, initial_models, resume_models, message
):
    _run_fake_batch(monkeypatch, tmp_path, initial_models)

    with pytest.raises(ValueError, match=message):
        _run_fake_batch(
            monkeypatch,
            tmp_path,
            resume_models,
            resume=True,
            benchmarks=("rdk",),
        )


@pytest.mark.parametrize(
    ("n_runs", "base_seed", "message"),
    [(3, 7, "n_runs"), (2, 8, "base_seed")],
)
def test_matrix_rejects_changed_run_grid(
    monkeypatch, tmp_path, n_runs, base_seed, message
):
    _run_fake_batch(monkeypatch, tmp_path, _MODEL_MATRIX)

    with pytest.raises(ValueError, match=message):
        _run_fake_batch(
            monkeypatch,
            tmp_path,
            _MODEL_MATRIX,
            resume=True,
            n_runs=n_runs,
            base_seed=base_seed,
        )


def test_legacy_resume_can_extend_run_grid(monkeypatch, tmp_path):
    _run_fake_batch(monkeypatch, tmp_path, n_runs=1)

    calls, output = _run_fake_batch(
        monkeypatch, tmp_path, n_runs=2, resume=True
    )

    assert [call["seed"] for call in calls] == [8]
    assert [
        run["seed"] for run in output["benchmarks"]["stroop_simon"]["runs"]
    ] == [7, 8]


def test_main_without_matrix_keeps_legacy_schema(monkeypatch, tmp_path):
    calls, output = _run_fake_batch(monkeypatch, tmp_path)

    assert len(calls) == 2
    assert "llm_models" not in output
    benchmark = output["benchmarks"]["stroop_simon"]
    assert "models" not in benchmark
    assert [run["seed"] for run in benchmark["runs"]] == [7, 8]
    assert not (tmp_path / "output" / "results_model_summary.csv").exists()

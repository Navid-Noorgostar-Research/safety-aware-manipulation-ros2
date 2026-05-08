"""Tests for the ablation framework.

Two layers:

- The torch-agnostic core (``AblationConfig``, ``AblationStudy``,
  ``AblationResults``, reporters) -- exercised with a tiny synthetic
  evaluator. These tests have no torch dependency.
- The safety-filter evaluator and canonical presets -- skipped if torch
  is not importable, since the filter is a torch ``nn.Module``.
"""

import csv
import io

import pytest

from net.ablation import (
    AblationConfig,
    AblationResults,
    AblationStudy,
    MetricRow,
    to_csv,
    to_markdown,
)


# ---------------------------------------------------------------------------
# Trivial evaluator for core-framework tests (no torch)
# ---------------------------------------------------------------------------


def _toy_evaluator(config, seed):
    """Return a deterministic dict of two metrics keyed off the config."""
    return {
        "scale": float(config.get("scale", 1.0)),
        "seed_value": float(seed),
    }


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------


def test_ablation_config_minimal():
    a = AblationConfig(name="x")
    assert a.name == "x"
    assert a.overrides == {}
    assert a.description == ""


def test_ablation_config_apply_to():
    base = {"a": 1, "b": 2}
    a = AblationConfig(name="x", overrides={"b": 99})
    out = a.apply_to(base)
    assert out == {"a": 1, "b": 99}
    # base is not mutated
    assert base == {"a": 1, "b": 2}


def test_ablation_config_empty_name_raises():
    with pytest.raises(ValueError, match="non-empty"):
        AblationConfig(name="")


def test_ablation_config_whitespace_name_raises():
    with pytest.raises(ValueError, match="whitespace"):
        AblationConfig(name="bad name")


def test_ablation_config_is_frozen():
    """Frozen dataclass: mutation must fail."""
    a = AblationConfig(name="x")
    with pytest.raises(Exception):
        a.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AblationStudy
# ---------------------------------------------------------------------------


def _basic_study():
    return AblationStudy(
        name="toy",
        base_config={"scale": 1.0},
        ablations=[
            AblationConfig("baseline"),
            AblationConfig("doubled", overrides={"scale": 2.0}),
            AblationConfig("zeroed", overrides={"scale": 0.0}),
        ],
        seed=7,
    )


def test_study_run_returns_one_row_per_ablation():
    results = _basic_study().run(_toy_evaluator)
    assert len(results) == 3
    assert [r.name for r in results.rows] == ["baseline", "doubled", "zeroed"]


def test_study_run_preserves_declaration_order():
    results = _basic_study().run(_toy_evaluator)
    assert tuple(r.name for r in results) == ("baseline", "doubled", "zeroed")


def test_study_seeds_per_row_are_stable_and_independent():
    """Each row's seed = base_seed + index, so reordering doesn't shift seeds."""
    results = _basic_study().run(_toy_evaluator)
    seeds = [r.metrics["seed_value"] for r in results]
    assert seeds == [7.0, 8.0, 9.0]


def test_study_run_produces_correct_overrides():
    results = _basic_study().run(_toy_evaluator)
    assert results["baseline"].metrics["scale"] == 1.0
    assert results["doubled"].metrics["scale"] == 2.0
    assert results["zeroed"].metrics["scale"] == 0.0


def test_study_run_is_deterministic_across_calls():
    s = _basic_study()
    r1 = s.run(_toy_evaluator)
    r2 = s.run(_toy_evaluator)
    assert r1.to_dict() == r2.to_dict()


def test_study_validate_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate ablation name"):
        AblationStudy(
            name="x",
            base_config={"a": 1},
            ablations=[AblationConfig("dup"), AblationConfig("dup")],
        )


def test_study_validate_rejects_empty_ablations():
    with pytest.raises(ValueError, match="non-empty"):
        AblationStudy(name="x", base_config={"a": 1}, ablations=[])


def test_study_strict_mode_rejects_unknown_override_key():
    with pytest.raises(ValueError, match="keys not in base_config"):
        AblationStudy(
            name="x",
            base_config={"a": 1},
            ablations=[AblationConfig("bad", overrides={"unknown_key": 0})],
        )


def test_study_strict_off_allows_extension_keys():
    # Should not raise.
    s = AblationStudy(
        name="x",
        base_config={"a": 1},
        ablations=[AblationConfig("ext", overrides={"new_key": 0})],
        strict=False,
    )
    assert s.ablations[0].overrides == {"new_key": 0}


def test_study_materialize_returns_full_configs_per_ablation():
    s = _basic_study()
    mats = s.materialize()
    assert mats["baseline"] == {"scale": 1.0}
    assert mats["doubled"] == {"scale": 2.0}


def test_study_run_rejects_non_dict_evaluator_output():
    s = _basic_study()
    with pytest.raises(TypeError, match="must return a dict"):
        s.run(lambda c, s_: "not a dict")  # type: ignore[arg-type]


def test_study_run_rejects_non_numeric_metric_value():
    s = _basic_study()
    with pytest.raises(TypeError, match="must be numeric"):
        s.run(lambda c, s_: {"x": "string"})  # type: ignore[arg-type]


def test_study_run_rejects_inconsistent_metric_keys():
    s = AblationStudy(
        name="x",
        base_config={"a": 1},
        ablations=[
            AblationConfig("first"),
            AblationConfig("second"),
        ],
    )
    counter = {"i": 0}
    def evaluator(cfg, seed):
        counter["i"] += 1
        return {"a": 1.0} if counter["i"] == 1 else {"a": 1.0, "b": 2.0}
    with pytest.raises(ValueError, match="different metric set"):
        s.run(evaluator)


# ---------------------------------------------------------------------------
# AblationResults
# ---------------------------------------------------------------------------


def test_results_metric_names_taken_from_first_row():
    results = _basic_study().run(_toy_evaluator)
    assert results.metric_names == ("scale", "seed_value")


def test_results_lookup_by_name():
    results = _basic_study().run(_toy_evaluator)
    assert results["doubled"].metrics["scale"] == 2.0


def test_results_lookup_unknown_raises():
    results = _basic_study().run(_toy_evaluator)
    with pytest.raises(KeyError):
        _ = results["nonexistent"]


def test_results_to_dict_round_trip_keys():
    results = _basic_study().run(_toy_evaluator)
    d = results.to_dict()
    assert d["study"] == "toy"
    assert d["seed"] == 7
    assert d["metric_names"] == ["scale", "seed_value"]
    assert len(d["rows"]) == 3
    assert d["rows"][0]["name"] == "baseline"


def test_results_empty_rows_rejected():
    with pytest.raises(ValueError, match="at least one row"):
        AblationResults(study_name="x", rows=[])


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------


def test_markdown_has_header_and_one_row_per_ablation():
    results = _basic_study().run(_toy_evaluator)
    md = to_markdown(results)
    assert "## Ablation: toy" in md
    assert "seed: 7" in md
    # one body row per ablation, plus header + separator
    body_lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert len(body_lines) == 3 + 2  # header, sep, 3 rows


def test_markdown_columns_aligned():
    """All rows must share the same column count."""
    results = _basic_study().run(_toy_evaluator)
    md = to_markdown(results)
    body_lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    counts = [ln.count("|") for ln in body_lines]
    assert len(set(counts)) == 1


def test_markdown_uses_decimal_precision_consistently():
    """Numbers must always show fixed precision (no '1' vs '0.5430' mixing)."""
    results = _basic_study().run(_toy_evaluator)
    md = to_markdown(results, precision=2) if False else to_markdown(results)
    # 'baseline' row scale = 1.0 -> rendered as '1.0000'
    assert "1.0000" in md


def test_csv_round_trip_through_csv_module():
    results = _basic_study().run(_toy_evaluator)
    text = to_csv(results)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["ablation", "scale", "seed_value", "description"]
    assert rows[1][0] == "baseline"
    assert float(rows[1][1]) == 1.0


def test_csv_strips_commas_from_descriptions():
    s = AblationStudy(
        name="x",
        base_config={"a": 1},
        ablations=[AblationConfig("a", description="comma, in description")],
    )
    text = s.run(_toy_evaluator).to_csv()
    rows = list(csv.reader(io.StringIO(text)))
    # description cell should not introduce a stray column
    assert len(rows[1]) == len(rows[0])


def test_to_dict_seed_propagation():
    results = _basic_study().run(_toy_evaluator)
    assert results.to_dict()["seed"] == 7


# ---------------------------------------------------------------------------
# MetricRow helpers
# ---------------------------------------------------------------------------


def test_metric_row_with_meta_returns_new_row():
    r = MetricRow(name="x", metrics={"a": 1.0})
    r2 = r.with_meta(seed=5)
    assert r.meta == {}
    assert r2.meta == {"seed": 5}
    assert r2.metrics is r.metrics or r2.metrics == r.metrics  # frozen, fine


# ---------------------------------------------------------------------------
# Safety-filter evaluator + canonical study (require torch)
# ---------------------------------------------------------------------------


torch = pytest.importorskip("torch")

from net.ablation import (  # noqa: E402
    robot_workspace_study,
    safety_filter_evaluator,
    safety_filter_study,
)


def test_safety_filter_evaluator_reports_expected_metric_keys():
    cfg = safety_filter_study().base_config
    metrics = safety_filter_evaluator(cfg, seed=0)
    expected = {
        "joint_violations",
        "velocity_violations",
        "base_speed_violations",
        "smoothness_violations",
        "collision_violations",
        "any_violations",
        "mean_correction",
    }
    assert set(metrics.keys()) == expected


def test_safety_filter_evaluator_is_deterministic():
    cfg = safety_filter_study().base_config
    a = safety_filter_evaluator(cfg, seed=0)
    b = safety_filter_evaluator(cfg, seed=0)
    assert a == b


def test_safety_filter_evaluator_seed_changes_results():
    cfg = safety_filter_study().base_config
    a = safety_filter_evaluator(cfg, seed=0)
    b = safety_filter_evaluator(cfg, seed=1)
    assert a != b


def test_safety_filter_evaluator_disabled_returns_zeros():
    cfg = dict(safety_filter_study().base_config)
    cfg["enabled"] = False
    metrics = safety_filter_evaluator(cfg, seed=0)
    assert all(v == 0.0 for v in metrics.values())


def test_safety_filter_study_runs_end_to_end():
    s = safety_filter_study(seed=0)
    results = s.run(safety_filter_evaluator)
    names = [r.name for r in results]
    assert names == [
        "full", "no_joint_limits", "no_velocity_limits",
        "no_base_speed", "no_smoothness", "no_collision", "disabled",
    ]


def test_safety_filter_study_no_joint_zeros_joint_metric():
    """The whole point of an ablation row: the disabled constraint is at 0."""
    results = safety_filter_study(seed=0).run(safety_filter_evaluator)
    assert results["no_joint_limits"].metrics["joint_violations"] == 0.0
    # but other constraints still fire on the same data
    assert results["no_joint_limits"].metrics["velocity_violations"] > 0.0


def test_safety_filter_study_no_velocity_zeros_velocity_metric():
    results = safety_filter_study(seed=0).run(safety_filter_evaluator)
    assert results["no_velocity_limits"].metrics["velocity_violations"] == 0.0


def test_safety_filter_study_no_base_speed_zeros_base_speed_metric():
    results = safety_filter_study(seed=0).run(safety_filter_evaluator)
    assert results["no_base_speed"].metrics["base_speed_violations"] == 0.0


def test_safety_filter_study_no_smoothness_zeros_smoothness_metric():
    results = safety_filter_study(seed=0).run(safety_filter_evaluator)
    assert results["no_smoothness"].metrics["smoothness_violations"] == 0.0


def test_safety_filter_study_no_collision_zeros_collision_metric():
    results = safety_filter_study(seed=0).run(safety_filter_evaluator)
    assert results["no_collision"].metrics["collision_violations"] == 0.0


def test_safety_filter_study_full_baseline_fires_each_constraint():
    """The 'full' baseline must trigger all five constraints on the synthetic data."""
    results = safety_filter_study(seed=0).run(safety_filter_evaluator)
    full = results["full"].metrics
    for k in (
        "joint_violations", "velocity_violations", "base_speed_violations",
        "smoothness_violations", "collision_violations",
    ):
        assert full[k] > 0.0, f"baseline did not fire {k}"


def test_safety_filter_study_disabled_zeros_everything():
    results = safety_filter_study(seed=0).run(safety_filter_evaluator)
    disabled = results["disabled"].metrics
    assert all(v == 0.0 for v in disabled.values())


def test_safety_filter_study_correction_is_positive_when_active():
    results = safety_filter_study(seed=0).run(safety_filter_evaluator)
    assert results["full"].metrics["mean_correction"] > 0.0
    assert results["disabled"].metrics["mean_correction"] == 0.0


def test_robot_workspace_study_orders_joint_violations_with_workspace_size():
    """Tighter workspace => more joint-limit violations; looser => fewer.

    (Mean correction is dominated by other constraints on this dataset, so
    we test the cleanest signal -- the per-constraint flag rate.)
    """
    results = robot_workspace_study(seed=0).run(safety_filter_evaluator)
    tight = results["tight"].metrics["joint_violations"]
    default = results["default"].metrics["joint_violations"]
    loose = results["loose"].metrics["joint_violations"]
    assert tight >= default >= loose


def test_safety_filter_study_results_to_markdown_contains_all_rows():
    results = safety_filter_study(seed=0).run(safety_filter_evaluator)
    md = results.to_markdown()
    for r in results:
        assert f"| {r.name}" in md


def test_safety_filter_study_seed_propagates():
    results = safety_filter_study(seed=42).run(safety_filter_evaluator)
    assert results.seed == 42
    # rows seeded with seed + index
    assert results.rows[0].meta["seed"] == 42
    assert results.rows[1].meta["seed"] == 43


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_main_writes_to_stdout(capsys):
    from net.ablation.run import main
    rc = main(["--study", "robot_workspace", "--format", "csv", "--seed", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    # CSV header must start with ablation column and include mean_correction
    assert out.startswith("ablation,")
    assert "mean_correction" in out
    assert "tight" in out and "default" in out and "loose" in out


def test_cli_main_writes_markdown_to_file(tmp_path):
    from net.ablation.run import main
    out = tmp_path / "ablation.md"
    rc = main([
        "--study", "robot_workspace", "--format", "md",
        "--seed", "0", "--out", str(out),
    ])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "## Ablation: robot_workspace" in text


def test_cli_unknown_study_rejected():
    from net.ablation.run import main
    with pytest.raises(SystemExit):
        main(["--study", "frobnicator"])


# ---------------------------------------------------------------------------
# Model-level ablation: scaffold + runner registration
# ---------------------------------------------------------------------------


from net.ablation import (  # noqa: E402
    MODEL_METRIC_NAMES,
    get_model_runner,
    model_ablation_evaluator,
    model_ablation_study,
    set_model_runner,
)


@pytest.fixture(autouse=True)
def _reset_model_runner():
    """Always start tests with the runner reset to the NaN stub."""
    set_model_runner(None)
    yield
    set_model_runner(None)


def test_model_ablation_metric_names_are_documented():
    expected = (
        "chamfer_l1", "iou", "topology_accuracy",
        "safety_violation_rate", "mean_correction",
    )
    assert MODEL_METRIC_NAMES == expected


def test_model_ablation_study_has_four_rows():
    s = model_ablation_study(seed=0)
    names = [a.name for a in s.ablations]
    assert names == [
        "full", "no_3d_input", "no_mobility_conditioning", "no_safety_filter",
    ]


def test_model_ablation_study_full_row_is_baseline():
    s = model_ablation_study(seed=0)
    full = s.ablations[0]
    assert full.overrides == {}


def test_model_ablation_no_3d_input_drops_to_label_only():
    s = model_ablation_study()
    row = next(a for a in s.ablations if a.name == "no_3d_input")
    assert row.overrides == {"input_dim": 1}


def test_model_ablation_no_mobility_conditioning_uses_observed_only():
    s = model_ablation_study()
    row = next(a for a in s.ablations if a.name == "no_mobility_conditioning")
    assert row.overrides == {"ee_conditioning": "observed"}


def test_model_ablation_no_safety_filter_disables_safety_flag():
    s = model_ablation_study()
    row = next(a for a in s.ablations if a.name == "no_safety_filter")
    assert row.overrides == {"safety_enabled": False}


def test_model_evaluator_returns_nan_when_no_runner_registered():
    metrics = model_ablation_evaluator({"input_dim": 4}, seed=0)
    assert set(metrics.keys()) == set(MODEL_METRIC_NAMES)
    for k, v in metrics.items():
        assert v != v, f"{k} expected NaN, got {v}"


def test_model_evaluator_raises_for_non_callable_runner():
    with pytest.raises(TypeError, match="callable or None"):
        set_model_runner(42)  # type: ignore[arg-type]


def test_model_evaluator_uses_registered_runner():
    captured = {}
    def runner(cfg, seed):
        captured["cfg"] = dict(cfg)
        captured["seed"] = seed
        return {k: 1.0 for k in MODEL_METRIC_NAMES}
    set_model_runner(runner)
    metrics = model_ablation_evaluator({"input_dim": 1}, seed=5)
    assert captured == {"cfg": {"input_dim": 1}, "seed": 5}
    assert all(metrics[k] == 1.0 for k in MODEL_METRIC_NAMES)


def test_model_evaluator_rejects_runner_with_missing_metrics():
    set_model_runner(lambda c, s: {"chamfer_l1": 0.1})
    with pytest.raises(ValueError, match="missing required metrics"):
        model_ablation_evaluator({}, seed=0)


def test_model_evaluator_rejects_runner_with_extra_metrics():
    set_model_runner(lambda c, s: {**{k: 0.0 for k in MODEL_METRIC_NAMES}, "bonus": 1.0})
    with pytest.raises(ValueError, match="unexpected metrics"):
        model_ablation_evaluator({}, seed=0)


def test_model_evaluator_rejects_non_dict_runner_output():
    set_model_runner(lambda c, s: [1.0, 2.0])  # type: ignore[return-value]
    with pytest.raises(TypeError, match="must return a dict"):
        model_ablation_evaluator({}, seed=0)


def test_model_evaluator_coerces_metric_values_to_float():
    set_model_runner(lambda c, s: {k: 1 for k in MODEL_METRIC_NAMES})
    metrics = model_ablation_evaluator({}, seed=0)
    assert all(isinstance(v, float) for v in metrics.values())


def test_get_model_runner_round_trips():
    assert get_model_runner() is None
    fn = lambda c, s: {k: 0.0 for k in MODEL_METRIC_NAMES}  # noqa: E731
    set_model_runner(fn)
    assert get_model_runner() is fn
    set_model_runner(None)
    assert get_model_runner() is None


def test_model_ablation_study_runs_with_stub_evaluator():
    """End-to-end: study.run produces a NaN-only table without crashing."""
    results = model_ablation_study(seed=0).run(model_ablation_evaluator)
    assert len(results) == 4
    for r in results:
        for v in r.metrics.values():
            assert v != v  # NaN


def test_model_ablation_study_runs_with_registered_runner():
    """End-to-end: registered runner sees the merged effective config."""
    seen = []
    def runner(cfg, seed):
        seen.append((dict(cfg), seed))
        return {k: float(seed) for k in MODEL_METRIC_NAMES}
    set_model_runner(runner)
    results = model_ablation_study(seed=10).run(model_ablation_evaluator)
    # Each row sees a merged config: full base for "full", overrides applied
    # for the others.
    cfgs = [c for c, _ in seen]
    assert cfgs[0]["input_dim"] == 4 and cfgs[0]["ee_conditioning"] == "both"
    assert cfgs[1]["input_dim"] == 1                    # no_3d_input
    assert cfgs[2]["ee_conditioning"] == "observed"     # no_mobility_conditioning
    assert cfgs[3]["safety_enabled"] is False           # no_safety_filter
    # per-row seeds are 10, 11, 12, 13
    seeds = [s for _, s in seen]
    assert seeds == [10, 11, 12, 13]
    # metrics propagate
    assert results["full"].metrics["chamfer_l1"] == 10.0
    assert results["no_safety_filter"].metrics["mean_correction"] == 13.0


def test_model_ablation_markdown_renders_nan_cells():
    results = model_ablation_study(seed=0).run(model_ablation_evaluator)
    md = results.to_markdown()
    # one NaN per metric cell per row
    assert md.count("NaN") >= len(MODEL_METRIC_NAMES) * len(results)
    # row labels still show up
    for r in results:
        assert f"| {r.name}" in md


def test_cli_model_ablation_writes_nan_table_to_stdout(capsys):
    from net.ablation.run import main
    rc = main(["--study", "model_ablation", "--format", "md", "--seed", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Ablation: model_ablation" in out
    assert "no_3d_input" in out
    assert "no_mobility_conditioning" in out
    assert "no_safety_filter" in out
    assert "NaN" in out  # default stub renders NaN


def test_cli_model_ablation_csv_includes_model_metric_columns(capsys):
    from net.ablation.run import main
    rc = main(["--study", "model_ablation", "--format", "csv", "--seed", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("ablation,")
    for m in MODEL_METRIC_NAMES:
        assert m in out

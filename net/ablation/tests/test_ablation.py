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

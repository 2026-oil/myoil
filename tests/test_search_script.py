from __future__ import annotations

from pathlib import Path

import yaml

from scripts import search as mod


def _case_eval(
    *,
    case_key: str,
    h1_growth: float,
    h2_growth: float,
    monotonic_up: bool,
    spike_pass: bool,
    mean_abs_pct_error: float = 0.05,
) -> mod.CaseEvaluation:
    return mod.CaseEvaluation(
        case_key=case_key,
        run_root=f"/tmp/{case_key}",
        log_path=f"/tmp/{case_key}.log",
        cutoff="2026-02-23 00:00:00",
        last_observed=100.0,
        h1_prediction=100.0 * (1.0 + h1_growth),
        h2_prediction=100.0 * (1.0 + h2_growth),
        h1_growth=h1_growth,
        h2_growth=h2_growth,
        monotonic_up=monotonic_up,
        spike_pass=spike_pass,
        mean_abs_pct_error=mean_abs_pct_error,
        retrieval_artifact=None,
        retrieval_applied=True,
    )


def test_build_retrieval_doc_updates_only_retrieval_fields() -> None:
    base_doc = {
        "retrieval": {
            "enabled": True,
            "top_k": 2,
            "trigger_quantile": 0.01,
            "min_similarity": 0.0,
            "temperature": 0.01,
            "blend_floor": 0.0,
            "blend_max": 1.0,
            "use_uncertainty_gate": True,
            "event_score_log_bonus_alpha": 0.0,
            "event_score_log_bonus_cap": 0.0,
            "star": {"thresh": 3.5},
        }
    }
    params = {
        "top_k": 5,
        "recency_gap_steps": 11,
        "trigger_quantile": 0.09,
        "min_similarity": 0.42,
        "temperature": 0.15,
        "blend_floor": 0.1,
        "blend_max": 0.7,
        "use_uncertainty_gate": False,
        "event_score_log_bonus_alpha": 0.2,
        "event_score_log_bonus_cap": 1.2,
    }

    updated = mod._build_retrieval_doc(base_doc, params)

    assert updated["retrieval"]["top_k"] == 5
    assert updated["retrieval"]["recency_gap_steps"] == 11
    assert updated["retrieval"]["trigger_quantile"] == 0.09
    assert updated["retrieval"]["min_similarity"] == 0.42
    assert updated["retrieval"]["temperature"] == 0.15
    assert updated["retrieval"]["blend_floor"] == 0.1
    assert updated["retrieval"]["blend_max"] == 0.7
    assert updated["retrieval"]["use_uncertainty_gate"] is False
    assert updated["retrieval"]["event_score_log_bonus_alpha"] == 0.2
    assert updated["retrieval"]["event_score_log_bonus_cap"] == 1.2
    assert updated["retrieval"]["star"] == {"thresh": 3.5}
    assert base_doc["retrieval"]["top_k"] == 2



def test_build_aa_forecast_doc_updates_star_fields_and_relative_retrieval_path(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "configs"
    retrieval_path = source_dir / "baseline_retrieval.yaml"
    base_doc = {
        "aa_forecast": {
            "lowess_frac": 0.35,
            "lowess_delta": 0.01,
            "thresh": 3.5,
            "star_anomaly_tails": {"upward": ["GPRD_THREAT"], "two_sided": []},
            "retrieval": {"config_path": "../retrieval/baseline_retrieval.yaml"},
        }
    }
    params = {
        "star_lowess_frac": 0.22,
        "star_lowess_delta": 0.005,
        "star_thresh": 4.2,
        "star_anomaly_tails_upward": ["GPRD", "Idx_OVX"],
    }

    updated = mod._build_aa_forecast_doc(
        base_doc,
        retrieval_config_path=retrieval_path,
        source_dir=source_dir,
        params=params,
    )

    assert updated["aa_forecast"]["lowess_frac"] == 0.22
    assert updated["aa_forecast"]["lowess_delta"] == 0.005
    assert updated["aa_forecast"]["thresh"] == 4.2
    assert updated["aa_forecast"]["star_anomaly_tails"]["upward"] == [
        "GPRD",
        "Idx_OVX",
    ]
    assert updated["aa_forecast"]["retrieval"]["config_path"] == "baseline_retrieval.yaml"



def test_score_trial_result_requires_both_cases_to_pass() -> None:
    strong = _case_eval(
        case_key="wti",
        h1_growth=0.05,
        h2_growth=0.12,
        monotonic_up=True,
        spike_pass=True,
    )
    weak = _case_eval(
        case_key="brent",
        h1_growth=0.02,
        h2_growth=0.06,
        monotonic_up=True,
        spike_pass=False,
    )

    not_yet = mod._score_trial_result(
        {"wti": strong, "brent": weak},
        spike_threshold=0.07,
    )
    passed = mod._score_trial_result(
        {
            "wti": strong,
            "brent": _case_eval(
                case_key="brent",
                h1_growth=0.03,
                h2_growth=0.09,
                monotonic_up=True,
                spike_pass=True,
            ),
        },
        spike_threshold=0.07,
    )

    assert not not_yet.pass_both
    assert passed.pass_both
    assert passed.objective > not_yet.objective



def test_build_recommendations_writes_two_yaml_outputs(tmp_path: Path) -> None:
    base_retrieval = {
        "retrieval": {
            "enabled": True,
            "top_k": 2,
            "recency_gap_steps": 8,
            "trigger_quantile": 0.01,
            "min_similarity": 0.0,
            "temperature": 0.01,
            "blend_floor": 0.0,
            "blend_max": 1.0,
            "use_uncertainty_gate": True,
            "event_score_log_bonus_alpha": 0.0,
            "event_score_log_bonus_cap": 0.0,
        }
    }
    base_aa = {
        "aa_forecast": {
            "lowess_frac": 0.35,
            "lowess_delta": 0.01,
            "thresh": 3.5,
            "star_anomaly_tails": {"upward": ["GPRD_THREAT"], "two_sided": []},
            "retrieval": {"config_path": "../retrieval/baseline_retrieval.yaml"},
        }
    }
    params = {
        "top_k": 3,
        "recency_gap_steps": 6,
        "trigger_quantile": 0.08,
        "min_similarity": 0.4,
        "temperature": 0.1,
        "blend_floor": 0.05,
        "blend_max": 0.9,
        "use_uncertainty_gate": False,
        "event_score_log_bonus_alpha": 0.3,
        "event_score_log_bonus_cap": 1.5,
        "star_lowess_frac": 0.25,
        "star_lowess_delta": 0.002,
        "star_thresh": 4.1,
        "star_anomaly_tails_upward": ["GPRD", "GPRD_ACT"],
    }

    outputs = mod._build_recommendations(
        bundle_root=tmp_path,
        base_retrieval_doc=base_retrieval,
        base_aa_doc=base_aa,
        best_params=params,
    )

    retrieval_doc = yaml.safe_load(Path(outputs["recommended_baseline_retrieval"]).read_text())
    aa_doc = yaml.safe_load(Path(outputs["recommended_aa_forecast"]).read_text())

    assert retrieval_doc["retrieval"]["top_k"] == 3
    assert aa_doc["aa_forecast"]["star_anomaly_tails"]["upward"] == [
        "GPRD",
        "GPRD_ACT",
    ]
    assert aa_doc["aa_forecast"]["retrieval"]["config_path"] == "recommended_baseline_retrieval.yaml"


def test_tail_choice_map_covers_all_non_empty_star_subsets() -> None:
    assert len(mod.TAIL_CHOICE_MAP) == 15
    assert mod.TAIL_CHOICE_MAP["gprd"] == ("GPRD",)
    assert mod.TAIL_CHOICE_MAP["gprd__idx_ovx"] == ("GPRD", "Idx_OVX")
    assert mod.TAIL_CHOICE_MAP["gprd_threat__gprd__gprd_act__idx_ovx"] == (
        "GPRD_THREAT",
        "GPRD",
        "GPRD_ACT",
        "Idx_OVX",
    )


def test_resolve_feature_selection_with_candidate_flags_promotes_star_columns() -> None:
    params = {
        "use_hist__gprd_threat": False,
        "use_hist__bs_core_index_a": True,
        "use_hist__bs_core_index_c": False,
        "use_hist__gprd": False,
        "use_star__gprd": True,
        "use_star__bs_core_index_c": True,
    }

    selected_hist, selected_star = mod._resolve_feature_selection(
        params=params,
        default_hist_exog_cols=("GPRD_THREAT", "GPRD", "GPRD_ACT"),
        hist_exog_candidates=(
            "GPRD_THREAT",
            "BS_Core_Index_A",
            "BS_Core_Index_C",
            "GPRD",
        ),
        star_candidates=("BS_Core_Index_C", "GPRD"),
    )

    assert selected_hist == ["BS_Core_Index_A", "BS_Core_Index_C", "GPRD"]
    assert selected_star == ["BS_Core_Index_C", "GPRD"]


def test_build_experiment_doc_can_override_dataset_path_and_hist_exog_cols() -> None:
    base_doc = {
        "task": {"name": "old"},
        "dataset": {
            "path": "data/df.csv",
            "target_col": "Com_CrudeOil",
            "dt_col": "dt",
            "hist_exog_cols": ["GPRD_THREAT"],
        },
        "aa_forecast": {"config_path": "yaml/plugins/aa_forecast/aa_forecast_gru-ret.yaml"},
    }

    updated = mod._build_experiment_doc(
        base_doc,
        Path("tmp/aa_forecast.yaml"),
        task_name="new-task",
        dataset_path=Path("data/test.csv"),
        hist_exog_cols=["BS_Core_Index_A", "GPRD"],
    )

    assert updated["task"]["name"] == "new-task"
    assert updated["dataset"]["path"] == "data/test.csv"
    assert updated["dataset"]["hist_exog_cols"] == ["BS_Core_Index_A", "GPRD"]
    assert updated["aa_forecast"]["config_path"] == "tmp/aa_forecast.yaml"

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests

TARGETS = ("Com_CrudeOil", "Com_BrentCrudeOil")
EXCLUDED_PREDICTORS = frozenset({"Com_CrudeOil", "Com_BrentCrudeOil"})
PRACTICAL_METHODS = ("lagged_correlation", "granger")
IMPLEMENTED_OPTIONAL_METHODS = (
    "tigramite_pcmci",
    "nonlincausality",
    "neural_gc",
    "tcdf",
)
OPTIONAL_METHODS = (
    "tigramite_pcmci",
    "nonlincausality",
    "neural_gc",
    "gvar",
    "jrngc",
    "tcdf",
    "causalformer",
    "cuts_plus",
    "sru_gci",
    "gc_xlstm",
)
SUPPORTED_METHODS = PRACTICAL_METHODS + OPTIONAL_METHODS

OPTIONAL_METHOD_METADATA: dict[str, dict[str, Any]] = {
    "tigramite_pcmci": {
        "label": "Tigramite / PCMCI",
        "import_name": "tigramite",
        "reason": "package not installed in the current environment",
    },
    "nonlincausality": {
        "label": "Nonlincausality",
        "import_name": "nonlincausality",
        "reason": "package not installed in the current environment or tensorflow backend missing",
    },
    "neural_gc": {
        "label": "Neural Granger Causality",
        "local_paths": ("casual/vendor/neural_gc",),
        "reason": "no vendored Neural-GC implementation was found",
    },
    "gvar": {
        "label": "GVAR self-explaining GC",
        "local_paths": ("casual/vendor/GVAR", "vendor/GVAR"),
        "reason": "no vendored GVAR implementation was found",
    },
    "jrngc": {
        "label": "JRNGC",
        "local_paths": ("casual/vendor/JRNGC", "vendor/JRNGC"),
        "reason": "no vendored JRNGC implementation was found",
    },
    "tcdf": {
        "label": "TCDF",
        "local_paths": ("casual/vendor/tcdf",),
        "reason": "no vendored TCDF implementation was found",
    },
    "causalformer": {
        "label": "CausalFormer",
        "local_paths": ("casual/vendor/CausalFormer", "vendor/CausalFormer"),
        "reason": "no vendored CausalFormer implementation was found",
    },
    "cuts_plus": {
        "label": "CUTS / CUTS+",
        "local_paths": ("casual/vendor/UNN", "vendor/UNN"),
        "reason": "no vendored CUTS/CUTS+ implementation was found",
    },
    "sru_gci": {
        "label": "SRU for GCI",
        "local_paths": ("casual/vendor/SRU_for_GCI", "vendor/SRU_for_GCI"),
        "reason": "no vendored SRU_for_GCI implementation was found",
    },
    "gc_xlstm": {
        "label": "GC-xLSTM",
        "local_paths": ("casual/vendor/GC-xLSTM", "vendor/GC-xLSTM"),
        "reason": "no vendored GC-xLSTM implementation was found",
    },
}


@dataclass(slots=True)
class PipelineConfig:
    csv_path: Path
    output_root: Path | None = None
    max_lag: int = 8
    top_k: int = 20
    methods: tuple[str, ...] = SUPPORTED_METHODS
    targets: tuple[str, ...] = TARGETS
    heavy_predictor_limit: int = 20
    neural_gc_max_iter: int = 200
    tcdf_epochs: int = 200


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_methods(methods_arg: str | None) -> tuple[str, ...]:
    if not methods_arg or methods_arg.strip().lower() == "all":
        return SUPPORTED_METHODS
    methods = tuple(part.strip() for part in methods_arg.split(",") if part.strip())
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def parse_targets(targets_arg: str | None) -> tuple[str, ...]:
    if not targets_arg:
        return TARGETS
    targets = tuple(part.strip() for part in targets_arg.split(",") if part.strip())
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {', '.join(unknown)}")
    return targets


def default_output_root() -> Path:
    return Path("runs") / f"casual-leading-indicators-{utc_timestamp()}"


def _predictor_family(name: str) -> str:
    return name.split("_", 1)[0]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "dt" in frame.columns:
        try:
            frame["dt"] = pd.to_datetime(frame["dt"])
        except Exception:
            pass
    return frame


def dataset_audit(frame: pd.DataFrame, csv_path: Path) -> dict[str, Any]:
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    return {
        "csv_path": str(csv_path),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "numeric_columns": len(numeric_columns),
        "targets_present": {target: target in frame.columns for target in TARGETS},
        "numeric_column_names": numeric_columns,
        "date_column_present": "dt" in frame.columns,
        "missing_values_total": int(frame.isna().sum().sum()),
    }


def candidate_predictors(frame: pd.DataFrame, target: str) -> list[str]:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    predictors: list[str] = []
    for column in numeric.columns:
        if column == target or column in EXCLUDED_PREDICTORS:
            continue
        series = numeric[column].dropna()
        if series.empty or series.nunique() <= 1:
            continue
        predictors.append(column)
    return predictors


def _prepare_numeric_variant(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if variant == "raw":
        return numeric
    if variant == "diff1":
        return numeric.diff().dropna()
    raise ValueError(f"Unsupported variant: {variant}")


def _standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    values = scaler.fit_transform(frame.values)
    return pd.DataFrame(values, columns=frame.columns, index=frame.index)


def compute_lagged_correlations(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    max_lag: int,
    variant: str,
) -> pd.DataFrame:
    numeric = _prepare_numeric_variant(frame, variant)
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        best_row: dict[str, Any] | None = None
        for lag in range(1, max_lag + 1):
            aligned = pd.DataFrame(
                {
                    "target": numeric[target],
                    "predictor": numeric[predictor].shift(lag),
                }
            ).dropna()
            if len(aligned) < max(25, max_lag * 3):
                continue
            corr = aligned["predictor"].corr(aligned["target"])
            if pd.isna(corr):
                continue
            candidate = {
                "predictor": predictor,
                "predictor_family": _predictor_family(predictor),
                "variant": variant,
                "best_lag": lag,
                "correlation": float(corr),
                "abs_correlation": float(abs(corr)),
                "sample_size": int(len(aligned)),
            }
            if best_row is None or candidate["abs_correlation"] > best_row["abs_correlation"]:
                best_row = candidate
        if best_row is not None:
            rows.append(best_row)

    columns = [
        "rank",
        "predictor",
        "predictor_family",
        "variant",
        "best_lag",
        "correlation",
        "abs_correlation",
        "sample_size",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows).sort_values(
        ["abs_correlation", "correlation", "predictor"],
        ascending=[False, False, True],
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)


def compute_pairwise_granger(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    max_lag: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    numeric = _prepare_numeric_variant(frame, "diff1")
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    min_obs = max(40, max_lag * 5)
    for predictor in predictors:
        aligned = numeric[[target, predictor]].dropna()
        if len(aligned) < min_obs:
            failures[predictor] = f"insufficient observations after differencing ({len(aligned)})"
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                tests = grangercausalitytests(aligned[[target, predictor]], maxlag=max_lag, verbose=False)
        except Exception as exc:
            failures[predictor] = str(exc)
            continue
        best_lag = None
        best_pvalue = None
        best_fstat = None
        for lag, outputs in tests.items():
            statistic, pvalue, *_ = outputs[0]["ssr_ftest"]
            if best_pvalue is None or pvalue < best_pvalue:
                best_lag = lag
                best_pvalue = float(pvalue)
                best_fstat = float(statistic)
        if best_lag is None or best_pvalue is None or best_fstat is None:
            failures[predictor] = "no usable lag statistics returned"
            continue
        rows.append(
            {
                "predictor": predictor,
                "predictor_family": _predictor_family(predictor),
                "variant": "diff1",
                "best_lag": int(best_lag),
                "best_pvalue": float(best_pvalue),
                "best_fstat": float(best_fstat),
                "score": float(-np.log10(max(best_pvalue, 1e-16))),
                "significant_0_05": bool(best_pvalue < 0.05),
                "sample_size": int(len(aligned)),
            }
        )

    columns = [
        "rank",
        "predictor",
        "predictor_family",
        "variant",
        "best_lag",
        "best_pvalue",
        "best_fstat",
        "score",
        "significant_0_05",
        "sample_size",
    ]
    if not rows:
        return pd.DataFrame(columns=columns), {"failures": failures}
    result = pd.DataFrame(rows).sort_values(
        ["best_pvalue", "score", "predictor"],
        ascending=[True, False, True],
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True), {"failures": failures}


def _normalize_ranking_table(table: pd.DataFrame, method_name: str, target: str) -> pd.DataFrame:
    ranked = table.copy()
    if ranked.empty:
        return ranked
    denominator = max(len(ranked) - 1, 1)
    ranked["method"] = method_name
    ranked["target"] = target
    ranked["normalized_rank_score"] = 1.0 - ((ranked["rank"] - 1) / denominator)
    return ranked


def synthesize_rankings(tables: list[pd.DataFrame], target: str, top_k: int) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame(
            columns=[
                "rank",
                "target",
                "predictor",
                "predictor_family",
                "support_count",
                "top_k_support_count",
                "mean_normalized_rank_score",
                "median_best_lag",
                "methods",
            ]
        )
    combined = pd.concat(tables, ignore_index=True)
    grouped = (
        combined.groupby(["predictor", "predictor_family"], dropna=False)
        .agg(
            support_count=("method", "nunique"),
            top_k_support_count=("rank", lambda s: int((s <= top_k).sum())),
            mean_normalized_rank_score=("normalized_rank_score", "mean"),
            median_best_lag=("best_lag", "median"),
            methods=("method", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["top_k_support_count", "support_count", "mean_normalized_rank_score", "predictor"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    grouped.insert(1, "target", target)
    return grouped


def _priority_predictors(
    normalized_tables: list[pd.DataFrame],
    fallback_table: pd.DataFrame,
    predictors: list[str],
    limit: int,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for table in normalized_tables:
        for predictor in table.sort_values("rank")["predictor"].tolist():
            if predictor not in seen:
                ordered.append(predictor)
                seen.add(predictor)
            if len(ordered) >= limit:
                return ordered
    if not fallback_table.empty:
        for predictor in fallback_table.sort_values("rank")["predictor"].tolist():
            if predictor not in seen:
                ordered.append(predictor)
                seen.add(predictor)
            if len(ordered) >= limit:
                return ordered
    for predictor in predictors:
        if predictor not in seen:
            ordered.append(predictor)
            seen.add(predictor)
        if len(ordered) >= limit:
            break
    return ordered


def _status_row(
    *,
    method: str,
    target: str,
    status: str,
    reason: str,
    table_path: str | None = None,
    metadata_path: str | None = None,
    row_count: int | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "target": target,
        "status": status,
        "reason": reason,
        "table_path": table_path or "",
        "metadata_path": metadata_path or "",
        "row_count": int(row_count or 0),
    }


def _write_method_metadata(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def probe_optional_method(method: str, repo_root: Path) -> tuple[str, str]:
    metadata = OPTIONAL_METHOD_METADATA[method]
    import_name = metadata.get("import_name")
    if import_name:
        try:
            importlib.import_module(str(import_name))
            if method in IMPLEMENTED_OPTIONAL_METHODS:
                return "blocked", "adapter available but method was not requested in this run"
            return "blocked", "package is installed but no local adapter has been wired yet"
        except Exception:
            return "skipped-not-feasible", str(metadata["reason"])
    for rel_path in metadata.get("local_paths", ()):
        if (repo_root / rel_path).exists():
            if method in IMPLEMENTED_OPTIONAL_METHODS:
                return "blocked", "vendored source exists and adapter is available"
            return "blocked", "vendored source exists but no local adapter has been wired yet"
    return "skipped-not-feasible", str(metadata["reason"])


def _dynamic_import_from_path(module_name: str, root: Path):
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(module_name)


def compute_tigramite_pcmci(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    max_lag: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    cols = [target, *predictors]
    numeric = _standardize_frame(_prepare_numeric_variant(frame[cols], "diff1").dropna())
    df = pp.DataFrame(data=numeric.to_numpy(), var_names=cols)
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=None, alpha_level=0.05)
    parents = pcmci.return_parents_dict(graph=results["graph"], val_matrix=results["val_matrix"])
    target_links = parents.get(0, [])
    rows: list[dict[str, Any]] = []
    for source_idx, lag in target_links:
        source_idx = int(source_idx)
        lag = int(lag)
        if source_idx == 0 or lag >= 0:
            continue
        best_lag = abs(lag)
        predictor = cols[source_idx]
        score = float(abs(results["val_matrix"][0, source_idx, best_lag]))
        pvalue = float(results["p_matrix"][0, source_idx, best_lag])
        rows.append(
            {
                "predictor": predictor,
                "predictor_family": _predictor_family(predictor),
                "variant": "diff1_standardized",
                "best_lag": best_lag,
                "score": score,
                "best_pvalue": pvalue,
                "significant_0_05": bool(pvalue < 0.05),
                "sample_size": int(len(numeric)),
            }
        )
    columns = [
        "rank",
        "predictor",
        "predictor_family",
        "variant",
        "best_lag",
        "score",
        "best_pvalue",
        "significant_0_05",
        "sample_size",
    ]
    if not rows:
        return pd.DataFrame(columns=columns), {"selected_predictors": predictors, "parents": []}
    table = pd.DataFrame(rows).sort_values(["best_pvalue", "score", "predictor"], ascending=[True, False, True])
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True), {"selected_predictors": predictors, "parents": [tuple(map(int, p)) for p in target_links]}


def compute_nonlincausality_rf(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    max_lag: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import nonlincausality as nlc

    numeric = _standardize_frame(_prepare_numeric_variant(frame[[target, *predictors]], "diff1").dropna())
    if len(numeric) < 120:
        raise ValueError(f"insufficient observations for nonlincausality ({len(numeric)})")
    train_end = int(len(numeric) * 0.6)
    val_end = int(len(numeric) * 0.8)
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for predictor in predictors:
        pair = numeric[[target, predictor]].to_numpy(dtype=float)
        train = pair[:train_end]
        val = pair[train_end:val_end]
        test = pair[val_end:]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = nlc.nonlincausality_sklearn(
                    x=train,
                    sklearn_model=RandomForestRegressor,
                    maxlag=max_lag,
                    params={
                        "n_estimators": [80],
                        "max_depth": [4],
                        "random_state": [0],
                        "n_jobs": [1],
                    },
                    x_test=test,
                    x_val=val,
                    plot=False,
                )
        except Exception as exc:
            failures[predictor] = str(exc)
            continue
        best_row: dict[str, Any] | None = None
        for lag, result in results.items():
            pvalue = float(result.p_value)
            if np.isnan(pvalue):
                continue
            rss_gain = float(result.best_RSS_X - result.best_RSS_XY)
            candidate = {
                "predictor": predictor,
                "predictor_family": _predictor_family(predictor),
                "variant": "diff1_standardized_rf",
                "best_lag": int(lag),
                "best_pvalue": pvalue,
                "test_statistic": float(result.test_statistic),
                "score": float(-np.log10(max(pvalue, 1e-16))),
                "rss_gain": rss_gain,
                "significant_0_05": bool(pvalue < 0.05),
                "sample_size": int(len(test)),
            }
            if best_row is None or candidate["best_pvalue"] < best_row["best_pvalue"]:
                best_row = candidate
        if best_row is not None:
            rows.append(best_row)
        else:
            failures[predictor] = "all lag p-values were NaN"
    columns = [
        "rank",
        "predictor",
        "predictor_family",
        "variant",
        "best_lag",
        "best_pvalue",
        "test_statistic",
        "score",
        "rss_gain",
        "significant_0_05",
        "sample_size",
    ]
    if not rows:
        return pd.DataFrame(columns=columns), {"selected_predictors": predictors, "failures": failures}
    table = pd.DataFrame(rows).sort_values(["best_pvalue", "score", "predictor"], ascending=[True, False, True])
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True), {"selected_predictors": predictors, "failures": failures}


def compute_neural_gc(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    max_lag: int,
    max_iter: int,
    repo_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import torch

    vendor_root = repo_root / "casual/vendor/neural_gc"
    if not vendor_root.exists():
        raise FileNotFoundError("casual/vendor/neural_gc is missing")
    module = _dynamic_import_from_path("models.cmlp", vendor_root)
    cMLP = getattr(module, "cMLP")
    train_model_ista = getattr(module, "train_model_ista")

    cols = [target, *predictors]
    numeric = _standardize_frame(_prepare_numeric_variant(frame[cols], "diff1").dropna())
    torch.set_num_threads(1)
    torch.manual_seed(0)
    X = torch.tensor(numeric.to_numpy(dtype=np.float32)).unsqueeze(0)
    model = cMLP(num_series=len(cols), lag=max_lag, hidden=[8])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_model_ista(model, X, lr=0.01, max_iter=max_iter, lam=0.002, lam_ridge=0.0, penalty="H", check_every=50, verbose=0)
    gc_matrix = model.GC(threshold=False, ignore_lag=False).detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for predictor_idx, predictor in enumerate(predictors, start=1):
        lag_scores = gc_matrix[0, predictor_idx]
        best_lag_idx = int(np.argmax(lag_scores))
        rows.append(
            {
                "predictor": predictor,
                "predictor_family": _predictor_family(predictor),
                "variant": "diff1_standardized",
                "best_lag": best_lag_idx + 1,
                "score": float(lag_scores[best_lag_idx]),
                "mean_score": float(np.mean(lag_scores)),
                "sample_size": int(len(numeric)),
            }
        )
    columns = ["rank", "predictor", "predictor_family", "variant", "best_lag", "score", "mean_score", "sample_size"]
    if not rows:
        return pd.DataFrame(columns=columns), {"selected_predictors": predictors}
    table = pd.DataFrame(rows).sort_values(["score", "mean_score", "predictor"], ascending=[False, False, True])
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True), {
        "selected_predictors": predictors,
        "preprocessing": "diff1_standardized",
        "hyperparameters": {"hidden": [8], "lam": 0.002, "lr": 0.01, "max_iter": max_iter},
    }


def compute_tcdf(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    max_lag: int,
    epochs: int,
    repo_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    vendor_root = repo_root / "casual/vendor/tcdf"
    if not vendor_root.exists():
        raise FileNotFoundError("casual/vendor/tcdf is missing")
    module = _dynamic_import_from_path("TCDF", vendor_root)

    cols = [target, *predictors]
    numeric = _standardize_frame(_prepare_numeric_variant(frame[cols], "diff1").dropna())
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        numeric.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
    try:
        causes, causes_with_delay, loss, scores = module.findcauses(
            target,
            cuda=False,
            epochs=epochs,
            kernel_size=max(2, max_lag),
            layers=1,
            log_interval=max(epochs, 1),
            lr=0.01,
            optimizername="Adam",
            seed=0,
            dilation_c=1,
            significance=0.8,
            file=str(tmp_path),
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    target_idx = cols.index(target)
    for idx in causes:
        idx = int(idx)
        if idx == target_idx:
            continue
        predictor = cols[idx]
        rows.append(
            {
                "predictor": predictor,
                "predictor_family": _predictor_family(predictor),
                "variant": "diff1_standardized",
                "best_lag": int(causes_with_delay.get((target_idx, idx), 1)),
                "score": float(scores[idx]),
                "sample_size": int(len(numeric)),
            }
        )
    columns = ["rank", "predictor", "predictor_family", "variant", "best_lag", "score", "sample_size"]
    if not rows:
        return pd.DataFrame(columns=columns), {
            "selected_predictors": predictors,
            "validated_causes": [],
            "loss": float(loss),
        }
    table = pd.DataFrame(rows).sort_values(["score", "predictor"], ascending=[False, True])
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True), {
        "selected_predictors": predictors,
        "validated_causes": [int(idx) for idx in causes],
        "loss": float(loss),
        "epochs": epochs,
    }


def render_report(*, config: PipelineConfig, audit: dict[str, Any], status_table: pd.DataFrame, synthesis_paths: dict[str, str]) -> str:
    lines = [
        "# casual df.csv causal leading-indicator report",
        "",
        "## Run configuration",
        f"- csv: `{config.csv_path}`",
        f"- max_lag: `{config.max_lag}`",
        f"- top_k synthesis support: `{config.top_k}`",
        f"- heavy_predictor_limit: `{config.heavy_predictor_limit}`",
        f"- methods requested: `{', '.join(config.methods)}`",
        f"- targets: `{', '.join(config.targets)}`",
        "",
        "## Dataset audit",
        f"- rows: `{audit['rows']}`",
        f"- columns: `{audit['columns']}`",
        f"- numeric columns: `{audit['numeric_columns']}`",
        f"- missing values total: `{audit['missing_values_total']}`",
        "",
        "## Method status summary",
    ]
    status_counts = status_table.groupby(["target", "status"]).size().reset_index(name="count")
    for _, row in status_counts.iterrows():
        lines.append(f"- {row['target']} / {row['status']}: {row['count']}")

    for target in config.targets:
        lines.extend(["", f"## {target}", f"- synthesis table: `{synthesis_paths.get(target, '')}`"])
        target_status = status_table[status_table["target"] == target]
        blocked = target_status[target_status["status"] != "success"][["method", "status", "reason"]]
        if blocked.empty:
            lines.append("- blocked/skipped methods: none")
        else:
            lines.append("- blocked/skipped methods:")
            for _, row in blocked.iterrows():
                lines.append(f"  - `{row['method']}` -> `{row['status']}` ({row['reason']})")
    lines.append("")
    return "\n".join(lines)


def _run_method(
    *,
    method: str,
    target: str,
    frame: pd.DataFrame,
    predictors: list[str],
    config: PipelineConfig,
    repo_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if method == "tigramite_pcmci":
        return compute_tigramite_pcmci(frame, target, predictors, config.max_lag)
    if method == "nonlincausality":
        return compute_nonlincausality_rf(frame, target, predictors, config.max_lag)
    if method == "neural_gc":
        return compute_neural_gc(frame, target, predictors, config.max_lag, config.neural_gc_max_iter, repo_root)
    if method == "tcdf":
        return compute_tcdf(frame, target, predictors, config.max_lag, config.tcdf_epochs, repo_root)
    raise ValueError(f"No adapter wired for method: {method}")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    repo_root = Path.cwd()
    output_root = config.output_root or default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    metadata_dir = output_root / "metadata"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(config.csv_path)
    audit = dataset_audit(frame, config.csv_path)
    _write_json(output_root / "dataset_audit.json", audit)

    manifest = {
        "generated_at_utc": utc_timestamp(),
        "csv_path": str(config.csv_path),
        "output_root": str(output_root),
        "max_lag": config.max_lag,
        "top_k": config.top_k,
        "heavy_predictor_limit": config.heavy_predictor_limit,
        "methods": list(config.methods),
        "targets": list(config.targets),
        "implemented_methods": list(PRACTICAL_METHODS + IMPLEMENTED_OPTIONAL_METHODS),
        "optional_methods": list(OPTIONAL_METHODS),
    }
    _write_json(output_root / "analysis_manifest.json", manifest)

    status_rows: list[dict[str, Any]] = []
    synthesis_tables: dict[str, pd.DataFrame] = {}
    synthesis_paths: dict[str, str] = {}

    for target in config.targets:
        if target not in frame.columns:
            raise ValueError(f"Target column missing: {target}")
        predictors = candidate_predictors(frame, target)
        normalized_tables: list[pd.DataFrame] = []
        raw_corr_table = pd.DataFrame()

        if "lagged_correlation" in config.methods:
            for variant in ("raw", "diff1"):
                method_name = f"lagged_correlation_{variant}"
                table = compute_lagged_correlations(frame, target, predictors, config.max_lag, variant)
                if variant == "raw":
                    raw_corr_table = table.copy()
                table_path = tables_dir / f"{target.lower()}__{method_name}.csv"
                table.to_csv(table_path, index=False, encoding="utf-8-sig")
                metadata_path = metadata_dir / f"{target.lower()}__{method_name}.json"
                _write_method_metadata(
                    metadata_path,
                    {
                        "method": method_name,
                        "target": target,
                        "status": "success" if not table.empty else "failed",
                        "variant": variant,
                        "predictor_count": len(predictors),
                        "max_lag": config.max_lag,
                        "preprocessing": {"variant": variant, "scaling": "none", "missing": "dropna after lagging"},
                    },
                )
                if table.empty:
                    status_rows.append(_status_row(method=method_name, target=target, status="failed", reason="no usable lagged-correlation rows were produced", table_path=str(table_path.relative_to(output_root)), metadata_path=str(metadata_path.relative_to(output_root))))
                else:
                    normalized_tables.append(_normalize_ranking_table(table, method_name, target))
                    status_rows.append(_status_row(method=method_name, target=target, status="success", reason="ok", table_path=str(table_path.relative_to(output_root)), metadata_path=str(metadata_path.relative_to(output_root)), row_count=len(table)))

        if "granger" in config.methods:
            method_name = "granger_diff1"
            table, metadata = compute_pairwise_granger(frame, target, predictors, config.max_lag)
            table_path = tables_dir / f"{target.lower()}__{method_name}.csv"
            table.to_csv(table_path, index=False, encoding="utf-8-sig")
            metadata_path = metadata_dir / f"{target.lower()}__{method_name}.json"
            _write_method_metadata(
                metadata_path,
                {
                    "method": method_name,
                    "target": target,
                    "status": "success" if not table.empty else "failed",
                    "variant": "diff1",
                    "predictor_count": len(predictors),
                    "max_lag": config.max_lag,
                    "preprocessing": {"variant": "diff1", "scaling": "none", "missing": "dropna after differencing"},
                    **metadata,
                },
            )
            if table.empty:
                status_rows.append(_status_row(method=method_name, target=target, status="failed", reason="no usable Granger rows were produced", table_path=str(table_path.relative_to(output_root)), metadata_path=str(metadata_path.relative_to(output_root))))
            else:
                normalized_tables.append(_normalize_ranking_table(table, method_name, target))
                status_rows.append(_status_row(method=method_name, target=target, status="success", reason="ok", table_path=str(table_path.relative_to(output_root)), metadata_path=str(metadata_path.relative_to(output_root)), row_count=len(table)))

        priority_predictors = _priority_predictors(normalized_tables, raw_corr_table, predictors, config.heavy_predictor_limit)
        for method in OPTIONAL_METHODS:
            if method not in config.methods:
                continue
            metadata_path = metadata_dir / f"{target.lower()}__{method}.json"
            table_path = tables_dir / f"{target.lower()}__{method}.csv"
            if method not in IMPLEMENTED_OPTIONAL_METHODS:
                status, reason = probe_optional_method(method, repo_root)
                _write_method_metadata(metadata_path, {"method": method, "target": target, "status": status, "reason": reason})
                status_rows.append(_status_row(method=method, target=target, status=status, reason=reason, metadata_path=str(metadata_path.relative_to(output_root))))
                continue
            try:
                table, metadata = _run_method(method=method, target=target, frame=frame, predictors=priority_predictors, config=config, repo_root=repo_root)
                table.to_csv(table_path, index=False, encoding="utf-8-sig")
                metadata_payload = {
                    "method": method,
                    "target": target,
                    "status": "success" if not table.empty else "failed",
                    "priority_predictors": priority_predictors,
                    "max_lag": config.max_lag,
                    **metadata,
                }
                _write_method_metadata(metadata_path, metadata_payload)
                if table.empty:
                    status_rows.append(_status_row(method=method, target=target, status="failed", reason="method executed but produced no validated rows", table_path=str(table_path.relative_to(output_root)), metadata_path=str(metadata_path.relative_to(output_root))))
                else:
                    normalized_tables.append(_normalize_ranking_table(table, method, target))
                    status_rows.append(_status_row(method=method, target=target, status="success", reason="ok", table_path=str(table_path.relative_to(output_root)), metadata_path=str(metadata_path.relative_to(output_root)), row_count=len(table)))
            except ModuleNotFoundError as exc:
                reason = f"missing dependency: {exc}"
                _write_method_metadata(metadata_path, {"method": method, "target": target, "status": "skipped-not-feasible", "reason": reason})
                status_rows.append(_status_row(method=method, target=target, status="skipped-not-feasible", reason=reason, metadata_path=str(metadata_path.relative_to(output_root))))
            except Exception as exc:
                _write_method_metadata(metadata_path, {"method": method, "target": target, "status": "failed", "reason": str(exc), "priority_predictors": priority_predictors})
                status_rows.append(_status_row(method=method, target=target, status="failed", reason=str(exc), metadata_path=str(metadata_path.relative_to(output_root))))

        synthesis = synthesize_rankings(normalized_tables, target, config.top_k)
        synthesis_path = tables_dir / f"{target.lower()}__synthesis.csv"
        synthesis.to_csv(synthesis_path, index=False, encoding="utf-8-sig")
        synthesis_tables[target] = synthesis
        synthesis_paths[target] = str(synthesis_path.relative_to(output_root))

    status_table = pd.DataFrame(status_rows).sort_values(["target", "status", "method"]).reset_index(drop=True)
    status_table.to_csv(output_root / "method_status.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at_utc": utc_timestamp(),
        "output_root": str(output_root),
        "successful_methods": int((status_table["status"] == "success").sum()),
        "non_success_methods": int((status_table["status"] != "success").sum()),
        "targets": {},
    }
    for target, synthesis in synthesis_tables.items():
        summary["targets"][target] = {
            "synthesis_path": synthesis_paths[target],
            "top_candidates": synthesis.head(10).to_dict(orient="records"),
        }
    _write_json(output_root / "summary.json", summary)

    report = render_report(config=config, audit=audit, status_table=status_table, synthesis_paths=synthesis_paths)
    (output_root / "report.md").write_text(report, encoding="utf-8")

    if int((status_table["status"] == "success").sum()) == 0:
        raise RuntimeError("No methods completed successfully; see method_status.csv for details.")

    return {"output_root": output_root, "status_table": status_table, "summary": summary}

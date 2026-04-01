from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tuning.search_space import SearchSpaceContract

NEC_MAIN_KEYS = {"enabled", "config_path"}
NEC_LINKED_KEYS = {
    "history_steps",
    "hist_columns",
    "preprocessing",
    "inference",
    "classifier",
    "normal",
    "extreme",
    "validation",
}
NEC_PREPROCESSING_KEYS = {"mode", "probability_feature", "gmm_components", "epsilon"}
NEC_INFERENCE_KEYS = {"mode", "threshold"}
NEC_VALIDATION_KEYS = {"windows"}
NEC_BRANCH_KEYS = {
    "model",
    "model_params",
    "variables",
    "alpha",
    "beta",
    "oversample_extreme_windows",
}


@dataclass(frozen=True)
class NecPreprocessingConfig:
    mode: str = "diff_std"
    gmm_components: int = 3
    epsilon: float = 1.5


@dataclass(frozen=True)
class NecBranchConfig:
    model: str
    model_params: dict[str, Any] = field(default_factory=dict)
    variables: tuple[str, ...] = field(default_factory=tuple)
    alpha: float | None = None
    beta: float | None = None
    oversample_extreme_windows: bool | None = None


@dataclass(frozen=True)
class NecInferenceConfig:
    mode: str = "soft_weighted"
    threshold: float = 0.5


@dataclass(frozen=True)
class NecValidationConfig:
    windows: int = 8


@dataclass(frozen=True)
class NecConfig:
    enabled: bool = False
    config_path: str | None = None
    preprocessing: NecPreprocessingConfig = field(default_factory=NecPreprocessingConfig)
    inference: NecInferenceConfig = field(default_factory=NecInferenceConfig)
    classifier: NecBranchConfig = field(
        default_factory=lambda: NecBranchConfig(
            model="LSTM",
            alpha=2.0,
            beta=0.5,
            oversample_extreme_windows=True,
        )
    )
    normal: NecBranchConfig = field(
        default_factory=lambda: NecBranchConfig(
            model="LSTM",
            oversample_extreme_windows=False,
        )
    )
    extreme: NecBranchConfig = field(
        default_factory=lambda: NecBranchConfig(
            model="LSTM",
            oversample_extreme_windows=True,
        )
    )
    validation: NecValidationConfig = field(default_factory=NecValidationConfig)


@dataclass(frozen=True)
class NecStageLoadedConfig:
    config: NecConfig
    source_path: Path
    source_type: str
    normalized_payload: dict[str, Any]
    input_hash: str
    resolved_hash: str
    search_space_path: Path | None = None
    search_space_hash: str | None = None
    search_space_payload: dict[str, Any] | None = None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce_positive_int(value: Any, *, field_name: str, allow_none: bool = False) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must be a positive integer")
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _coerce_non_negative_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative float")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative float") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative float")
    return parsed


def _coerce_positive_float(value: Any, *, field_name: str) -> float:
    parsed = _coerce_non_negative_float(value, field_name=field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return parsed


def _coerce_probability(value: Any, *, field_name: str) -> float:
    parsed = _coerce_non_negative_float(value, field_name=field_name)
    if parsed > 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return parsed


def _normalize_model_name(value: Any, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    model_name = str(value).strip()
    if not model_name:
        raise ValueError(f"{field_name} is required")
    return model_name


def _normalize_model_params(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


def _normalize_branch_block(
    payload: Any,
    *,
    section: str,
    coerce_name_tuple: Any,
    unknown_keys: Any,
    coerce_bool: Any,
    defaults: NecBranchConfig,
) -> NecBranchConfig:
    if not isinstance(payload, dict):
        raise ValueError(f"{section} must be a mapping")
    values = dict(payload)
    unknown_keys(values, allowed=NEC_BRANCH_KEYS, section=section)
    if "model" not in values:
        raise ValueError(f"{section}.model is required")
    if "variables" not in values:
        raise ValueError(f"{section}.variables is required")
    return NecBranchConfig(
        model=_normalize_model_name(values.get("model"), field_name=f"{section}.model"),
        model_params=_normalize_model_params(values.get("model_params"), field_name=f"{section}.model_params"),
        variables=coerce_name_tuple(values.get("variables"), field_name=f"{section}.variables"),
        alpha=_coerce_positive_float(
            values.get("alpha", defaults.alpha),
            field_name=f"{section}.alpha",
        )
        if values.get("alpha", defaults.alpha) is not None
        else None,
        beta=_coerce_probability(
            values.get("beta", defaults.beta),
            field_name=f"{section}.beta",
        )
        if values.get("beta", defaults.beta) is not None
        else None,
        oversample_extreme_windows=coerce_bool(
            values.get("oversample_extreme_windows"),
            field_name=f"{section}.oversample_extreme_windows",
            default=defaults.oversample_extreme_windows,
        ),
    )


def normalize_nec_config(value: Any, *, unknown_keys: Any, coerce_bool: Any, coerce_optional_path_string: Any) -> NecConfig:
    if value is None:
        return NecConfig()
    if not isinstance(value, dict):
        raise ValueError("nec must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=NEC_MAIN_KEYS, section="nec")
    enabled = coerce_bool(payload.get("enabled"), field_name="nec.enabled", default=False)
    config_path = coerce_optional_path_string(payload.get("config_path"), field_name="nec.config_path")
    if enabled and config_path is None:
        config_path = "yaml/plugins/nec_lstm.yaml"
    return NecConfig(enabled=enabled, config_path=config_path)


def normalize_linked_nec_config(value: Any, *, unknown_keys: Any, coerce_bool: Any, coerce_name_tuple: Any) -> NecConfig:
    if value is None:
        raise ValueError("nec routed YAML must define a top-level nec block")
    if not isinstance(value, dict):
        raise ValueError("nec routed YAML block must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=NEC_LINKED_KEYS, section="nec")
    default = NecConfig(enabled=True)

    preprocessing_payload = dict(payload.get("preprocessing") or {})
    unknown_keys(preprocessing_payload, allowed=NEC_PREPROCESSING_KEYS, section="nec.preprocessing")
    mode = str(preprocessing_payload.get("mode", default.preprocessing.mode)).strip()
    if mode != "diff_std":
        raise ValueError("nec.preprocessing.mode must be 'diff_std' to preserve paper preprocessing")
    preprocessing = NecPreprocessingConfig(
        mode=mode,
        gmm_components=_coerce_positive_int(
            preprocessing_payload.get("gmm_components", default.preprocessing.gmm_components),
            field_name="nec.preprocessing.gmm_components",
        ),
        epsilon=_coerce_non_negative_float(
            preprocessing_payload.get("epsilon", default.preprocessing.epsilon),
            field_name="nec.preprocessing.epsilon",
        ),
    )

    inference_payload = dict(payload.get("inference") or {})
    unknown_keys(inference_payload, allowed=NEC_INFERENCE_KEYS, section="nec.inference")
    inference_mode = str(inference_payload.get("mode", default.inference.mode)).strip()
    if inference_mode not in {"soft_weighted", "hard_threshold"}:
        raise ValueError("nec.inference.mode must be 'soft_weighted' or 'hard_threshold'")
    inference = NecInferenceConfig(
        mode=inference_mode,
        threshold=_coerce_probability(
            inference_payload.get("threshold", default.inference.threshold),
            field_name="nec.inference.threshold",
        ),
    )

    validation_payload = dict(payload.get("validation") or {})
    unknown_keys(validation_payload, allowed=NEC_VALIDATION_KEYS, section="nec.validation")
    validation = NecValidationConfig(
        windows=_coerce_positive_int(
            validation_payload.get("windows", default.validation.windows),
            field_name="nec.validation.windows",
        )
    )

    removed_top_level = [key for key in ("history_steps", "hist_columns") if key in payload]
    if removed_top_level:
        raise ValueError(
            "nec plugin YAML no longer supports top-level key(s): " + ", ".join(sorted(removed_top_level))
        )
    if "probability_feature" in preprocessing_payload:
        raise ValueError(
            "nec.preprocessing.probability_feature has been removed; probability feature is always enabled"
        )

    classifier = _normalize_branch_block(
        payload.get("classifier"),
        section="nec.classifier",
        coerce_name_tuple=coerce_name_tuple,
        unknown_keys=unknown_keys,
        coerce_bool=coerce_bool,
        defaults=default.classifier,
    )
    normal = _normalize_branch_block(
        payload.get("normal"),
        section="nec.normal",
        coerce_name_tuple=coerce_name_tuple,
        unknown_keys=unknown_keys,
        coerce_bool=coerce_bool,
        defaults=default.normal,
    )
    extreme = _normalize_branch_block(
        payload.get("extreme"),
        section="nec.extreme",
        coerce_name_tuple=coerce_name_tuple,
        unknown_keys=unknown_keys,
        coerce_bool=coerce_bool,
        defaults=default.extreme,
    )

    return NecConfig(
        enabled=True,
        preprocessing=preprocessing,
        inference=inference,
        classifier=classifier,
        normal=normal,
        extreme=extreme,
        validation=validation,
    )


def resolve_nec_route_path(repo_root: Path, nec: NecConfig) -> Path:
    selected = nec.config_path
    if not selected:
        raise ValueError("nec config_path did not resolve a selected config path")
    route_path = Path(selected)
    if route_path.is_absolute():
        return route_path
    return (repo_root / route_path).resolve()


def _branch_to_dict(branch: NecBranchConfig) -> dict[str, Any]:
    return {
        "model": branch.model,
        "model_params": dict(branch.model_params),
        "variables": list(branch.variables),
        "alpha": branch.alpha,
        "beta": branch.beta,
        "oversample_extreme_windows": branch.oversample_extreme_windows,
    }


def nec_branch_configs(config: NecConfig) -> dict[str, NecBranchConfig]:
    return {
        "classifier": config.classifier,
        "normal": config.normal,
        "extreme": config.extreme,
    }


def nec_active_hist_columns(config: NecConfig) -> tuple[str, ...]:
    ordered: list[str] = []
    for branch in nec_branch_configs(config).values():
        for column in branch.variables:
            if column not in ordered:
                ordered.append(column)
    return tuple(ordered)


def nec_to_dict(config: NecConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["classifier"] = _branch_to_dict(config.classifier)
    payload["normal"] = _branch_to_dict(config.normal)
    payload["extreme"] = _branch_to_dict(config.extreme)
    payload["active_hist_columns"] = list(nec_active_hist_columns(config))
    if not payload.get("enabled", False):
        return {"enabled": False, "config_path": payload.get("config_path")}
    return payload


def load_nec_stage1(
    repo_root: Path,
    *,
    source_path: Path,
    source_type: str,
    nec: NecConfig,
    search_space_contract: SearchSpaceContract | None,
) -> NecStageLoadedConfig:
    del source_type, search_space_contract
    from app_config import _coerce_bool, _coerce_name_tuple, _hash_text, _load_document, _resolve_relative_config_reference, _unknown_keys

    selected_config_path = nec.config_path
    if selected_config_path is None:
        raise ValueError("nec enabled but config_path was not resolved")
    stage_source_path = _resolve_relative_config_reference(repo_root, source_path, selected_config_path)
    if not stage_source_path.exists():
        raise FileNotFoundError(f"nec selected route does not exist: {stage_source_path}")
    stage_source_type = "toml" if stage_source_path.suffix.lower() == ".toml" else "yaml"
    raw_text = stage_source_path.read_text(encoding="utf-8")
    stage_payload = _load_document(stage_source_path, stage_source_type)
    if not isinstance(stage_payload, dict):
        raise ValueError(
            "nec config_path must resolve to a mapping with top-level nec key; "
            f"got {type(stage_payload).__name__} from {stage_source_path}"
        )
    supported_top_level = {"nec"}
    extra = sorted(set(stage_payload) - supported_top_level)
    if extra:
        raise ValueError("nec plugin YAML contains unsupported key(s): " + ", ".join(extra))
    linked = normalize_linked_nec_config(
        stage_payload.get("nec"),
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
        coerce_name_tuple=_coerce_name_tuple,
    )
    resolved = replace(linked, enabled=True, config_path=selected_config_path)
    normalized_payload = {"nec": nec_to_dict(resolved)}
    resolved_text = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=False)
    return NecStageLoadedConfig(
        config=resolved,
        source_path=stage_source_path,
        source_type=stage_source_type,
        normalized_payload=normalized_payload,
        input_hash=_hash_text(raw_text),
        resolved_hash=_hash_text(resolved_text),
        search_space_path=None,
        search_space_hash=None,
        search_space_payload=None,
    )

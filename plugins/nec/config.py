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
    "classifier",
    "normal",
    "extreme",
    "validation",
}
NEC_PREPROCESSING_KEYS = {"mode", "probability_feature", "gmm_components", "epsilon"}
NEC_VALIDATION_KEYS = {"windows"}
NEC_FORECAST_KEYS = {
    "hidden_dim",
    "layer_dim",
    "dropout",
    "batch_size",
    "train_volume",
    "epochs",
    "early_stop_patience",
    "encoder_lr",
    "head_lr",
    "oversampling",
    "normal_ratio",
}
NEC_CLASSIFIER_KEYS = NEC_FORECAST_KEYS | {"bce_weight", "mse_weight"}


@dataclass(frozen=True)
class NecPreprocessingConfig:
    mode: str = "diff_std"
    probability_feature: bool = True
    gmm_components: int = 3
    epsilon: float = 1.5


@dataclass(frozen=True)
class NecForecastConfig:
    hidden_dim: int
    layer_dim: int
    dropout: float
    batch_size: int
    train_volume: int
    epochs: int
    early_stop_patience: int
    encoder_lr: float
    head_lr: float
    oversampling: bool
    normal_ratio: float


@dataclass(frozen=True)
class NecClassifierConfig:
    hidden_dim: int
    layer_dim: int
    dropout: float
    batch_size: int
    train_volume: int
    epochs: int
    early_stop_patience: int
    encoder_lr: float
    head_lr: float
    oversampling: bool
    normal_ratio: float
    bce_weight: float
    mse_weight: float


@dataclass(frozen=True)
class NecValidationConfig:
    windows: int = 8


@dataclass(frozen=True)
class NecConfig:
    enabled: bool = False
    config_path: str | None = None
    history_steps: int | None = None
    hist_columns: tuple[str, ...] = field(default_factory=tuple)
    preprocessing: NecPreprocessingConfig = field(default_factory=NecPreprocessingConfig)
    classifier: NecClassifierConfig = field(
        default_factory=lambda: NecClassifierConfig(
            hidden_dim=1024,
            layer_dim=4,
            dropout=0.4,
            batch_size=64,
            train_volume=1000,
            epochs=100,
            early_stop_patience=4,
            encoder_lr=0.001,
            head_lr=0.0005,
            oversampling=True,
            normal_ratio=0.0,
            bce_weight=0.5,
            mse_weight=0.5,
        )
    )
    normal: NecForecastConfig = field(
        default_factory=lambda: NecForecastConfig(
            hidden_dim=1024,
            layer_dim=4,
            dropout=0.4,
            batch_size=64,
            train_volume=180000,
            epochs=80,
            early_stop_patience=3,
            encoder_lr=0.001,
            head_lr=0.0005,
            oversampling=False,
            normal_ratio=0.0,
        )
    )
    extreme: NecForecastConfig = field(
        default_factory=lambda: NecForecastConfig(
            hidden_dim=512,
            layer_dim=4,
            dropout=0.4,
            batch_size=32,
            train_volume=50000,
            epochs=100,
            early_stop_patience=4,
            encoder_lr=0.001,
            head_lr=0.0005,
            oversampling=True,
            normal_ratio=0.0,
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


def _coerce_probability(value: Any, *, field_name: str) -> float:
    parsed = _coerce_non_negative_float(value, field_name=field_name)
    if parsed > 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return parsed


def _normalize_forecast_block(payload: Any, *, section: str, default: NecForecastConfig, unknown_keys: Any) -> NecForecastConfig:
    if payload is None:
        return default
    if not isinstance(payload, dict):
        raise ValueError(f"{section} must be a mapping")
    values = dict(payload)
    unknown_keys(values, allowed=NEC_FORECAST_KEYS, section=section)
    return NecForecastConfig(
        hidden_dim=_coerce_positive_int(values.get("hidden_dim", default.hidden_dim), field_name=f"{section}.hidden_dim"),
        layer_dim=_coerce_positive_int(values.get("layer_dim", default.layer_dim), field_name=f"{section}.layer_dim"),
        dropout=_coerce_probability(values.get("dropout", default.dropout), field_name=f"{section}.dropout"),
        batch_size=_coerce_positive_int(values.get("batch_size", default.batch_size), field_name=f"{section}.batch_size"),
        train_volume=_coerce_positive_int(values.get("train_volume", default.train_volume), field_name=f"{section}.train_volume"),
        epochs=_coerce_positive_int(values.get("epochs", default.epochs), field_name=f"{section}.epochs"),
        early_stop_patience=_coerce_positive_int(values.get("early_stop_patience", default.early_stop_patience), field_name=f"{section}.early_stop_patience"),
        encoder_lr=_coerce_non_negative_float(values.get("encoder_lr", default.encoder_lr), field_name=f"{section}.encoder_lr"),
        head_lr=_coerce_non_negative_float(values.get("head_lr", default.head_lr), field_name=f"{section}.head_lr"),
        oversampling=bool(values.get("oversampling", default.oversampling)),
        normal_ratio=_coerce_probability(values.get("normal_ratio", default.normal_ratio), field_name=f"{section}.normal_ratio"),
    )


def _normalize_classifier_block(payload: Any, *, section: str, default: NecClassifierConfig, unknown_keys: Any) -> NecClassifierConfig:
    if payload is None:
        return default
    if not isinstance(payload, dict):
        raise ValueError(f"{section} must be a mapping")
    values = dict(payload)
    unknown_keys(values, allowed=NEC_CLASSIFIER_KEYS, section=section)
    return NecClassifierConfig(
        hidden_dim=_coerce_positive_int(values.get("hidden_dim", default.hidden_dim), field_name=f"{section}.hidden_dim"),
        layer_dim=_coerce_positive_int(values.get("layer_dim", default.layer_dim), field_name=f"{section}.layer_dim"),
        dropout=_coerce_probability(values.get("dropout", default.dropout), field_name=f"{section}.dropout"),
        batch_size=_coerce_positive_int(values.get("batch_size", default.batch_size), field_name=f"{section}.batch_size"),
        train_volume=_coerce_positive_int(values.get("train_volume", default.train_volume), field_name=f"{section}.train_volume"),
        epochs=_coerce_positive_int(values.get("epochs", default.epochs), field_name=f"{section}.epochs"),
        early_stop_patience=_coerce_positive_int(values.get("early_stop_patience", default.early_stop_patience), field_name=f"{section}.early_stop_patience"),
        encoder_lr=_coerce_non_negative_float(values.get("encoder_lr", default.encoder_lr), field_name=f"{section}.encoder_lr"),
        head_lr=_coerce_non_negative_float(values.get("head_lr", default.head_lr), field_name=f"{section}.head_lr"),
        oversampling=bool(values.get("oversampling", default.oversampling)),
        normal_ratio=_coerce_probability(values.get("normal_ratio", default.normal_ratio), field_name=f"{section}.normal_ratio"),
        bce_weight=_coerce_probability(values.get("bce_weight", default.bce_weight), field_name=f"{section}.bce_weight"),
        mse_weight=_coerce_probability(values.get("mse_weight", default.mse_weight), field_name=f"{section}.mse_weight"),
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
        config_path = "yaml/plugins/nec.yaml"
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
        probability_feature=coerce_bool(preprocessing_payload.get("probability_feature"), field_name="nec.preprocessing.probability_feature", default=default.preprocessing.probability_feature),
        gmm_components=_coerce_positive_int(preprocessing_payload.get("gmm_components", default.preprocessing.gmm_components), field_name="nec.preprocessing.gmm_components"),
        epsilon=_coerce_non_negative_float(preprocessing_payload.get("epsilon", default.preprocessing.epsilon), field_name="nec.preprocessing.epsilon"),
    )

    validation_payload = dict(payload.get("validation") or {})
    unknown_keys(validation_payload, allowed=NEC_VALIDATION_KEYS, section="nec.validation")
    validation = NecValidationConfig(
        windows=_coerce_positive_int(validation_payload.get("windows", default.validation.windows), field_name="nec.validation.windows")
    )
    hist_columns = coerce_name_tuple(payload.get("hist_columns"), field_name="nec.hist_columns")

    return NecConfig(
        enabled=True,
        history_steps=_coerce_positive_int(payload.get("history_steps", default.history_steps), field_name="nec.history_steps", allow_none=True),
        hist_columns=hist_columns,
        preprocessing=preprocessing,
        classifier=_normalize_classifier_block(payload.get("classifier"), section="nec.classifier", default=default.classifier, unknown_keys=unknown_keys),
        normal=_normalize_forecast_block(payload.get("normal"), section="nec.normal", default=default.normal, unknown_keys=unknown_keys),
        extreme=_normalize_forecast_block(payload.get("extreme"), section="nec.extreme", default=default.extreme, unknown_keys=unknown_keys),
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


def nec_to_dict(config: NecConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["hist_columns"] = list(config.hist_columns)
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

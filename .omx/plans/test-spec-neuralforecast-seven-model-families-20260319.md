# Test Spec — NeuralForecast seven external model families

## Test Objectives
- Prove each delivered model class conforms to NeuralForecast package/runtime expectations.
- Prove shared-surface wiring (`models/__init__.py`, `auto.py`, `core.py`) is internally consistent.
- Prove any delivered Auto wrappers satisfy `BaseAuto` argument/config expectations and can execute a minimal fit path.

## Test Inventory
### 1. Package import tests
- Add or extend per-model tests so each new model can be imported from `neuralforecast.auto` or `neuralforecast.models` as appropriate.
- Verify the package export list in `neuralforecast/models/__init__.py:1-44` includes the new model names.

### 2. Direct model behavior tests
For each new model file under `tests/test_models/`:
- `check_model(ModelClass, ["airpassengers"])` or a lighter targeted equivalent, following `tests/test_models/test_xlinear.py:8-10` and `tests/test_models/test_timexer.py:8-9`.
- If a model needs special setup (multivariate, optional dependency, or patch length), pin those kwargs in the test file.

### 3. Auto wrapper tests
For each delivered Auto wrapper:
- `check_args(AutoModel, exclude_args=['cls_model'])` (`tests/test_models/test_helpers.py:7-20`).
- `get_default_config(... backend='optuna')` smoke with a one-step config override.
- `get_default_config(... backend='ray')` smoke with a one-step config override.
- `model.fit(dataset=setup_dataset)` minimal path, matching `tests/test_models/test_xlinear.py:11-33`.

### 4. Registry / serialization tests
- Update or add assertions around `neuralforecast/core.py:138-208` so new entries participate in model lookup.
- If practical, keep this as a lightweight focused test rather than broadening the most expensive existing suites.

### 5. CPU smoke checks
- Run representative `NeuralForecast.fit/predict` smoke checks on CPU for:
  - Lane A representative
  - Lane B representative
  - Lane C representative
  - every delivered Auto wrapper
- Use CPU-safe trainer kwargs when needed: `accelerator='cpu', devices=1, strategy='auto'`.

## Command Matrix
1. `uv run pytest --no-cov tests/test_models/test_nonstationary_transformer.py tests/test_models/test_deepedm.py`
2. `uv run pytest --no-cov tests/test_models/test_mamba.py tests/test_models/test_smamba.py tests/test_models/test_cmamba.py`
3. `uv run pytest --no-cov tests/test_models/test_xlstm_mixer.py tests/test_models/test_duet.py`
4. `uv run pytest --no-cov tests/test_common/test_model_checks.py -k "nonstationary or deepedm or mamba or smamba or cmamba or xlstm or duet"`
5. `uv run pytest --no-cov tests/test_core.py -k "model_filename or cpu or serialize"` (only if new targeted coverage is added there)
6. `pre-commit run --all-files` after code stabilization if the touched surface becomes broad enough.

## Pass / Fail Rules
- PASS only if imports, targeted pytest, and feasible CPU smoke commands all pass.
- FAIL if any new model requires an undeclared dependency, breaks shared-surface wiring, or lacks dedicated targeted coverage.
- PARTIAL is acceptable during implementation only when the blocker is explicitly captured (for example, optional upstream dependency triage), but final completion cannot remain partial.

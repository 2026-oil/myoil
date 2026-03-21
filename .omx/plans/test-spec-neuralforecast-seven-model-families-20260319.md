# Test Spec — NeuralForecast seven external model families

## Test Objectives
- Prove each delivered model class conforms to NeuralForecast package/runtime expectations.
- Prove shared-surface wiring (`models/__init__.py`, `auto.py`, `core.py`) is internally consistent, including save/load registry coverage.
- Prove any delivered Auto wrappers satisfy `BaseAuto` argument/config expectations and can execute a minimal fit path.
- Prove dependency-gated models fail intentionally with an explicit ImportError message if full repo-native runtime support is not yet feasible.

## Test Inventory
### 1. Package import tests
- Add per-model tests so each new model can be imported from the final public surface (`neuralforecast.auto` and/or `neuralforecast.models` as appropriate).
- Verify the export list in `neuralforecast/models/__init__.py:1-44` includes the new model names after Lane D lands.

### 2. Direct model behavior tests
For each new model file under `tests/test_models/`:
- Preferred: `check_model(ModelClass, ["airpassengers"])` or a lighter targeted equivalent, following `tests/test_models/test_xlinear.py:8-10` and `tests/test_models/test_timexer.py:8-9`.
- If a model needs special setup (multivariate, optional dependency, patch length, etc.), pin those kwargs in the test file.
- If a model is dependency-gated, replace full behavior smoke with an explicit ImportError-message assertion instead of silently omitting the test.

### 3. Auto wrapper tests
For each delivered Auto wrapper:
- `check_args(AutoModel, exclude_args=['cls_model'])` (`tests/test_models/test_helpers.py:7-20`).
- `get_default_config(... backend='optuna')` smoke with a one-step config override.
- `get_default_config(... backend='ray')` smoke with a one-step config override.
- `model.fit(dataset=setup_dataset)` minimal path, matching `tests/test_models/test_xlinear.py:11-33`.

### 4. Registry / save-load tests
- Add a focused lightweight test (preferred new file: `tests/test_common/test_model_registry_new_models.py`) that asserts:
  - new names are present in `MODEL_FILENAME_DICT`
  - alias resolution used by `NeuralForecast.save/load` recognizes the delivered models (`neuralforecast/core.py:1815-1914`)
  - supported public import names and registry names stay in sync
- Do not rely on `tests/test_common/test_model_checks.py:7-9` as the primary gate because `check_loss_functions` can swallow exceptions (`neuralforecast/common/_model_checks.py:174-180`).

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
4. `uv run pytest --no-cov tests/test_common/test_model_registry_new_models.py`
5. `uv run pytest --no-cov tests/test_core.py -k "save or load or model_filename"` (only if the targeted registry/save-load coverage is added there instead of a new lightweight file)
6. `pre-commit run --all-files`

## Pass / Fail Rules
- PASS only if imports, targeted pytest, focused registry/save-load coverage, and feasible CPU smoke commands all pass.
- FAIL if any new model requires an undeclared dependency, breaks shared-surface wiring, lacks an intentional dependency-guard test, or is missing from the save/load registry path.
- PARTIAL is acceptable during implementation only when the blocker is explicitly captured, but final completion cannot remain partial.

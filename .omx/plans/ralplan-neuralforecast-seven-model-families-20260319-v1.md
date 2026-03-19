# RALPLAN — NeuralForecast seven external model families (v1)

## Requirements Summary
- The deep-interview spec fixes the objective as repo-native, stable integration of seven requested public model families, not exact upstream parity (`.omx/specs/deep-interview-seven-external-model-families.md:23-27`).
- Scope is the seven model classes plus required shared surfaces, with docs/phase1/new dependencies/paper-score reproduction explicitly out of scope (`.omx/specs/deep-interview-seven-external-model-families.md:29-60`).
- Completion is defined by importability, shared-surface wiring, targeted pytest, and feasible CPU smoke checks (`.omx/specs/deep-interview-seven-external-model-families.md:62-66`).
- The current package surface exposes model imports from `neuralforecast/models/__init__.py:1-44`, Auto wrappers and model re-exports from `neuralforecast/auto.py:4-53`, and serialization/model lookup registry entries from `neuralforecast/core.py:138-208`.
- Existing repo-native patterns worth following are `BaseModel` subclasses in `neuralforecast/common/_base_model.py:83-259`, optional external-package guarding in `neuralforecast/models/xlstm.py:12-18,184-188`, and paired model/auto tests such as `tests/test_models/test_xlinear.py:1-33` and `tests/test_models/test_timexer.py:1-35`.

## RALPLAN-DR Summary
### Principles
1. Prefer repo-native integration over literal upstream reproduction.
2. Preserve current package conventions for exports, registry wiring, and tests.
3. Avoid new dependency requirements; use optional guards or in-repo implementations when necessary.
4. Validate on CPU-first smoke paths on this host.
5. Keep Auto wrapper support intentionally narrower than model-class support if difficulty or dependency burden diverges.

### Decision Drivers
1. User-fixed scope and non-goals from the deep-interview spec.
2. Existing package surfaces require synchronized updates across models, auto wrappers, and registry.
3. Host/runtime constraints favor CPU-only validation and no new dependencies.

### Viable Options
#### Option A — Monolithic single-wave integration
- Add all seven models, all candidate Auto wrappers, all shared-surface wiring, and all tests in one pass.
- Pros: shortest path if everything ports cleanly.
- Cons: highest merge/debug risk; harder to isolate optional-dependency and shape-contract issues.

#### Option B — Preferred phased lane plan with shared-surface consolidation
- Implement model families in bounded lanes, then land shared-surface wiring and tests once each lane proves its anchor models.
- Pros: isolates risk by family, keeps diffs reviewable, matches the repo's existing per-model test pattern.
- Cons: requires explicit integration checkpoints and one shared-surface consolidation pass.

#### Option C — Minimal class-only import support, defer Auto wrappers entirely
- Add seven model classes and direct tests, but skip Auto wrappers in the first delivery.
- Pros: lowest implementation scope.
- Cons: leaves repo integration incomplete relative to current package norms and user-approved decision boundary that allows, not forbids, selective Auto coverage.

### Decision
Choose **Option B**.

### Why Option B wins
- It satisfies the user’s full seven-model scope while preserving the explicit ability to scale Auto support by implementation difficulty (`.omx/specs/deep-interview-seven-external-model-families.md:51-54,68-71`).
- It mirrors existing repo conventions where each model tends to own a dedicated test file while package-wide surfaces (`models/__init__.py`, `auto.py`, `core.py`) are updated centrally.
- It gives a clean place to handle families that may need optional-dependency guards, similar to `xLSTM` (`neuralforecast/models/xlstm.py:12-18,184-188`).

### Invalidation rationale for rejected options
- Option A was rejected because simultaneous model + auto + registry + smoke work across seven upstream codebases is too coupled for first-pass integration.
- Option C was rejected because the repository already exposes Auto wrappers for many comparable models (`neuralforecast/auto.py:4-53`), so a total deferral would under-deliver on repo-native integration.

## Implementation Steps
### Lane 0 — Brownfield guardrails
1. Snapshot current branch status and avoid unrelated dirty-file churn during implementation.
2. Use `BaseModel` constructor/contract semantics as the compatibility target for every new model class (`neuralforecast/common/_base_model.py:83-259`).
3. For any upstream dependency not already present, prefer a local reimplementation or optional import guard rather than adding a package dependency (`.omx/specs/deep-interview-seven-external-model-families.md:56-60`, `neuralforecast/models/xlstm.py:12-18`).

### Lane A — Nonstationary Transformer + DeepEDM anchors
1. Add `neuralforecast/models/nonstationary_transformer.py` and `neuralforecast/models/deepedm.py` as repo-native `BaseModel` subclasses.
2. Normalize constructor signatures to NeuralForecast expectations: `h`, `input_size`, loss args, trainer kwargs, exogenous toggles, and alias/random-seed support.
3. If either upstream method requires helper blocks, keep those helpers colocated in the model module unless shared by another incoming family.
4. Add direct tests mirroring the existing per-model pattern (`tests/test_models/test_xlinear.py:1-33`).
5. Candidate Auto support: **Nonstationary Transformer only by default**. DeepEDM gets Auto support only if its search space stays simple and dependency-free.

### Lane B — Mamba family anchors (Mamba, S-Mamba, CMamba)
1. Add `neuralforecast/models/mamba.py`, `smamba.py`, and `cmamba.py`.
2. Reuse a common internal block/helper module only if it reduces duplication without introducing a new abstraction layer; otherwise keep each module self-contained.
3. Add direct tests for each class plus minimal CPU smoke coverage.
4. Preferred Auto support set: `AutoMamba`, `AutoSMamba`, `AutoCMamba` if search-space definitions can follow the existing `BaseAuto` pattern (`neuralforecast/auto.py:548-615`, `2562-2644`).

### Lane C — xLSTM-Mixer + DUET anchors
1. Add `neuralforecast/models/xlstm_mixer.py` and `duet.py`.
2. Audit whether either family needs optional upstream packages; if yes, gate them like `xLSTM` instead of adding dependencies (`neuralforecast/models/xlstm.py:12-18,184-188`).
3. Add direct tests with the same import/check-model/autowrapper pattern used by TimeXer and XLinear (`tests/test_models/test_timexer.py:1-35`, `tests/test_models/test_xlinear.py:1-33`).
4. Auto support is optional and may be omitted if dependency or configuration complexity is materially higher than the other lanes.

### Lane D — Shared surfaces and registry consolidation
1. Update `neuralforecast/models/__init__.py` to export/import the seven new model classes alongside the existing list (`neuralforecast/models/__init__.py:1-44`).
2. Update `neuralforecast/auto.py` imports, `__all__`, and any new `BaseAuto` subclasses so the auto module remains the canonical import surface for model + Auto pairs (`neuralforecast/auto.py:4-53`, `548-615`, `2562-2644`).
3. Update `neuralforecast/core.py` imports and `MODEL_FILENAME_DICT` for every supported model and supported Auto wrapper so model loading/check_model registry coverage remains coherent (`neuralforecast/core.py:138-208`).
4. Confirm whether `tests/test_common/test_model_checks.py:1-9` automatically picks up the new models through `MODEL_FILENAME_DICT`; if runtime cost is too high, scope registry assertions into lighter targeted tests instead of broadening the slowest suite unnecessarily.

### Lane E — Verification and release gate
1. Add/adjust targeted tests under `tests/test_models/` and any necessary registry/auto tests under `tests/test_common/` or `tests/test_core.py` only where direct coverage is missing.
2. Run lint/type/test/smoke in ascending cost order.
3. Keep CPU-only smoke as the default host-safe validation path, using patterns already present in `tests/test_core.py` CPU config cases.
4. Record unsupported Auto wrappers explicitly in the final implementation report if OMX decides to limit support breadth.

## Acceptance Criteria
1. Seven new model modules exist and are importable from the package surface.
2. `neuralforecast/models/__init__.py`, `neuralforecast/auto.py`, and `neuralforecast/core.py` are internally consistent for every delivered model / Auto wrapper pair.
3. Every delivered model has a dedicated targeted test file following current repo conventions.
4. Every delivered Auto wrapper (if any) has constructor/default-config tests and at least one minimal fit path.
5. Targeted pytest commands for the new surfaces pass.
6. Feasible CPU-only smoke training/prediction checks pass for representative models from each lane.

## Risks and Mitigations
- **Risk:** upstream architectures rely on extra packages not present in this repo.
  - **Mitigation:** prefer local modules or optional import guards; never add new dependencies.
- **Risk:** shared-surface drift causes imports or registry serialization failures.
  - **Mitigation:** treat `models/__init__.py`, `auto.py`, and `core.py` as one consolidation lane with explicit tests.
- **Risk:** Auto wrapper search spaces become the long pole.
  - **Mitigation:** deliver all model classes first, then enable Auto wrappers only for lanes with straightforward configs.
- **Risk:** GPU-oriented upstream defaults fail on this host.
  - **Mitigation:** enforce CPU-only smoke defaults during validation.

## Verification Steps
1. `uv run pytest --no-cov tests/test_models/test_nonstationary_transformer.py tests/test_models/test_deepedm.py`
2. `uv run pytest --no-cov tests/test_models/test_mamba.py tests/test_models/test_smamba.py tests/test_models/test_cmamba.py`
3. `uv run pytest --no-cov tests/test_models/test_xlstm_mixer.py tests/test_models/test_duet.py`
4. `uv run pytest --no-cov tests/test_common/test_model_checks.py -k "nonstationary or deepedm or mamba or smamba or cmamba or xlstm or duet"`
5. `uv run pytest --no-cov tests/test_core.py -k "cpu or serialize or model_filename"` (only if core-level coverage is added/changed)
6. Focused CPU smoke command(s) to be pinned during implementation, using `accelerator='cpu', devices=1, strategy='auto'` where trainer kwargs are needed.

## ADR
- **Decision:** Use a phased lane plan to add all seven model classes, selectively add Auto wrappers, and consolidate shared package surfaces after lane anchors land.
- **Drivers:** user-fixed repo-native scope; no new dependencies; existing package export/registry/test conventions.
- **Alternatives considered:** monolithic single-wave integration; class-only import support without Auto wrappers.
- **Why chosen:** best balance of full scope, low-risk reviewability, and compatibility with current NeuralForecast patterns.
- **Consequences:** implementation is multi-file and should be executed in bounded slices; Auto wrapper breadth may legitimately differ across families.
- **Follow-ups:** execute via ralph or team, then perform a verification pass that re-checks imports, registry, and CPU smoke evidence.

## Available-Agent-Types Roster
- `executor` — implementation across bounded file groups
- `architect` — repo-fit review and tradeoff validation
- `critic` / `code-reviewer` — plan/code review
- `test-engineer` / `verifier` — targeted pytest + smoke design and evidence review
- `dependency-expert` — upstream repo surface and optional-dependency evaluation
- `writer` — final migration/runbook summary if needed

## Follow-up Staffing Guidance
### Ralph path
- 1 `executor` lane for model modules (medium)
- 1 `executor` lane for shared surfaces (`models/__init__.py`, `auto.py`, `core.py`) (medium)
- 1 `test-engineer` lane for new tests and smoke harnesses (medium)
- Optional `dependency-expert` consult if any upstream repo needs package triage (medium)

### Team path
- Worker 1 (`executor`, medium): Lane A anchor models
- Worker 2 (`executor`, medium): Lane B mamba-family models
- Worker 3 (`executor`, medium): Lane C xLSTM-Mixer + DUET
- Worker 4 (`executor` or `test-engineer`, medium): Lane D shared surfaces + verification harness
- Leader follow-up: `verifier` pass to confirm imports/registry/tests before shutdown

## Launch Hints
- Ralph: `$ralph .omx/plans/prd-neuralforecast-seven-model-families-20260319.md`
- Team: `$team .omx/plans/prd-neuralforecast-seven-model-families-20260319.md`
- OMX team CLI equivalent: `omx team start .omx/plans/prd-neuralforecast-seven-model-families-20260319.md`

## Team -> Ralph Verification Path
1. Team proves code integration is landed with targeted test outputs and smoke artifacts.
2. Ralph re-runs the highest-signal verification commands, fixes any residual breakage, and produces the final completion evidence.

## Changelog
- Initial v1 draft from deep-interview spec plus current repository surface inspection.

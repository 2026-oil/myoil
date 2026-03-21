# PRD — NeuralForecast seven external model families

## Problem
NeuralForecast currently exposes many built-in model families through `neuralforecast/models/__init__.py`, `neuralforecast/auto.py`, and `neuralforecast/core.py`, but it does not yet include the seven requested public model families identified in the deep-interview spec (`.omx/specs/deep-interview-seven-external-model-families.md:29-60`). Save/load support also depends on `MODEL_FILENAME_DICT` and alias resolution in `neuralforecast/core.py:138-208,1815-1914`, so adding models without registry parity would leave the package surface incomplete.

## Goal
Add seven official public model families to the repo in a NeuralForecast-native form that is stable, testable, dependency-conscious, and compatible with existing import, Auto, and save/load registry surfaces.

## Users / Jobs To Be Done
- Repo users want to instantiate these new models through the same package surfaces they already use.
- Maintainers want the integration to follow current BaseModel/BaseAuto/test conventions instead of introducing special-case infrastructure.

## In Scope
- Model classes for Nonstationary Transformer, CMamba, Mamba, S-Mamba, xLSTM-Mixer, DUET, DeepEDM.
- Shared-surface updates for `neuralforecast/models/__init__.py`, `neuralforecast/auto.py`, and `neuralforecast/core.py`.
- Targeted tests, registry/save-load coverage, and feasible CPU smoke validation.

## Out of Scope
- Docs updates
- Phase1 changes
- New package dependencies
- Paper-score reproduction
- Exact upstream parity where it harms repo-native integration

## Functional Requirements
1. Each new model is a `BaseModel`-compatible implementation that can be imported from the package model surface (`neuralforecast/common/_base_model.py:83-259`, `neuralforecast/models/__init__.py:1-44`).
2. Supported Auto wrappers follow the current `BaseAuto` signature and config pattern (`neuralforecast/auto.py:17-18,548-615,2562-2644`, `tests/test_models/test_helpers.py:7-20`).
3. The registry in `neuralforecast/core.py:138-208,1815-1914` remains coherent for model lookup and save/load.
4. Every delivered model has either a runnable repo-native path or an explicit delayed ImportError policy with targeted guard coverage.
5. New tests mirror the per-model structure used in existing files like `tests/test_models/test_xlinear.py:1-33` while keeping registry verification focused.

## Non-Functional Requirements
- No new dependencies.
- CPU-first validation.
- Small, reviewable, reversible slices.
- Prefer reuse/deletion over extra abstraction.
- Keep shared-surface changes single-owner to avoid partially wired public imports.

## Support Matrix Policy
- All seven model classes are required.
- Auto wrappers are selective and implementation-difficulty-gated.
- Default candidate Auto support set: Nonstationary Transformer, Mamba, S-Mamba, CMamba.
- DeepEDM, xLSTM-Mixer, and DUET get Auto wrappers only if they remain dependency-free and simple enough to maintain.
- Dependency-gated models must surface an informative ImportError and a matching test if full repo-native implementation is not feasible initially.

## Milestones
1. Anchor model implementation lanes (A/B/C).
2. Shared-surface consolidation lane (D).
3. Verification lane (E).
4. Final evidence report.

## Success Metrics
- Seven model classes import cleanly from the final package surface.
- Shared-surface and registry/save-load tests pass.
- Targeted pytest suites pass.
- CPU smoke checks pass for at least one representative model per lane and for any delivered Auto wrappers.

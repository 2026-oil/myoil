# Deep Interview Spec — seven external model families

## Metadata
- Profile: standard
- Rounds: 4
- Final ambiguity: 0.14
- Threshold: 0.20
- Context type: brownfield
- Context snapshot: `.omx/context/seven-external-model-families-20260319T130131Z.md`
- Transcript: latest `.omx/interviews/seven-external-model-families-*.md`
- Residual risk: none; readiness gates resolved

## Clarity Breakdown
| Dimension | Score |
| --- | ---: |
| Intent | 0.95 |
| Outcome | 0.90 |
| Scope | 0.90 |
| Constraints | 0.80 |
| Success Criteria | 0.90 |
| Context | 0.75 |

## Intent
Integrate the seven requested official public model families into NeuralForecast in a repo-native, stable way.

## Desired Outcome
Users can import the seven model classes, use the wired shared surfaces, pass targeted tests, and run feasible CPU-only smoke checks, without aiming for exact upstream code reproduction or paper-score replication.

## In Scope
- Add these seven model families:
  - Nonstationary Transformer
  - CMamba
  - Mamba
  - S-Mamba
  - xLSTM-Mixer
  - DUET
  - DeepEDM
- Required shared surfaces:
  - `neuralforecast/models/__init__.py`
  - `neuralforecast/auto.py`
  - relevant tests
- Repo-native adaptation to existing NeuralForecast patterns

## Out of Scope / Non-goals
- Docs changes
- Phase1 changes
- Adding new dependencies
- Reproducing paper scores
- Exact 100 percent upstream implementation parity where it conflicts with stable repo-native integration

## Decision Boundaries
OMX may decide without further confirmation:
- implementation order among the seven model families
- Auto wrapper support breadth based on implementation difficulty

## Constraints
- Keep integration repo-native and maintainable
- Do not add new dependencies
- Prefer CPU-only smoke validation where feasible on this host
- Shared surfaces and tests must be updated as needed for supported models

## Testable Acceptance Criteria
1. All seven model classes are importable.
2. Required shared surfaces are wired.
3. Relevant pytest coverage passes.
4. Where feasible, CPU-only minimal fit/predict smoke verification passes.

## Assumptions Exposed and Resolved
- Prior seven-model non-goal policy still applies here: resolved yes.
- Auto wrappers do not need to cover all seven if implementation difficulty is disproportionate: resolved yes, OMX may choose support breadth.
- The objective is stable integration, not exact upstream replication: resolved yes.

## Technical Context Findings
- Repository is brownfield under `neuralforecast/`.
- Target model files do not yet exist under `neuralforecast/models/` in this workspace.
- No `.omx/plans/` artifacts currently exist in this workspace.
- Workspace contains unrelated dirty files, so future execution should avoid unrelated churn.

## Condensed Transcript
- Scope/non-goals reused from prior seven-model lane.
- Auto wrapper breadth delegated to OMX based on implementation difficulty.
- Success criteria fixed to importability, shared surfaces, pytest, and feasible CPU smoke checks.
- Repo-native stable integration prioritized over exact upstream reproduction.

# Deep Interview Spec: merge-blackswan-tlab-df

## Metadata
- Profile: standard
- Rounds: 3
- Final ambiguity: 0.155
- Threshold: 0.20
- Context type: brownfield
- Context snapshot: `.omx/context/merge-blackswan-tlab-df-20260319T131232Z.md`
- Transcript: see latest `.omx/interviews/merge-blackswan-tlab-df-*.md`

## Clarity Breakdown
| Dimension | Score |
| --- | ---: |
| Intent | 0.75 |
| Outcome | 0.90 |
| Scope | 0.88 |
| Constraints | 0.72 |
| Success Criteria | 0.58 |
| Context | 0.92 |

## Intent
Create one modeling-ready weekly dataframe by combining the Tlab weekly dataset with the BlackSwan core index dataset.

## Desired Outcome
Produce a merged CSV under the `data/` folder for downstream research/NeuralForecast use.

## In Scope
- Read `data/BlackSwan_Core_Index_260319.csv`
- Read `data/Data_Weekly_Tlab_260318.csv`
- Normalize `dt` so the two sources share a common date key
- Merge all columns using `dt`
- Use an **outer join**
- Save the final merged CSV under `data/`

## Out of Scope / Non-goals
- Missing-value handling
- Column deletion
- Column rename
- Extra preprocessing beyond the `dt` normalization needed for merge
- Any work beyond generating the merged CSV

## Decision Boundaries
- OMX may normalize `dt` as needed to perform the merge.
- OMX must use an outer join.
- OMX must save the merged output under `data/`.
- Output filename is still implied as `df.csv`, but exact date-string formatting in the final `dt` column was not explicitly confirmed.

## Constraints
- Do not touch unrelated existing workspace changes.
- Prefer minimal, reversible implementation.
- Preserve all source columns.

## Testable Acceptance Criteria
- A merged CSV exists under `data/df.csv`.
- The merged file contains all columns from both inputs.
- Merge key is `dt` after normalization.
- Join semantics are outer join.
- No additional imputation, dropping, renaming, or transformations are performed.

## Assumptions Exposed + Resolutions
- Join type ambiguity: resolved to outer join.
- Scope ambiguity: resolved to merge-only with no extra preprocessing.
- Output location ambiguity: resolved to `data/`.
- Final `dt` display format: left as implementation choice unless clarified later.

## Technical Context Findings
- `BlackSwan_Core_Index_260319.csv` uses `YYYY-MM-DD` dates.
- `Data_Weekly_Tlab_260318.csv` uses `YYYY.M.D` dates.
- Both files have 584 rows, suggesting aligned weekly cadence but not guaranteeing perfect row identity.

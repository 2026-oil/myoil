# Autopilot Spec - merge-blackswan-tlab-df

Input source of truth: `.omx/specs/deep-interview-merge-blackswan-tlab-df.md`

## Goal
Create `data/df.csv` by combining:
- `data/BlackSwan_Core_Index_260319.csv`
- `data/Data_Weekly_Tlab_260318.csv`

## Contract
- Normalize `dt`
- Merge all columns on `dt`
- Join type: outer
- Output path: `data/df.csv`
- Preserve all source columns
- No missing-value handling
- No column rename or deletion
- No extra preprocessing beyond merge-enabling `dt` normalization

## Implementation notes
- Prefer stdlib CSV tooling to avoid introducing dependency assumptions.
- Emit UTF-8 with BOM (`utf-8-sig`) for consistency with prior research CSV handling.

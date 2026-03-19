# Autopilot Implementation Plan - merge-blackswan-tlab-df

1. Read both CSVs with `utf-8-sig` handling.
2. Normalize each `dt` into canonical `YYYY-MM-DD`.
3. Build a union of all dates (outer join semantics).
4. Preserve Tlab columns first, then append BlackSwan columns.
5. Write merged result to `data/df.csv`.
6. Verify row count, column count, date ordering, and presence of source-exclusive values.

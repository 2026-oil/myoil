# WTI / Brent Correlation Analysis for `data/df.csv`

## Scope
- Raw targets: `Com_CrudeOil`, `Com_BrentCrudeOil`
- Differenced targets: `diff(Com_CrudeOil)`, `diff(Com_BrentCrudeOil)`
- Method: Pearson correlation over the numeric frame; differenced targets use first-differenced numeric values.
- Presentation outputs round coefficients to 3 decimals.

## Artifact Inventory
- Tables: `tables/`
- Figures: `figures/`
- Summary: `summary.json`

## Target: `Com_CrudeOil`
- rows: 89
- n_obs range: 584..584
- raw CSV: `tables/raw_Com_CrudeOil_correlations.csv`
- display CSV: `tables/raw_Com_CrudeOil_correlations_display.csv`
- markdown table: `tables/raw_Com_CrudeOil_correlations.md`
- bar image: `figures/raw_Com_CrudeOil_correlations_bar.png`
- full table image: `figures/raw_Com_CrudeOil_correlations_table.png`
- strongest positives: `Com_BrentCrudeOil` (0.992), `Com_Gasoline` (0.960), `Com_BloombergCommodity_BCOM` (0.823), `Com_LME_Ni_Cash` (0.791), `Com_Coal` (0.771)
- strongest negatives: `Com_LME_Ni_Inv` (-0.704), `Com_LME_Al_Inv` (-0.594), `Com_Wool` (-0.503), `Com_LME_Zn_Inv` (-0.442), `Com_LME_Cu_Inv` (-0.404)

## Target: `Com_BrentCrudeOil`
- rows: 89
- n_obs range: 584..584
- raw CSV: `tables/raw_Com_BrentCrudeOil_correlations.csv`
- display CSV: `tables/raw_Com_BrentCrudeOil_correlations_display.csv`
- markdown table: `tables/raw_Com_BrentCrudeOil_correlations.md`
- bar image: `figures/raw_Com_BrentCrudeOil_correlations_bar.png`
- full table image: `figures/raw_Com_BrentCrudeOil_correlations_table.png`
- strongest positives: `Com_CrudeOil` (0.992), `Com_Gasoline` (0.951), `Com_BloombergCommodity_BCOM` (0.805), `Com_LME_Ni_Cash` (0.781), `Com_Coal` (0.764)
- strongest negatives: `Com_LME_Ni_Inv` (-0.713), `Com_LME_Al_Inv` (-0.599), `Com_LME_Zn_Inv` (-0.473), `Com_Wool` (-0.448), `Com_LME_Cu_Inv` (-0.392)

## Target: `diff(Com_CrudeOil)`
- rows: 89
- n_obs range: 583..583
- raw CSV: `tables/diff_Com_CrudeOil_correlations.csv`
- display CSV: `tables/diff_Com_CrudeOil_correlations_display.csv`
- markdown table: `tables/diff_Com_CrudeOil_correlations.md`
- bar image: `figures/diff_Com_CrudeOil_correlations_bar.png`
- full table image: `figures/diff_Com_CrudeOil_correlations_table.png`
- strongest positives: `Com_BrentCrudeOil` (0.944), `Com_Gasoline` (0.781), `Com_BloombergCommodity_BCOM` (0.725), `Com_LME_Al_Cash` (0.366), `Com_PalmOil` (0.358)
- strongest negatives: `Idx_OVX` (-0.369), `BS_Core_Index_A` (-0.340), `BS_Core_Index_Integrated` (-0.324), `Idx_SnPVIX` (-0.268), `EX_USD_BRL` (-0.264)

## Target: `diff(Com_BrentCrudeOil)`
- rows: 89
- n_obs range: 583..583
- raw CSV: `tables/diff_Com_BrentCrudeOil_correlations.csv`
- display CSV: `tables/diff_Com_BrentCrudeOil_correlations_display.csv`
- markdown table: `tables/diff_Com_BrentCrudeOil_correlations.md`
- bar image: `figures/diff_Com_BrentCrudeOil_correlations_bar.png`
- full table image: `figures/diff_Com_BrentCrudeOil_correlations_table.png`
- strongest positives: `Com_CrudeOil` (0.944), `Com_Gasoline` (0.817), `Com_BloombergCommodity_BCOM` (0.764), `Com_LME_Al_Cash` (0.369), `Com_PalmOil` (0.367)
- strongest negatives: `Idx_SnPVIX` (-0.280), `BS_Core_Index_A` (-0.273), `BS_Core_Index_Integrated` (-0.263), `EX_USD_BRL` (-0.262), `Idx_OVX` (-0.246)


# df.csv Multicollinearity Analysis

## Method
- Analysis unit: target-specific design matrix
- Targets: `Com_CrudeOil`, `Com_BrentCrudeOil`
- Levels: raw and first difference
- Statsmodels anchor: `variance_inflation_factor`, `OLSResults.condition_number`, `OLSResults.eigenvals`
- Condition/eigen diagnostics use z-scored predictors plus intercept for scale comparability across mixed-unit variables.
- For readability, VIF figures cap infinite values at 110 percent of the largest finite plotted VIF; CSV tables keep the raw infinite values.

## Fixed Rule Set
- `VIF > 5`: high multicollinearity
- `VIF > 10`: severe multicollinearity
- `|corr| >= 0.8`: supporting pairwise-correlation flag
- `condition number > 30`: matrix-conditioning warning
- `near-zero eigenvalue` or eigenvalue-ratio collapse (`min/max <= 1e-6`): near-singularity warning

## Artifact Inventory
- summary: `summary.json`
- tables: `tables/`
- figures: `figures/`

## Target: `Com_CrudeOil`

### Raw vs diff comparison
- high-VIF counts: raw=86, diff1=23
- severe-VIF counts: raw=82, diff1=19
- persisting high-VIF variables: BS_Core_Index_A, BS_Core_Index_B, BS_Core_Index_C, BS_Core_Index_Integrated, Bonds_AUS_10Y, Bonds_MOVE, Bonds_US_10Y, Bonds_US_1Y, Bonds_US_2Y, Com_BloombergCommodity_BCOM, Com_BrentCrudeOil, Com_LMEX
- raw-only high-VIF variables: Bonds_AUS_1Y, Bonds_BRZ_10Y, Bonds_BRZ_1Y, Bonds_CHN_10Y, Bonds_CHN_1Y, Bonds_CHN_20Y, Bonds_CHN_2Y, Bonds_CHN_30Y, Bonds_CHN_5Y, Bonds_IND_10Y, Bonds_IND_1Y, Bonds_KOR_10Y
- diff-only high-VIF variables: none
- comparison figure: `figures/vif_count_compare_com_crudeoil.png`

### raw
- observations used: 584
- predictors used: 88
- high-VIF predictors: 86 | severe predictors: 82
- condition number: 2251814774404938.50 (warning)
- min eigenvalue: 3.9328e-27; eigenvalue ratio: 1.97213e-31; near singularity: yes
- top VIF figure: `figures/top_vif_com_crudeoil_raw.png`
- flagged correlation heatmap: `figures/flagged_corr_heatmap_com_crudeoil_raw.png`
- full VIF table: `tables/vif_com_crudeoil_raw.csv`
- flagged pairwise correlations: `tables/flagged_pairwise_com_crudeoil_raw.csv`
- grouped proposals: `tables/group_summary_com_crudeoil_raw.csv`
- variable reduction suggestions:
- group_2: keep `Com_Cotton` as the first representative candidate; members [BS_Core_Index_B, Bonds_AUS_10Y, Bonds_AUS_1Y, Bonds_CHN_10Y, Bonds_CHN_1Y, Bonds_CHN_20Y, Bonds_CHN_2Y, Bonds_CHN_30Y, Bonds_CHN_5Y, Bonds_KOR_10Y, Bonds_KOR_1Y, Bonds_MOVE, Bonds_US_10Y, Bonds_US_1Y, Bonds_US_2Y, Bonds_US_3M, Com_Barley, Com_BloombergCommodity_BCOM, Com_BrentCrudeOil, Com_Canola, Com_Coal, Com_Cocoa, Com_Coffee, Com_Corn, Com_Cotton, Com_Gasoline, Com_Gold, Com_LMEX, Com_LME_Al_Cash, Com_LME_Cu_Cash, Com_LME_Ni_Cash, Com_LME_Sn_Cash, Com_LME_Zn_Cash, Com_NaturalGas, Com_Oat, Com_PalmOil, Com_Silver, Com_Soybeans, Com_SunflowerOil, Com_Uranium, Com_Wheat, Com_Wool, EX_AUD_USD, EX_INR_USD, EX_USD_BRL, EX_USD_CNY, EX_USD_JPY, EX_USD_KRW, Idx_DxyUSD, Idx_HangSeng, Idx_SnP500, Idx_SnPGlobal1200] form a redundant block (max VIF=inf, max |corr|=0.996).
- group_1: keep `Idx_OVX` as the first representative candidate; members [BS_Core_Index_A, BS_Core_Index_Integrated, Idx_GVZ, Idx_OVX, Idx_SnPVIX] form a redundant block (max VIF=inf, max |corr|=0.897).
- group_7: keep `Idx_Shanghai50` as the first representative candidate; members [Idx_CH50, Idx_CSI300, Idx_Shanghai50] form a redundant block (max VIF=683.41, max |corr|=0.962).
- group_3: keep `Bonds_BRZ_10Y` as the first representative candidate; members [Bonds_BRZ_10Y, Bonds_BRZ_1Y] form a redundant block (max VIF=207.58, max |corr|=0.934).
- group_4: keep `Bonds_IND_10Y` as the first representative candidate; members [Bonds_IND_10Y, Bonds_IND_1Y] form a redundant block (max VIF=175.36, max |corr|=0.861).
- group_6: keep `Com_LME_Zn_Inv` as the first representative candidate; members [Com_LME_Al_Inv, Com_LME_Ni_Inv, Com_LME_Zn_Inv] form a redundant block (max VIF=107.89, max |corr|=0.863).
- group_5: keep `Com_Cheese` as the first representative candidate; members [Com_Cheese, Com_Milk] form a redundant block (max VIF=43.65, max |corr|=0.938).

### diff1
- observations used: 583
- predictors used: 88
- high-VIF predictors: 23 | severe predictors: 19
- condition number: 2787659043020064.00 (warning)
- min eigenvalue: 8.33013e-28; eigenvalue ratio: 1.28683e-31; near singularity: yes
- top VIF figure: `figures/top_vif_com_crudeoil_diff1.png`
- flagged correlation heatmap: `figures/flagged_corr_heatmap_com_crudeoil_diff1.png`
- full VIF table: `tables/vif_com_crudeoil_diff1.csv`
- flagged pairwise correlations: `tables/flagged_pairwise_com_crudeoil_diff1.csv`
- grouped proposals: `tables/group_summary_com_crudeoil_diff1.csv`
- variable reduction suggestions:
- group_1: keep `BS_Core_Index_A` as the first representative candidate; members [BS_Core_Index_A, BS_Core_Index_Integrated, Idx_SnPVIX] form a redundant block (max VIF=inf, max |corr|=0.939).
- group_2: keep `BS_Core_Index_B` as the first representative candidate; members [BS_Core_Index_B, Bonds_MOVE] form a redundant block (max VIF=562949953421312.00, max |corr|=0.862).
- group_8: keep `Idx_SnP500` as the first representative candidate; members [Idx_SnP500, Idx_SnPGlobal1200] form a redundant block (max VIF=42.81, max |corr|=0.963).
- group_7: keep `Idx_Shanghai` as the first representative candidate; members [Idx_CH50, Idx_CSI300, Idx_Shanghai, Idx_Shanghai50] form a redundant block (max VIF=32.03, max |corr|=0.957).
- group_6: keep `Com_LME_Cu_Cash` as the first representative candidate; members [Com_LMEX, Com_LME_Cu_Cash] form a redundant block (max VIF=26.75, max |corr|=0.903).
- group_4: keep `Bonds_US_1Y` as the first representative candidate; members [Bonds_US_1Y, Bonds_US_2Y] form a redundant block (max VIF=10.83, max |corr|=0.864).
- group_3: keep `Bonds_AUS_10Y` as the first representative candidate; members [Bonds_AUS_10Y, Bonds_US_10Y] form a redundant block (max VIF=9.18, max |corr|=0.826).
- group_5: keep `Com_Gasoline` as the first representative candidate; members [Com_BrentCrudeOil, Com_Gasoline] form a redundant block (max VIF=8.89, max |corr|=0.817).

## Target: `Com_BrentCrudeOil`

### Raw vs diff comparison
- high-VIF counts: raw=86, diff1=23
- severe-VIF counts: raw=82, diff1=19
- persisting high-VIF variables: BS_Core_Index_A, BS_Core_Index_B, BS_Core_Index_C, BS_Core_Index_Integrated, Bonds_AUS_10Y, Bonds_MOVE, Bonds_US_10Y, Bonds_US_1Y, Bonds_US_2Y, Com_BloombergCommodity_BCOM, Com_CrudeOil, Com_LMEX
- raw-only high-VIF variables: Bonds_AUS_1Y, Bonds_BRZ_10Y, Bonds_BRZ_1Y, Bonds_CHN_10Y, Bonds_CHN_1Y, Bonds_CHN_20Y, Bonds_CHN_2Y, Bonds_CHN_30Y, Bonds_CHN_5Y, Bonds_IND_10Y, Bonds_IND_1Y, Bonds_KOR_10Y
- diff-only high-VIF variables: none
- comparison figure: `figures/vif_count_compare_com_brentcrudeoil.png`

### raw
- observations used: 584
- predictors used: 88
- high-VIF predictors: 86 | severe predictors: 82
- condition number: 2251814785099849.00 (warning)
- min eigenvalue: 3.93733e-27; eigenvalue ratio: 1.97213e-31; near singularity: yes
- top VIF figure: `figures/top_vif_com_brentcrudeoil_raw.png`
- flagged correlation heatmap: `figures/flagged_corr_heatmap_com_brentcrudeoil_raw.png`
- full VIF table: `tables/vif_com_brentcrudeoil_raw.csv`
- flagged pairwise correlations: `tables/flagged_pairwise_com_brentcrudeoil_raw.csv`
- grouped proposals: `tables/group_summary_com_brentcrudeoil_raw.csv`
- variable reduction suggestions:
- group_2: keep `Com_Cotton` as the first representative candidate; members [BS_Core_Index_B, Bonds_AUS_10Y, Bonds_AUS_1Y, Bonds_CHN_10Y, Bonds_CHN_1Y, Bonds_CHN_20Y, Bonds_CHN_2Y, Bonds_CHN_30Y, Bonds_CHN_5Y, Bonds_KOR_10Y, Bonds_KOR_1Y, Bonds_MOVE, Bonds_US_10Y, Bonds_US_1Y, Bonds_US_2Y, Bonds_US_3M, Com_Barley, Com_BloombergCommodity_BCOM, Com_Canola, Com_Coal, Com_Cocoa, Com_Coffee, Com_Corn, Com_Cotton, Com_CrudeOil, Com_Gasoline, Com_Gold, Com_LMEX, Com_LME_Al_Cash, Com_LME_Cu_Cash, Com_LME_Ni_Cash, Com_LME_Sn_Cash, Com_LME_Zn_Cash, Com_NaturalGas, Com_Oat, Com_PalmOil, Com_Silver, Com_Soybeans, Com_SunflowerOil, Com_Uranium, Com_Wheat, Com_Wool, EX_AUD_USD, EX_INR_USD, EX_USD_BRL, EX_USD_CNY, EX_USD_JPY, EX_USD_KRW, Idx_DxyUSD, Idx_HangSeng, Idx_SnP500, Idx_SnPGlobal1200] form a redundant block (max VIF=inf, max |corr|=0.996).
- group_1: keep `Idx_OVX` as the first representative candidate; members [BS_Core_Index_A, BS_Core_Index_Integrated, Idx_GVZ, Idx_OVX, Idx_SnPVIX] form a redundant block (max VIF=inf, max |corr|=0.897).
- group_7: keep `Idx_Shanghai50` as the first representative candidate; members [Idx_CH50, Idx_CSI300, Idx_Shanghai50] form a redundant block (max VIF=676.98, max |corr|=0.962).
- group_3: keep `Bonds_BRZ_10Y` as the first representative candidate; members [Bonds_BRZ_10Y, Bonds_BRZ_1Y] form a redundant block (max VIF=207.70, max |corr|=0.934).
- group_4: keep `Bonds_IND_10Y` as the first representative candidate; members [Bonds_IND_10Y, Bonds_IND_1Y] form a redundant block (max VIF=174.99, max |corr|=0.861).
- group_6: keep `Com_LME_Zn_Inv` as the first representative candidate; members [Com_LME_Al_Inv, Com_LME_Ni_Inv, Com_LME_Zn_Inv] form a redundant block (max VIF=108.22, max |corr|=0.863).
- group_5: keep `Com_Cheese` as the first representative candidate; members [Com_Cheese, Com_Milk] form a redundant block (max VIF=43.75, max |corr|=0.938).

### diff1
- observations used: 583
- predictors used: 88
- high-VIF predictors: 23 | severe predictors: 19
- condition number: 2787756974950238.50 (warning)
- min eigenvalue: 8.32221e-28; eigenvalue ratio: 1.28674e-31; near singularity: yes
- top VIF figure: `figures/top_vif_com_brentcrudeoil_diff1.png`
- flagged correlation heatmap: `figures/flagged_corr_heatmap_com_brentcrudeoil_diff1.png`
- full VIF table: `tables/vif_com_brentcrudeoil_diff1.csv`
- flagged pairwise correlations: `tables/flagged_pairwise_com_brentcrudeoil_diff1.csv`
- grouped proposals: `tables/group_summary_com_brentcrudeoil_diff1.csv`
- variable reduction suggestions:
- group_1: keep `BS_Core_Index_Integrated` as the first representative candidate; members [BS_Core_Index_A, BS_Core_Index_Integrated, Idx_SnPVIX] form a redundant block (max VIF=inf, max |corr|=0.939).
- group_2: keep `BS_Core_Index_B` as the first representative candidate; members [BS_Core_Index_B, Bonds_MOVE] form a redundant block (max VIF=3002399751580330.50, max |corr|=0.862).
- group_7: keep `Idx_SnP500` as the first representative candidate; members [Idx_SnP500, Idx_SnPGlobal1200] form a redundant block (max VIF=42.64, max |corr|=0.963).
- group_6: keep `Idx_Shanghai` as the first representative candidate; members [Idx_CH50, Idx_CSI300, Idx_Shanghai, Idx_Shanghai50] form a redundant block (max VIF=32.02, max |corr|=0.957).
- group_5: keep `Com_LME_Cu_Cash` as the first representative candidate; members [Com_LMEX, Com_LME_Cu_Cash] form a redundant block (max VIF=26.90, max |corr|=0.903).
- group_4: keep `Bonds_US_1Y` as the first representative candidate; members [Bonds_US_1Y, Bonds_US_2Y] form a redundant block (max VIF=10.83, max |corr|=0.864).
- group_3: keep `Bonds_AUS_10Y` as the first representative candidate; members [Bonds_AUS_10Y, Bonds_US_10Y] form a redundant block (max VIF=9.14, max |corr|=0.826).


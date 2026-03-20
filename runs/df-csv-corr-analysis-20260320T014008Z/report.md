# WTI / Brent Correlation Analysis for `data/df.csv`

## Scope
- Targets: `Com_CrudeOil` (WTI), `Com_BrentCrudeOil` (Brent)
- Methods: raw Pearson correlation; first-differenced Pearson correlation (`dt` excluded from differencing)
- Rows: 584
- Numeric columns excluding `dt`: 89
- Other variables compared against each target: 87

## Artifact Summary
- Raw bar chart: `runs/df-csv-corr-analysis-20260320T014008Z/raw_target_vs_rest_bar.png`
- Diff bar chart: `runs/df-csv-corr-analysis-20260320T014008Z/diff_target_vs_rest_bar.png`
- Raw heatmap: `runs/df-csv-corr-analysis-20260320T014008Z/raw_target_vs_rest_heatmap.png`
- Diff heatmap: `runs/df-csv-corr-analysis-20260320T014008Z/diff_target_vs_rest_heatmap.png`

## Executive Summary
### Raw Pearson
- **WTI (Com_CrudeOil)**: strongest positives include `Com_Gasoline` (0.960), `Com_BloombergCommodity_BCOM` (0.823), `Com_LME_Ni_Cash` (0.791), `Com_Coal` (0.771), `Com_Canola` (0.759). strongest negatives include `Com_LME_Ni_Inv` (-0.704), `Com_LME_Al_Inv` (-0.594), `Com_Wool` (-0.503), `Com_LME_Zn_Inv` (-0.442), `Com_LME_Cu_Inv` (-0.404). Magnitude counts: |r|>=0.7 = 16, |r|>=0.5 = 42, |r|>=0.3 = 56.
- **Brent (Com_BrentCrudeOil)**: strongest positives include `Com_Gasoline` (0.951), `Com_BloombergCommodity_BCOM` (0.805), `Com_LME_Ni_Cash` (0.781), `Com_Coal` (0.764), `Com_Cotton` (0.738). strongest negatives include `Com_LME_Ni_Inv` (-0.713), `Com_LME_Al_Inv` (-0.599), `Com_LME_Zn_Inv` (-0.473), `Com_Wool` (-0.448), `Com_LME_Cu_Inv` (-0.392). Magnitude counts: |r|>=0.7 = 13, |r|>=0.5 = 37, |r|>=0.3 = 55.
- **Common patterns**: `Com_Gasoline` (WTI 0.960, Brent 0.951), `Com_BloombergCommodity_BCOM` (WTI 0.823, Brent 0.805), `Com_LME_Ni_Cash` (WTI 0.791, Brent 0.781), `Com_Coal` (WTI 0.771, Brent 0.764), `Com_Cotton` (WTI 0.755, Brent 0.738), `Com_LME_Al_Cash` (WTI 0.753, Brent 0.734), `Com_Canola` (WTI 0.759, Brent 0.720), `Bonds_KOR_10Y` (WTI 0.736, Brent 0.733), `Com_LMEX` (WTI 0.740, Brent 0.719), `Com_Barley` (WTI 0.735, Brent 0.724).
- **Largest WTI vs Brent differences**: `Com_Wool` (gap 0.055; WTI -0.503, Brent -0.448), `Com_Sugar` (gap 0.053; WTI 0.537, Brent 0.484), `Com_PalmOil` (gap 0.051; WTI 0.736, Brent 0.685), `Bonds_IND_10Y` (gap 0.046; WTI 0.146, Brent 0.192), `Com_LME_Zn_Spread` (gap 0.044; WTI -0.126, Brent -0.170), `Com_SunflowerOil` (gap 0.042; WTI 0.629, Brent 0.586), `Idx_HangSeng` (gap 0.041; WTI -0.318, Brent -0.277), `Bonds_CHN_10Y` (gap 0.040; WTI -0.280, Brent -0.240).
- **Interpretation points**: Raw correlations are more likely to reflect shared long-run trend levels, while differenced correlations emphasize co-movement in weekly changes. Variables that stay large in both views are stronger candidates for stable co-movement; variables that collapse after differencing are more trend-driven than shock-driven.

### Differenced Pearson
- **WTI (Com_CrudeOil)**: strongest positives include `Com_Gasoline` (0.781), `Com_BloombergCommodity_BCOM` (0.725), `Com_LME_Al_Cash` (0.366), `Com_PalmOil` (0.358), `Com_LMEX` (0.337). strongest negatives include `Idx_OVX` (-0.369), `BS_Core_Index_A` (-0.340), `BS_Core_Index_Integrated` (-0.324), `Idx_SnPVIX` (-0.268), `EX_USD_BRL` (-0.264). Magnitude counts: |r|>=0.7 = 2, |r|>=0.5 = 2, |r|>=0.3 = 8.
- **Brent (Com_BrentCrudeOil)**: strongest positives include `Com_Gasoline` (0.817), `Com_BloombergCommodity_BCOM` (0.764), `Com_LME_Al_Cash` (0.369), `Com_PalmOil` (0.367), `Com_LMEX` (0.364). strongest negatives include `Idx_SnPVIX` (-0.280), `BS_Core_Index_A` (-0.273), `BS_Core_Index_Integrated` (-0.263), `EX_USD_BRL` (-0.262), `Idx_OVX` (-0.246). Magnitude counts: |r|>=0.7 = 2, |r|>=0.5 = 2, |r|>=0.3 = 5.
- **Common patterns**: `Com_Gasoline` (WTI 0.781, Brent 0.817), `Com_BloombergCommodity_BCOM` (WTI 0.725, Brent 0.764), `Com_LME_Al_Cash` (WTI 0.366, Brent 0.369), `Com_PalmOil` (WTI 0.358, Brent 0.367), `Com_LMEX` (WTI 0.337, Brent 0.364).
- **Largest WTI vs Brent differences**: `Idx_OVX` (gap 0.123; WTI -0.369, Brent -0.246), `BS_Core_Index_A` (gap 0.066; WTI -0.340, Brent -0.273), `BS_Core_Index_Integrated` (gap 0.061; WTI -0.324, Brent -0.263), `Bonds_US_3M` (gap 0.040; WTI 0.122, Brent 0.162), `Com_BloombergCommodity_BCOM` (gap 0.039; WTI 0.725, Brent 0.764), `Bonds_CHN_30Y` (gap 0.038; WTI 0.098, Brent 0.137), `Com_Gasoline` (gap 0.035; WTI 0.781, Brent 0.817), `Com_Uranium` (gap 0.035; WTI 0.036, Brent 0.071).
- **Interpretation points**: Raw correlations are more likely to reflect shared long-run trend levels, while differenced correlations emphasize co-movement in weekly changes. Variables that stay large in both views are stronger candidates for stable co-movement; variables that collapse after differencing are more trend-driven than shock-driven.

## Raw Pearson Tables
### WTI (Com_CrudeOil) vs Rest

| variable | pearson_corr | abs_corr | sign | n_obs |
| --- | --- | --- | --- | --- |
| Com_Gasoline | 0.9596 | 0.9596 | positive | 584 |
| Com_BloombergCommodity_BCOM | 0.8232 | 0.8232 | positive | 584 |
| Com_LME_Ni_Cash | 0.7914 | 0.7914 | positive | 584 |
| Com_Coal | 0.7706 | 0.7706 | positive | 584 |
| Com_Canola | 0.7594 | 0.7594 | positive | 584 |
| Com_Cotton | 0.7548 | 0.7548 | positive | 584 |
| Com_LME_Al_Cash | 0.7531 | 0.7531 | positive | 584 |
| Com_LMEX | 0.7399 | 0.7399 | positive | 584 |
| Bonds_KOR_10Y | 0.7362 | 0.7362 | positive | 584 |
| Com_PalmOil | 0.7362 | 0.7362 | positive | 584 |
| Com_Barley | 0.7352 | 0.7352 | positive | 584 |
| Com_Corn | 0.7349 | 0.7349 | positive | 584 |
| Com_Oat | 0.7269 | 0.7269 | positive | 584 |
| Com_Wheat | 0.7244 | 0.7244 | positive | 584 |
| Com_Soybeans | 0.7129 | 0.7129 | positive | 584 |
| Com_LME_Zn_Cash | 0.6868 | 0.6868 | positive | 584 |
| Com_LME_Cu_Cash | 0.6618 | 0.6618 | positive | 584 |
| Com_SunflowerOil | 0.6287 | 0.6287 | positive | 584 |
| Com_LME_Sn_Cash | 0.6278 | 0.6278 | positive | 584 |
| Bonds_US_2Y | 0.6237 | 0.6237 | positive | 584 |
| Bonds_KOR_1Y | 0.6181 | 0.6181 | positive | 584 |
| Com_Rice | 0.6137 | 0.6137 | positive | 584 |
| Com_NaturalGas | 0.6055 | 0.6055 | positive | 584 |
| Bonds_US_1Y | 0.5880 | 0.5880 | positive | 584 |
| Bonds_MOVE | 0.5736 | 0.5736 | positive | 584 |
| Bonds_US_10Y | 0.5722 | 0.5722 | positive | 584 |
| Com_HRC_Steel | 0.5600 | 0.5600 | positive | 584 |
| BS_Core_Index_B | 0.5522 | 0.5522 | positive | 584 |
| Idx_SnP500 | 0.5371 | 0.5371 | positive | 584 |
| Com_Sugar | 0.5368 | 0.5368 | positive | 584 |
| Com_Milk | 0.5326 | 0.5326 | positive | 584 |
| EX_USD_JPY | 0.5325 | 0.5325 | positive | 584 |
| Idx_SnPGlobal1200 | 0.5260 | 0.5260 | positive | 584 |
| Bonds_US_3M | 0.5227 | 0.5227 | positive | 584 |
| EX_INR_USD | 0.5204 | 0.5204 | positive | 584 |
| Com_Uranium | 0.5177 | 0.5177 | positive | 584 |
| Bonds_AUS_10Y | 0.5065 | 0.5065 | positive | 584 |
| Com_Steel | 0.5040 | 0.5040 | positive | 584 |
| Com_Lumber | 0.5011 | 0.5011 | positive | 584 |
| Com_Iron_Ore | 0.4870 | 0.4870 | positive | 584 |
| Com_LME_Pb_Cash | 0.4759 | 0.4759 | positive | 584 |
| Idx_DxyUSD | 0.4649 | 0.4649 | positive | 584 |
| EX_USD_KRW | 0.4371 | 0.4371 | positive | 584 |
| Com_OrangeJuice | 0.4334 | 0.4334 | positive | 584 |
| EX_USD_BRL | 0.4233 | 0.4233 | positive | 584 |
| Bonds_AUS_1Y | 0.4224 | 0.4224 | positive | 584 |
| Com_Cheese | 0.4163 | 0.4163 | positive | 584 |
| Com_Coffee | 0.4149 | 0.4149 | positive | 584 |
| Bonds_BRZ_1Y | 0.3026 | 0.3026 | positive | 584 |
| Com_Gold | 0.2880 | 0.2880 | positive | 584 |
| Idx_CH50 | 0.2617 | 0.2617 | positive | 584 |
| BS_Core_Index_C | 0.2510 | 0.2510 | positive | 584 |
| Com_Cocoa | 0.2388 | 0.2388 | positive | 584 |
| Com_Silver | 0.2241 | 0.2241 | positive | 584 |
| Bonds_BRZ_10Y | 0.2110 | 0.2110 | positive | 584 |
| EX_USD_CNY | 0.1871 | 0.1871 | positive | 584 |
| BS_Core_Index_Integrated | 0.1794 | 0.1794 | positive | 584 |
| Idx_CSI300 | 0.1761 | 0.1761 | positive | 584 |
| Bonds_IND_10Y | 0.1462 | 0.1462 | positive | 584 |
| Com_LME_Cu_Spread | 0.1322 | 0.1322 | positive | 584 |
| Idx_Shanghai50 | 0.1120 | 0.1120 | positive | 584 |
| Com_LME_Ni_Spread | 0.0993 | 0.0993 | positive | 584 |
| Com_LME_Al_Spread | 0.0841 | 0.0841 | positive | 584 |
| Idx_Shanghai | 0.0746 | 0.0746 | positive | 584 |
| Bonds_IND_1Y | 0.0684 | 0.0684 | positive | 584 |
| Idx_SnPVIX | -0.0532 | 0.0532 | negative | 584 |
| Com_LME_Sn_Spread | -0.0935 | 0.0935 | negative | 584 |
| Com_LME_Zn_Spread | -0.1259 | 0.1259 | negative | 584 |
| Com_LME_Pb_Spread | -0.1275 | 0.1275 | negative | 584 |
| Idx_GVZ | -0.1383 | 0.1383 | negative | 584 |
| BS_Core_Index_A | -0.1813 | 0.1813 | negative | 584 |
| Com_LME_Sn_Inv | -0.1997 | 0.1997 | negative | 584 |
| Bonds_CHN_5Y | -0.2618 | 0.2618 | negative | 584 |
| Bonds_CHN_20Y | -0.2752 | 0.2752 | negative | 584 |
| Com_LME_Pb_Inv | -0.2774 | 0.2774 | negative | 584 |
| Bonds_CHN_2Y | -0.2795 | 0.2795 | negative | 584 |
| Bonds_CHN_10Y | -0.2802 | 0.2802 | negative | 584 |
| EX_AUD_USD | -0.2835 | 0.2835 | negative | 584 |
| Idx_OVX | -0.2873 | 0.2873 | negative | 584 |
| Bonds_CHN_1Y | -0.2936 | 0.2936 | negative | 584 |
| Idx_HangSeng | -0.3184 | 0.3184 | negative | 584 |
| Bonds_CHN_30Y | -0.3546 | 0.3546 | negative | 584 |
| Com_LME_Cu_Inv | -0.4036 | 0.4036 | negative | 584 |
| Com_LME_Zn_Inv | -0.4418 | 0.4418 | negative | 584 |
| Com_Wool | -0.5034 | 0.5034 | negative | 584 |
| Com_LME_Al_Inv | -0.5943 | 0.5943 | negative | 584 |
| Com_LME_Ni_Inv | -0.7043 | 0.7043 | negative | 584 |

### Brent (Com_BrentCrudeOil) vs Rest

| variable | pearson_corr | abs_corr | sign | n_obs |
| --- | --- | --- | --- | --- |
| Com_Gasoline | 0.9513 | 0.9513 | positive | 584 |
| Com_BloombergCommodity_BCOM | 0.8049 | 0.8049 | positive | 584 |
| Com_LME_Ni_Cash | 0.7812 | 0.7812 | positive | 584 |
| Com_Coal | 0.7644 | 0.7644 | positive | 584 |
| Com_Cotton | 0.7381 | 0.7381 | positive | 584 |
| Com_LME_Al_Cash | 0.7340 | 0.7340 | positive | 584 |
| Bonds_KOR_10Y | 0.7329 | 0.7329 | positive | 584 |
| Com_Barley | 0.7236 | 0.7236 | positive | 584 |
| Com_Canola | 0.7203 | 0.7203 | positive | 584 |
| Com_LMEX | 0.7191 | 0.7191 | positive | 584 |
| Com_Corn | 0.7106 | 0.7106 | positive | 584 |
| Com_Wheat | 0.7057 | 0.7057 | positive | 584 |
| Com_Oat | 0.6942 | 0.6942 | positive | 584 |
| Com_LME_Zn_Cash | 0.6874 | 0.6874 | positive | 584 |
| Com_PalmOil | 0.6852 | 0.6852 | positive | 584 |
| Com_Soybeans | 0.6759 | 0.6759 | positive | 584 |
| Com_LME_Cu_Cash | 0.6395 | 0.6395 | positive | 584 |
| Bonds_US_2Y | 0.6393 | 0.6393 | positive | 584 |
| Bonds_KOR_1Y | 0.6269 | 0.6269 | positive | 584 |
| Bonds_US_1Y | 0.6037 | 0.6037 | positive | 584 |
| Com_Rice | 0.5993 | 0.5993 | positive | 584 |
| Com_NaturalGas | 0.5960 | 0.5960 | positive | 584 |
| Com_LME_Sn_Cash | 0.5948 | 0.5948 | positive | 584 |
| Com_SunflowerOil | 0.5864 | 0.5864 | positive | 584 |
| Bonds_US_10Y | 0.5792 | 0.5792 | positive | 584 |
| Bonds_MOVE | 0.5554 | 0.5554 | positive | 584 |
| BS_Core_Index_B | 0.5403 | 0.5403 | positive | 584 |
| Com_HRC_Steel | 0.5372 | 0.5372 | positive | 584 |
| Bonds_US_3M | 0.5370 | 0.5370 | positive | 584 |
| Com_Steel | 0.5198 | 0.5198 | positive | 584 |
| EX_USD_JPY | 0.5186 | 0.5186 | positive | 584 |
| Idx_SnP500 | 0.5122 | 0.5122 | positive | 584 |
| Com_Milk | 0.5030 | 0.5030 | positive | 584 |
| Idx_SnPGlobal1200 | 0.5024 | 0.5024 | positive | 584 |
| EX_INR_USD | 0.5021 | 0.5021 | positive | 584 |
| Com_Uranium | 0.4969 | 0.4969 | positive | 584 |
| Bonds_AUS_10Y | 0.4947 | 0.4947 | positive | 584 |
| Com_Sugar | 0.4837 | 0.4837 | positive | 584 |
| Com_LME_Pb_Cash | 0.4752 | 0.4752 | positive | 584 |
| Com_Lumber | 0.4695 | 0.4695 | positive | 584 |
| Idx_DxyUSD | 0.4612 | 0.4612 | positive | 584 |
| Com_Iron_Ore | 0.4602 | 0.4602 | positive | 584 |
| Bonds_AUS_1Y | 0.4220 | 0.4220 | positive | 584 |
| EX_USD_KRW | 0.4138 | 0.4138 | positive | 584 |
| Com_OrangeJuice | 0.4079 | 0.4079 | positive | 584 |
| EX_USD_BRL | 0.3921 | 0.3921 | positive | 584 |
| Com_Cheese | 0.3866 | 0.3866 | positive | 584 |
| Com_Coffee | 0.3791 | 0.3791 | positive | 584 |
| Bonds_BRZ_1Y | 0.2662 | 0.2662 | positive | 584 |
| Com_Gold | 0.2639 | 0.2639 | positive | 584 |
| Idx_CH50 | 0.2524 | 0.2524 | positive | 584 |
| BS_Core_Index_C | 0.2435 | 0.2435 | positive | 584 |
| Com_Cocoa | 0.2132 | 0.2132 | positive | 584 |
| Com_Silver | 0.2002 | 0.2002 | positive | 584 |
| Bonds_IND_10Y | 0.1923 | 0.1923 | positive | 584 |
| EX_USD_CNY | 0.1875 | 0.1875 | positive | 584 |
| Bonds_BRZ_10Y | 0.1802 | 0.1802 | positive | 584 |
| BS_Core_Index_Integrated | 0.1647 | 0.1647 | positive | 584 |
| Idx_CSI300 | 0.1546 | 0.1546 | positive | 584 |
| Com_LME_Cu_Spread | 0.1333 | 0.1333 | positive | 584 |
| Idx_Shanghai50 | 0.1099 | 0.1099 | positive | 584 |
| Com_LME_Ni_Spread | 0.1063 | 0.1063 | positive | 584 |
| Bonds_IND_1Y | 0.1037 | 0.1037 | positive | 584 |
| Com_LME_Al_Spread | 0.1023 | 0.1023 | positive | 584 |
| Idx_Shanghai | 0.0478 | 0.0478 | positive | 584 |
| Idx_SnPVIX | -0.0629 | 0.0629 | negative | 584 |
| Com_LME_Sn_Spread | -0.0722 | 0.0722 | negative | 584 |
| Com_LME_Pb_Spread | -0.1198 | 0.1198 | negative | 584 |
| Com_LME_Zn_Spread | -0.1700 | 0.1700 | negative | 584 |
| Idx_GVZ | -0.1722 | 0.1722 | negative | 584 |
| Com_LME_Sn_Inv | -0.1919 | 0.1919 | negative | 584 |
| BS_Core_Index_A | -0.1936 | 0.1936 | negative | 584 |
| Bonds_CHN_5Y | -0.2250 | 0.2250 | negative | 584 |
| Bonds_CHN_10Y | -0.2403 | 0.2403 | negative | 584 |
| Bonds_CHN_20Y | -0.2446 | 0.2446 | negative | 584 |
| Bonds_CHN_2Y | -0.2471 | 0.2471 | negative | 584 |
| Bonds_CHN_1Y | -0.2643 | 0.2643 | negative | 584 |
| Idx_HangSeng | -0.2773 | 0.2773 | negative | 584 |
| Idx_OVX | -0.2777 | 0.2777 | negative | 584 |
| EX_AUD_USD | -0.2850 | 0.2850 | negative | 584 |
| Com_LME_Pb_Inv | -0.3025 | 0.3025 | negative | 584 |
| Bonds_CHN_30Y | -0.3155 | 0.3155 | negative | 584 |
| Com_LME_Cu_Inv | -0.3922 | 0.3922 | negative | 584 |
| Com_Wool | -0.4483 | 0.4483 | negative | 584 |
| Com_LME_Zn_Inv | -0.4730 | 0.4730 | negative | 584 |
| Com_LME_Al_Inv | -0.5992 | 0.5992 | negative | 584 |
| Com_LME_Ni_Inv | -0.7130 | 0.7130 | negative | 584 |

## First-differenced Pearson Tables
### WTI (Com_CrudeOil) vs Rest

| variable | pearson_corr | abs_corr | sign | n_obs |
| --- | --- | --- | --- | --- |
| Com_Gasoline | 0.7811 | 0.7811 | positive | 583 |
| Com_BloombergCommodity_BCOM | 0.7251 | 0.7251 | positive | 583 |
| Com_LME_Al_Cash | 0.3659 | 0.3659 | positive | 583 |
| Com_PalmOil | 0.3584 | 0.3584 | positive | 583 |
| Com_LMEX | 0.3368 | 0.3368 | positive | 583 |
| Com_LME_Cu_Cash | 0.2825 | 0.2825 | positive | 583 |
| EX_AUD_USD | 0.2618 | 0.2618 | positive | 583 |
| Com_Soybeans | 0.2555 | 0.2555 | positive | 583 |
| Idx_SnPGlobal1200 | 0.2494 | 0.2494 | positive | 583 |
| Idx_HangSeng | 0.2454 | 0.2454 | positive | 583 |
| Bonds_US_10Y | 0.2346 | 0.2346 | positive | 583 |
| Bonds_US_2Y | 0.2278 | 0.2278 | positive | 583 |
| Idx_SnP500 | 0.2252 | 0.2252 | positive | 583 |
| Com_LME_Zn_Cash | 0.2232 | 0.2232 | positive | 583 |
| Com_SunflowerOil | 0.2197 | 0.2197 | positive | 583 |
| Com_Wheat | 0.2151 | 0.2151 | positive | 583 |
| Com_Canola | 0.2093 | 0.2093 | positive | 583 |
| Com_Sugar | 0.2088 | 0.2088 | positive | 583 |
| Com_Coal | 0.2027 | 0.2027 | positive | 583 |
| Com_Cotton | 0.1953 | 0.1953 | positive | 583 |
| Bonds_IND_10Y | 0.1925 | 0.1925 | positive | 583 |
| Com_LME_Pb_Cash | 0.1824 | 0.1824 | positive | 583 |
| Com_Corn | 0.1820 | 0.1820 | positive | 583 |
| Bonds_US_1Y | 0.1815 | 0.1815 | positive | 583 |
| Com_LME_Sn_Cash | 0.1804 | 0.1804 | positive | 583 |
| Bonds_AUS_10Y | 0.1715 | 0.1715 | positive | 583 |
| Bonds_KOR_10Y | 0.1699 | 0.1699 | positive | 583 |
| Com_NaturalGas | 0.1679 | 0.1679 | positive | 583 |
| Idx_CSI300 | 0.1638 | 0.1638 | positive | 583 |
| Idx_Shanghai50 | 0.1603 | 0.1603 | positive | 583 |
| Idx_CH50 | 0.1598 | 0.1598 | positive | 583 |
| Bonds_IND_1Y | 0.1555 | 0.1555 | positive | 583 |
| Idx_Shanghai | 0.1453 | 0.1453 | positive | 583 |
| EX_USD_JPY | 0.1379 | 0.1379 | positive | 583 |
| Com_Iron_Ore | 0.1348 | 0.1348 | positive | 583 |
| Bonds_US_3M | 0.1217 | 0.1217 | positive | 583 |
| Bonds_CHN_10Y | 0.1213 | 0.1213 | positive | 583 |
| Bonds_CHN_5Y | 0.1100 | 0.1100 | positive | 583 |
| Bonds_AUS_1Y | 0.1048 | 0.1048 | positive | 583 |
| Bonds_CHN_30Y | 0.0982 | 0.0982 | positive | 583 |
| Bonds_CHN_20Y | 0.0921 | 0.0921 | positive | 583 |
| Com_Milk | 0.0908 | 0.0908 | positive | 583 |
| Com_Cocoa | 0.0814 | 0.0814 | positive | 583 |
| Com_Rice | 0.0811 | 0.0811 | positive | 583 |
| Com_LME_Ni_Cash | 0.0726 | 0.0726 | positive | 583 |
| Com_Coffee | 0.0691 | 0.0691 | positive | 583 |
| Com_Steel | 0.0678 | 0.0678 | positive | 583 |
| Com_Silver | 0.0657 | 0.0657 | positive | 583 |
| Bonds_CHN_2Y | 0.0621 | 0.0621 | positive | 583 |
| Com_Gold | 0.0580 | 0.0580 | positive | 583 |
| Com_Cheese | 0.0467 | 0.0467 | positive | 583 |
| Com_LME_Zn_Spread | 0.0391 | 0.0391 | positive | 583 |
| Com_OrangeJuice | 0.0367 | 0.0367 | positive | 583 |
| Com_LME_Cu_Spread | 0.0359 | 0.0359 | positive | 583 |
| Com_Lumber | 0.0358 | 0.0358 | positive | 583 |
| Com_Uranium | 0.0357 | 0.0357 | positive | 583 |
| Com_Oat | 0.0350 | 0.0350 | positive | 583 |
| Bonds_KOR_1Y | 0.0337 | 0.0337 | positive | 583 |
| EX_INR_USD | 0.0257 | 0.0257 | positive | 583 |
| Com_Wool | 0.0184 | 0.0184 | positive | 583 |
| Bonds_BRZ_1Y | 0.0157 | 0.0157 | positive | 583 |
| Com_LME_Pb_Inv | 0.0136 | 0.0136 | positive | 583 |
| Bonds_CHN_1Y | 0.0122 | 0.0122 | positive | 583 |
| Com_LME_Zn_Inv | 0.0112 | 0.0112 | positive | 583 |
| Com_HRC_Steel | 0.0009 | 0.0009 | positive | 583 |
| Com_LME_Sn_Inv | -0.0011 | 0.0011 | negative | 583 |
| Com_LME_Ni_Inv | -0.0106 | 0.0106 | negative | 583 |
| Com_Barley | -0.0168 | 0.0168 | negative | 583 |
| Com_LME_Pb_Spread | -0.0220 | 0.0220 | negative | 583 |
| Com_LME_Al_Inv | -0.0227 | 0.0227 | negative | 583 |
| Com_LME_Cu_Inv | -0.0228 | 0.0228 | negative | 583 |
| Bonds_BRZ_10Y | -0.0481 | 0.0481 | negative | 583 |
| Idx_DxyUSD | -0.0493 | 0.0493 | negative | 583 |
| Com_LME_Ni_Spread | -0.0594 | 0.0594 | negative | 583 |
| Com_LME_Sn_Spread | -0.0671 | 0.0671 | negative | 583 |
| Com_LME_Al_Spread | -0.0706 | 0.0706 | negative | 583 |
| EX_USD_KRW | -0.0716 | 0.0716 | negative | 583 |
| BS_Core_Index_C | -0.0846 | 0.0846 | negative | 583 |
| EX_USD_CNY | -0.0980 | 0.0980 | negative | 583 |
| Idx_GVZ | -0.1269 | 0.1269 | negative | 583 |
| BS_Core_Index_B | -0.1346 | 0.1346 | negative | 583 |
| Bonds_MOVE | -0.1353 | 0.1353 | negative | 583 |
| EX_USD_BRL | -0.2643 | 0.2643 | negative | 583 |
| Idx_SnPVIX | -0.2681 | 0.2681 | negative | 583 |
| BS_Core_Index_Integrated | -0.3237 | 0.3237 | negative | 583 |
| BS_Core_Index_A | -0.3396 | 0.3396 | negative | 583 |
| Idx_OVX | -0.3693 | 0.3693 | negative | 583 |

### Brent (Com_BrentCrudeOil) vs Rest

| variable | pearson_corr | abs_corr | sign | n_obs |
| --- | --- | --- | --- | --- |
| Com_Gasoline | 0.8166 | 0.8166 | positive | 583 |
| Com_BloombergCommodity_BCOM | 0.7641 | 0.7641 | positive | 583 |
| Com_LME_Al_Cash | 0.3685 | 0.3685 | positive | 583 |
| Com_PalmOil | 0.3665 | 0.3665 | positive | 583 |
| Com_LMEX | 0.3641 | 0.3641 | positive | 583 |
| Com_LME_Cu_Cash | 0.2999 | 0.2999 | positive | 583 |
| EX_AUD_USD | 0.2653 | 0.2653 | positive | 583 |
| Idx_SnPGlobal1200 | 0.2640 | 0.2640 | positive | 583 |
| Idx_HangSeng | 0.2618 | 0.2618 | positive | 583 |
| Bonds_US_2Y | 0.2602 | 0.2602 | positive | 583 |
| Com_Soybeans | 0.2579 | 0.2579 | positive | 583 |
| Bonds_US_10Y | 0.2560 | 0.2560 | positive | 583 |
| Com_SunflowerOil | 0.2517 | 0.2517 | positive | 583 |
| Com_Wheat | 0.2469 | 0.2469 | positive | 583 |
| Com_LME_Zn_Cash | 0.2376 | 0.2376 | positive | 583 |
| Idx_SnP500 | 0.2359 | 0.2359 | positive | 583 |
| Com_Coal | 0.2188 | 0.2188 | positive | 583 |
| Bonds_US_1Y | 0.2138 | 0.2138 | positive | 583 |
| Com_Canola | 0.2128 | 0.2128 | positive | 583 |
| Com_Cotton | 0.2094 | 0.2094 | positive | 583 |
| Bonds_IND_10Y | 0.2067 | 0.2067 | positive | 583 |
| Com_Sugar | 0.2049 | 0.2049 | positive | 583 |
| Com_LME_Sn_Cash | 0.1964 | 0.1964 | positive | 583 |
| Com_LME_Pb_Cash | 0.1960 | 0.1960 | positive | 583 |
| Com_Corn | 0.1914 | 0.1914 | positive | 583 |
| Idx_CSI300 | 0.1902 | 0.1902 | positive | 583 |
| Idx_Shanghai50 | 0.1882 | 0.1882 | positive | 583 |
| Bonds_KOR_10Y | 0.1868 | 0.1868 | positive | 583 |
| Bonds_AUS_10Y | 0.1842 | 0.1842 | positive | 583 |
| Com_NaturalGas | 0.1809 | 0.1809 | positive | 583 |
| Idx_CH50 | 0.1804 | 0.1804 | positive | 583 |
| Idx_Shanghai | 0.1793 | 0.1793 | positive | 583 |
| EX_USD_JPY | 0.1628 | 0.1628 | positive | 583 |
| Bonds_US_3M | 0.1618 | 0.1618 | positive | 583 |
| Com_Iron_Ore | 0.1590 | 0.1590 | positive | 583 |
| Bonds_IND_1Y | 0.1412 | 0.1412 | positive | 583 |
| Bonds_CHN_10Y | 0.1402 | 0.1402 | positive | 583 |
| Bonds_CHN_30Y | 0.1366 | 0.1366 | positive | 583 |
| Bonds_CHN_5Y | 0.1217 | 0.1217 | positive | 583 |
| Bonds_CHN_20Y | 0.1151 | 0.1151 | positive | 583 |
| Bonds_AUS_1Y | 0.1132 | 0.1132 | positive | 583 |
| Com_Steel | 0.1007 | 0.1007 | positive | 583 |
| Com_Milk | 0.0981 | 0.0981 | positive | 583 |
| Com_LME_Ni_Cash | 0.0952 | 0.0952 | positive | 583 |
| Com_Rice | 0.0918 | 0.0918 | positive | 583 |
| Com_Cocoa | 0.0866 | 0.0866 | positive | 583 |
| Com_Silver | 0.0821 | 0.0821 | positive | 583 |
| Com_Gold | 0.0775 | 0.0775 | positive | 583 |
| Com_Uranium | 0.0708 | 0.0708 | positive | 583 |
| Com_Coffee | 0.0685 | 0.0685 | positive | 583 |
| Com_Lumber | 0.0674 | 0.0674 | positive | 583 |
| Bonds_CHN_2Y | 0.0605 | 0.0605 | positive | 583 |
| Com_LME_Zn_Spread | 0.0592 | 0.0592 | positive | 583 |
| Com_LME_Cu_Spread | 0.0556 | 0.0556 | positive | 583 |
| Com_Cheese | 0.0547 | 0.0547 | positive | 583 |
| EX_INR_USD | 0.0461 | 0.0461 | positive | 583 |
| Bonds_KOR_1Y | 0.0379 | 0.0379 | positive | 583 |
| Com_Oat | 0.0345 | 0.0345 | positive | 583 |
| Com_Wool | 0.0333 | 0.0333 | positive | 583 |
| Com_OrangeJuice | 0.0308 | 0.0308 | positive | 583 |
| Com_HRC_Steel | 0.0194 | 0.0194 | positive | 583 |
| Com_LME_Pb_Inv | 0.0150 | 0.0150 | positive | 583 |
| Bonds_CHN_1Y | 0.0134 | 0.0134 | positive | 583 |
| Com_LME_Zn_Inv | 0.0133 | 0.0133 | positive | 583 |
| Bonds_BRZ_1Y | 0.0068 | 0.0068 | positive | 583 |
| Com_LME_Sn_Inv | 0.0065 | 0.0065 | positive | 583 |
| Com_LME_Pb_Spread | -0.0109 | 0.0109 | negative | 583 |
| Com_LME_Cu_Inv | -0.0131 | 0.0131 | negative | 583 |
| Com_LME_Ni_Inv | -0.0198 | 0.0198 | negative | 583 |
| Com_Barley | -0.0229 | 0.0229 | negative | 583 |
| Com_LME_Al_Inv | -0.0240 | 0.0240 | negative | 583 |
| Idx_DxyUSD | -0.0271 | 0.0271 | negative | 583 |
| Com_LME_Al_Spread | -0.0492 | 0.0492 | negative | 583 |
| Bonds_BRZ_10Y | -0.0557 | 0.0557 | negative | 583 |
| EX_USD_KRW | -0.0586 | 0.0586 | negative | 583 |
| Com_LME_Ni_Spread | -0.0619 | 0.0619 | negative | 583 |
| Com_LME_Sn_Spread | -0.0628 | 0.0628 | negative | 583 |
| BS_Core_Index_C | -0.0663 | 0.0663 | negative | 583 |
| Idx_GVZ | -0.0920 | 0.0920 | negative | 583 |
| EX_USD_CNY | -0.1019 | 0.1019 | negative | 583 |
| BS_Core_Index_B | -0.1220 | 0.1220 | negative | 583 |
| Bonds_MOVE | -0.1337 | 0.1337 | negative | 583 |
| Idx_OVX | -0.2462 | 0.2462 | negative | 583 |
| EX_USD_BRL | -0.2616 | 0.2616 | negative | 583 |
| BS_Core_Index_Integrated | -0.2632 | 0.2632 | negative | 583 |
| BS_Core_Index_A | -0.2734 | 0.2734 | negative | 583 |
| Idx_SnPVIX | -0.2797 | 0.2797 | negative | 583 |

## Common Pattern Tables
### Raw: common high-magnitude same-sign variables (WTI vs Brent)

| variable | wti_corr | brent_corr | joint_strength |
| --- | --- | --- | --- |
| Com_Gasoline | 0.9596 | 0.9513 | 1.9110 |
| Com_BloombergCommodity_BCOM | 0.8232 | 0.8049 | 1.6281 |
| Com_LME_Ni_Cash | 0.7914 | 0.7812 | 1.5726 |
| Com_Coal | 0.7706 | 0.7644 | 1.5350 |
| Com_Cotton | 0.7548 | 0.7381 | 1.4929 |
| Com_LME_Al_Cash | 0.7531 | 0.7340 | 1.4870 |
| Com_Canola | 0.7594 | 0.7203 | 1.4797 |
| Bonds_KOR_10Y | 0.7362 | 0.7329 | 1.4691 |
| Com_LMEX | 0.7399 | 0.7191 | 1.4590 |
| Com_Barley | 0.7352 | 0.7236 | 1.4588 |

### Raw: largest WTI-Brent correlation gaps

| variable | wti_corr | brent_corr | corr_gap |
| --- | --- | --- | --- |
| Com_Wool | -0.5034 | -0.4483 | 0.0551 |
| Com_Sugar | 0.5368 | 0.4837 | 0.0531 |
| Com_PalmOil | 0.7362 | 0.6852 | 0.0510 |
| Bonds_IND_10Y | 0.1462 | 0.1923 | 0.0460 |
| Com_LME_Zn_Spread | -0.1259 | -0.1700 | 0.0441 |
| Com_SunflowerOil | 0.6287 | 0.5864 | 0.0423 |
| Idx_HangSeng | -0.3184 | -0.2773 | 0.0410 |
| Bonds_CHN_10Y | -0.2802 | -0.2403 | 0.0398 |
| Bonds_CHN_30Y | -0.3546 | -0.3155 | 0.0391 |
| Com_Canola | 0.7594 | 0.7203 | 0.0390 |

### Differenced: common high-magnitude same-sign variables (WTI vs Brent)

| variable | wti_corr | brent_corr | joint_strength |
| --- | --- | --- | --- |
| Com_Gasoline | 0.7811 | 0.8166 | 1.5977 |
| Com_BloombergCommodity_BCOM | 0.7251 | 0.7641 | 1.4892 |
| Com_LME_Al_Cash | 0.3659 | 0.3685 | 0.7345 |
| Com_PalmOil | 0.3584 | 0.3665 | 0.7250 |
| Com_LMEX | 0.3368 | 0.3641 | 0.7009 |

### Differenced: largest WTI-Brent correlation gaps

| variable | wti_corr | brent_corr | corr_gap |
| --- | --- | --- | --- |
| Idx_OVX | -0.3693 | -0.2462 | 0.1231 |
| BS_Core_Index_A | -0.3396 | -0.2734 | 0.0661 |
| BS_Core_Index_Integrated | -0.3237 | -0.2632 | 0.0605 |
| Bonds_US_3M | 0.1217 | 0.1618 | 0.0401 |
| Com_BloombergCommodity_BCOM | 0.7251 | 0.7641 | 0.0390 |
| Bonds_CHN_30Y | 0.0982 | 0.1366 | 0.0384 |
| Com_Gasoline | 0.7811 | 0.8166 | 0.0355 |
| Com_Uranium | 0.0357 | 0.0708 | 0.0351 |
| Idx_GVZ | -0.1269 | -0.0920 | 0.0349 |
| Idx_Shanghai | 0.1453 | 0.1793 | 0.0340 |

## Output Paths
- report: `runs/df-csv-corr-analysis-20260320T014008Z/report.md`
- raw_wti_csv: `runs/df-csv-corr-analysis-20260320T014008Z/raw_Com_CrudeOil_correlations.csv`
- raw_brent_csv: `runs/df-csv-corr-analysis-20260320T014008Z/raw_Com_BrentCrudeOil_correlations.csv`
- diff_wti_csv: `runs/df-csv-corr-analysis-20260320T014008Z/diff_Com_CrudeOil_correlations.csv`
- diff_brent_csv: `runs/df-csv-corr-analysis-20260320T014008Z/diff_Com_BrentCrudeOil_correlations.csv`
- raw_bar: `runs/df-csv-corr-analysis-20260320T014008Z/raw_target_vs_rest_bar.png`
- diff_bar: `runs/df-csv-corr-analysis-20260320T014008Z/diff_target_vs_rest_bar.png`
- raw_heatmap: `runs/df-csv-corr-analysis-20260320T014008Z/raw_target_vs_rest_heatmap.png`
- diff_heatmap: `runs/df-csv-corr-analysis-20260320T014008Z/diff_target_vs_rest_heatmap.png`
- summary_json: `runs/df-csv-corr-analysis-20260320T014008Z/summary.json`

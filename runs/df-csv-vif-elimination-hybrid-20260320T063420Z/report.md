# Hybrid Iterative VIF Elimination Summary

- threshold: VIF <= 10.0
- selection tie-break: highest approx VIF, then lower abs(target correlation), then variable name
- final verification: exact per-variable OLS VIF on reduced set

## Com_CrudeOil:raw
- n_obs: 584
- initial_predictors: 89
- removed_count: 58
- final_predictors: 31
- final_max_vif: 8.832229133408669
- final_high_count_gt5: 15
- final_severe_count_gt10: 0
- first_removed: BS_Core_Index_Integrated, BS_Core_Index_A, BS_Core_Index_B, Idx_SnP500, Com_LMEX, Bonds_US_1Y, Bonds_US_2Y, Bonds_AUS_10Y, Idx_CH50, Bonds_CHN_30Y
- last_removed: Com_Wheat, Idx_HangSeng, Com_Coal, EX_USD_CNY, Bonds_KOR_10Y, Com_Coffee, Com_OrangeJuice, Com_Cotton, Com_LME_Ni_Cash, Com_Barley
- kept_sample: Com_LME_Pb_Cash, Com_LME_Ni_Spread, Com_LME_Cu_Spread, Com_LME_Al_Spread, Com_LME_Pb_Spread, Com_LME_Sn_Spread, Com_LME_Zn_Spread, Com_LME_Cu_Inv, Com_LME_Pb_Inv, Com_LME_Sn_Inv

## Com_CrudeOil:diff1
- n_obs: 583
- initial_predictors: 89
- removed_count: 9
- final_predictors: 80
- final_max_vif: 7.159293181826744
- final_high_count_gt5: 2
- final_severe_count_gt10: 0
- first_removed: BS_Core_Index_Integrated, BS_Core_Index_A, BS_Core_Index_B, Idx_SnPGlobal1200, Idx_CSI300, Com_LMEX, Idx_Shanghai50, Com_BloombergCommodity_BCOM, Bonds_US_2Y
- last_removed: BS_Core_Index_Integrated, BS_Core_Index_A, BS_Core_Index_B, Idx_SnPGlobal1200, Idx_CSI300, Com_LMEX, Idx_Shanghai50, Com_BloombergCommodity_BCOM, Bonds_US_2Y
- kept_sample: Com_LME_Ni_Cash, Com_LME_Cu_Cash, Com_LME_Al_Cash, Com_LME_Pb_Cash, Com_LME_Sn_Cash, Com_LME_Zn_Cash, Com_LME_Ni_Spread, Com_LME_Cu_Spread, Com_LME_Al_Spread, Com_LME_Pb_Spread

## Com_BrentCrudeOil:raw
- n_obs: 584
- initial_predictors: 89
- removed_count: 58
- final_predictors: 31
- final_max_vif: 8.832229133408669
- final_high_count_gt5: 15
- final_severe_count_gt10: 0
- first_removed: BS_Core_Index_Integrated, BS_Core_Index_A, BS_Core_Index_B, Idx_SnP500, Com_LMEX, Bonds_US_1Y, Bonds_US_2Y, Bonds_AUS_10Y, Idx_CH50, Bonds_CHN_30Y
- last_removed: Com_Wheat, Idx_HangSeng, Com_Coal, EX_USD_CNY, Bonds_KOR_10Y, Com_Coffee, Com_OrangeJuice, Com_Cotton, Com_LME_Ni_Cash, Com_Barley
- kept_sample: Com_LME_Pb_Cash, Com_LME_Ni_Spread, Com_LME_Cu_Spread, Com_LME_Al_Spread, Com_LME_Pb_Spread, Com_LME_Sn_Spread, Com_LME_Zn_Spread, Com_LME_Cu_Inv, Com_LME_Pb_Inv, Com_LME_Sn_Inv

## Com_BrentCrudeOil:diff1
- n_obs: 583
- initial_predictors: 89
- removed_count: 9
- final_predictors: 80
- final_max_vif: 7.159293181826744
- final_high_count_gt5: 2
- final_severe_count_gt10: 0
- first_removed: BS_Core_Index_Integrated, BS_Core_Index_A, BS_Core_Index_B, Idx_SnPGlobal1200, Idx_CSI300, Com_LMEX, Idx_Shanghai50, Com_BloombergCommodity_BCOM, Bonds_US_2Y
- last_removed: BS_Core_Index_Integrated, BS_Core_Index_A, BS_Core_Index_B, Idx_SnPGlobal1200, Idx_CSI300, Com_LMEX, Idx_Shanghai50, Com_BloombergCommodity_BCOM, Bonds_US_2Y
- kept_sample: Com_LME_Ni_Cash, Com_LME_Cu_Cash, Com_LME_Al_Cash, Com_LME_Pb_Cash, Com_LME_Sn_Cash, Com_LME_Zn_Cash, Com_LME_Ni_Spread, Com_LME_Cu_Spread, Com_LME_Al_Spread, Com_LME_Pb_Spread

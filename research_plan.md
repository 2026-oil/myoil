# 8주 유가 예측 연구 계획

## 목표
- **H1~H5 (급등 이전 구간)**: MAPE ≤ 5%
- **H6~H8 (급등 구간)**: MAPE < 7%
- **고정**: 8주 예측 horizon, NEC 아키텍처 준수, 데이터 누수 절대 금지

## 데이터 현황
- `data/df.csv`: 584주 (2015-01 ~ 2026), 121개 변수
- 타겟: `Com_CrudeOil` (WTI), `Com_BrentCrudeOil` (Brent)
- 변수 카테고리: LME 금속, 환율, 채권 금리/스프레드, 원자재, 주식지수, 변동성(OVX/GVIX), 지정학 리스크(GPRD), BS 지수

## 연구 페이즈

### Phase 1: Feature Engineering + Baseline (즉시 시작)
1. **전체 변수 활용**: 121개 변수 중 상관관계 기반 필터링 + 도메인 지식 기반 선택
2. **Optuna HPO 확장**: `opt_n_trial: 100` → `300+`, 검색공간 확대
3. **Multi-model NEC**: TimeXer, iTransformer, LSTM, TSMixerx, AAForecast, PatchTST, NHiTS, DLinear, GRU
4. **변수 그룹화 실험**:
   - oil_core: OVX, Oil_Spread, Gasoline, Coal, HRC_Steel
   - macro_full: 채권, 환율, 주식지수 전체
   - gprd: GPRD_THREAT, GPRD_ACT, GPRD
   - lme_momentum: LME 금속 + 재고
   - all_vars: 전체 121개 변수

### Phase 2: Advanced Techniques
1. **Custom Loss Functions**:
   - Late-horizon weighted loss (H6-H8 가중치 증가)
   - Quantile loss (불확실성 밴드)
   - Huber-MSE hybrid (이상치 강건)
2. **Model Architecture Variants** (_new suffix):
   - `lstm_new`: Attention-enhanced LSTM, bidirectional encoder
   - `timexer_new`: Multi-scale patch, cross-var attention 강화
   - `itransformer_new`: Channel-independent + global attention
3. **EM / Kalman Filter 전처리**:
   - Kalman Filter 기반 노이즈 제거 + 트렌드 추출
   - EM 기반 GMM 개선 (NEC preprocessing 모드 확장)
   - LPPL (Log-Periodic Power Law) 급등 신호 탐지

### Phase 3: Ensemble + Optimization
1. **Multi-ensemble NEC branches**
2. **Residual stacking**: XGBoost/LightGBM residual 보정
3. **Final MAPE 튜닝**: H6-H8 집중 최적화

## Leakage Prevention 규칙
- expanding-window CV만 사용 (기존 설정 준수)
- futr_exog_cols는 미래에 실제로 알 수 있는 값만
- 모든 파생_FEATURE는 시점 t 기준 과거 정보만 사용
- diff/standardize는 fold별 train 구간 내에서만

## 실험 관리
- 모든 실험은 `yaml/experiment/research_phase*/` 에 YAML로 기록
- `task.name` 으로 실험 식별
- `runs/` 에 산출물 자동 저장
- Optuna `opt_n_trial` 설정으로 재현성 확보

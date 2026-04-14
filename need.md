# need.md

## 목적
Brent AA-Forecast 실험에서, **retrieval 없이**도 마지막 fold 기준
- `h1 >= 78`
- `h2 >= 85`
- `h2 > h1`
- `(h2 - h1) >= 4`
를 만족하도록 만들기 위해, 현재 시점에서 **리서치가 필요한 구조 병목**을 정리한다.

---

## 현재까지 확인된 사실
- 비교 범위는 **non-AA baseline vs AA-GRU vs AA-Informer** 3개만 사용한다.
- 모든 AA 실험은 **`retrieval.enabled=false`** 기준으로 본다.
- best no-retrieval 후보는 현재 대략 다음 수준이다.
  - `stability_dh`: `h1 ≈ 77.54`, `h2 ≈ 82.88`, `gap ≈ 5.34`
- 즉, **우상향과 gap 4 이상은 확보**되기 시작했지만,
  **절대 레벨 uplift가 부족해서 78/85 문턱을 못 넘고 있다.**

---

## 핵심 병목 진단

### 1. 방향성은 어느 정도 잡지만, spike amplitude를 충분히 못 올림
- 최근 no-retrieval 후보들은 `h2 - h1 >= 4`를 만족하는 경우가 생겼다.
- 하지만 `h1`, `h2` 자체가 실제 spike level까지 충분히 못 올라간다.
- 즉 병목은 **방향성(direction)** 보다는 **amplitude transport / level uplift 부족**에 가깝다.

### 2. 작은 파라미터 튜닝의 한계가 뚜렷함
- lowess_frac, hidden size, decoder size, memory gain 계수 등 미세조정은 결과를 약간 흔들 뿐,
  `85`를 넘기는 구조적 돌파로 이어지지 않았다.
- 따라서 앞으로는 **micro tuning보다 구조 실험**이 우선이다.

### 3. STAR 그룹 제약이 구조 탐색을 막고 있음
- 현재 구현에서 STAR hist exog 그룹은 사실상 **hist_exog prefix 제약**을 갖는다.
- 그래서 직관적으로 유효해 보이는 tail 재배치가 실제로는 invalid가 되거나 shape mismatch를 유발했다.
- 즉, **더 많은 exogenous spike 신호를 안전하게 STAR/anomaly 경로에 태우는 설계**가 부족하다.

### 4. spike path / regime path / uncertainty path 중 어떤 경로가 실제 uplift를 담당해야 하는지 불명확함
- 현재 구조는 event/regime/path 관련 분기가 여러 개 있으나,
  어떤 branch가 실제로 `h2` amplitude uplift에 기여하는지가 불분명한 경우가 있었다.
- 일부 branch는 연결은 되어 있어도 영향력이 약하거나, selector에 의해 사실상 묻히는 정황이 있었다.

### 5. uncertainty/selection이 spike를 과도하게 보수적으로 누를 가능성
- no-retrieval에서도 uncertainty selection 또는 path selection이 지나치게 low-dispersion 쪽으로 기울면,
  결과적으로 spike 예측이 눌릴 수 있다.
- 이 부분은 단순 dropout 수치 문제가 아니라,
  **어떤 path를 선택하고 어떤 path를 버리는지의 구조 문제**로 봐야 한다.

### 6. 아키텍처 변경 시 내부 차원 계약(dim contract)이 약함
- 일부 구조 실험에서 `mat1 and mat2 shapes cannot be multiplied`류 shape mismatch가 발생했다.
- 이는 새 구조 실험을 더 공격적으로 하기 전에,
  **decoder / path / regime / memory branch의 차원 계약을 명시적으로 관리할 필요**가 있음을 뜻한다.

---

## 지금 리서치가 필요한 주제

### A. Spike amplitude를 직접 운반하는 구조
리서치 질문:
- event/regime/path 정보를 최종 forecast level uplift로 더 강하게 전달하려면,
  어떤 구조가 retrieval 없이도 효과적인가?
- 현재의 단일 additive path 말고,
  **amplitude head / level-shift head / path transport head**를 어떻게 분리 또는 재결합해야 하는가?

찾아볼 키워드:
- anomaly-aware transformer forecasting
- regime-aware time series transformer
- mixture-of-experts time series forecasting
- event-aware non-stationary forecasting

### B. Rare-event / tail-aware forecasting 구조
리서치 질문:
- 희귀 spike 구간에서 평균회귀를 피하려면,
  classification-style anomaly detector가 아니라 **forecasting head 자체를 tail-aware하게 만드는 방법**은 무엇인가?
- EVT, evidential, rare-event transformer 아이디어 중
  inference hack 없이 구조에 녹일 수 있는 부분은 무엇인가?

찾아볼 키워드:
- tail-aware time series forecasting
- rare-event transformer forecasting
- evidential uncertainty time series forecasting
- extreme value forecasting transformer

### C. STAR grouping 확장 방식
리서치 질문:
- 현재 prefix 제약을 깨지 않으면서도,
  commodity-risk burst 신호(`Com_LMEX`, `Com_BloombergCommodity_BCOM`, `Idx_OVX`, `BS_Core_*`)를
  anomaly-aware path에 더 잘 태울 수 있는 방법은 무엇인가?
- 단순 tail list 추가가 아니라,
  **group routing / multi-group STAR / hierarchical STAR** 같은 설계가 가능한가?

### D. Selector / router 구조
리서치 질문:
- 현재 선택기가 variance-minimizing 쪽으로 너무 기울어져 spike를 누르고 있지는 않은가?
- low-dispersion selector 대신,
  **regime-conditional selector / path router / mixture gate**를 둘 수 있는가?
- 단, horizon bonus 없이 전 구간 공통 구조로 작동해야 한다.

### E. Code-path integrity audit
리서치 질문:
- 현재 model 내부에서 선언된 branch 중
  실제 최종 output에 충분히 연결되지 않거나 영향력이 약한 branch는 없는가?
- path별 gradient/activation/weight mass를 추적해서,
  **실제로 죽어 있는 경로(dead branch)** 를 식별할 수 있는가?

---

## 우선순위

### 1순위
**Amplitude transport 구조**
- spike 정보를 level forecast까지 실질적으로 올려 보내는 구조 연구

### 2순위
**Selector / router 구조**
- spike path가 보수적 선택기에 묻히지 않도록 하는 구조 연구

### 3순위
**STAR grouping 확장**
- prefix 제약을 만족하면서 더 많은 exog spike 신호를 반영하는 구조 연구

### 4순위
**Tail-aware training / uncertainty 구조**
- rare-event 친화적이되 horizon hack이 아닌 구조 연구

---

## 하지 말아야 할 것
- retrieval 다시 켜기
- h2 보너스, continuation bonus, drift injection
- 최근 target 상승분을 이용한 uplift
- lowess_frac, hidden size 같은 작은 값만 계속 바꾸는 실험 반복
- 구조 근거 없이 tail list만 임의로 바꾸는 시도

---

## 다음 액션 가이드
1. 기존 no-retrieval best 후보(`stability_dh` 포함) 기준으로
   **공통 구조 패턴**을 먼저 정리한다.
2. 구조 실험은 반드시 아래 셋 중 하나로 분류한다.
   - amplitude transport
   - selector/router
   - STAR grouping
3. 각 구조 실험은 micro tuning이 아니라,
   **한 가지 명시적 가설**을 갖고 들어간다.
4. 모든 결과는 exp 시트에 남기고,
   keep/discard 기준도 같이 기록한다.

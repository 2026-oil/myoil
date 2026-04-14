# AAForecast Informer 실행 흐름 설명

이 문서는 `yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml` 를 실제로 실행했을 때,
`AAForecast` 의 **Informer 경로가 어떤 순서로 흘러가는지**를
앞서 정리한 STAR/GRU 설명 스타일 그대로 풀어쓴 문서다.

설명 기준은 **이 로컬 checkout 구현**이다.
즉 문체는 우리가 계속 맞춰온 설명 스타일을 따르되,
사실관계는 현재 레포의 `yaml/`, `app_config.py`, `runtime_support/`,
`neuralforecast/models/aaforecast/` 기준으로 잡는다.

---

## 한 줄 정의

이 YAML로 AAForecast Informer를 실행하면 본질적으로

\[
\text{experiment yaml}
\rightarrow
\text{plugin config merge}
\rightarrow
\text{diff window}
\rightarrow
\text{STAR outputs}
\rightarrow
\text{Informer encoder states}
\rightarrow
\text{anomaly-aware sparse attention}
\rightarrow
\text{event-conditioned horizon-aware decoder}
\rightarrow
\text{anchor add-back}
\]

입니다.

즉 Informer 경로의 핵심은,

- **입력 window를 먼저 diff 기준으로 만들고**
- **STAR가 target / 주요 exog의 shock signal을 뽑고**
- **Informer encoder가 시점별 representation을 만들고**
- **critical 시점을 sparse attention으로 다시 강조한 뒤**
- **event_summary / event_trajectory / non_star_regime까지 decoder에 직접 넣어**
- **모델 내부에서는 마지막 diff input anchor를 다시 더하고, 바깥 runtime restore path에서 최종 Brent level forecast로 복원하는 구조**

라는 점입니다.

---

## 0) 이 YAML이 실제로 고정하는 실행 맥락

이번 설명 기준 experiment YAML은
`yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml` 입니다.
여기서 고정되는 것은 우선

- `task.name = brentoil_case1_parity_aaforecast_informer`
- `dataset.path = data/df.csv`
- `target_col = Com_BrentCrudeOil`
- `hist_exog_cols = [GPRD_THREAT, BS_Core_Index_A, GPRD, GPRD_ACT, BS_Core_Index_B, BS_Core_Index_C, Idx_OVX, Com_LMEX, Com_BloombergCommodity_BCOM, Idx_DxyUSD]`
- `aa_forecast.config_path = yaml/plugins/aa_forecast/aa_forecast_parity_informer_stability_dh.yaml`

입니다.

그리고 공통 setting 쪽에서는 실제 실행 컨텍스트가

- `transformations_target = diff`
- `transformations_exog = diff`
- `input_size = 64`
- `horizon = 2`

로 잡혀 있습니다.

즉 이 모델은 **원시 Brent level을 바로 backbone에 넣는 게 아니라**, 먼저

\[
\text{raw training rows}
\rightarrow
\text{diff-transformed target/exog window}
\]

를 만든 뒤 그 위에서 STAR와 Informer를 돌립니다.

plugin YAML까지 합치면 AAForecast Informer 쪽 핵심 파라미터는

- `model = informer`
- `hidden_size = 128`
- `n_head = 4`
- `encoder_layers = 2`
- `dropout = 0.0`
- `linear_hidden_size = 96`
- `factor = 3`
- `decoder_hidden_size = 128`
- `decoder_layers = 4`
- `season_length = 4`
- `lowess_frac = 0.35`
- `lowess_delta = 0.01`
- `thresh = 3.5`
- `uncertainty.enabled = true`
- `retrieval.enabled = false`

입니다.

또 STAR tail rule은

- target: `two_sided`
- `GPRD_THREAT`: `upward`

로 잡혀 있습니다.

즉 충격을 보는 방법도 채널별로 다릅니다.

---

## 1) runtime에서 실제로 어디로 들어가나

이 YAML 실행의 큰 런타임 흐름은 개념적으로

\[
\texttt{main.py}
\rightarrow
\texttt{runtime\_support.runner.main()}
\rightarrow
\texttt{load\_app\_config(...)}
\rightarrow
\texttt{build\_model(...)}
\rightarrow
\texttt{AAForecast(..., backbone=informer)}
\]

입니다.

즉 이 YAML이 `Informer` 모델을 직접 만드는 것이 아니라,
실제로는 **AAForecast wrapper 안에 Informer backbone adapter를 꽂은 모델**을 만듭니다.

여기서 중요한 건 두 가지입니다.

### 1-1. experiment YAML과 plugin YAML은 분리돼 있다

experiment YAML은 데이터셋/타깃/작업 이름을 주고,
plugin YAML은 실제 AAForecast backbone/STAR/uncertainty/retrieval 설정을 줍니다.

즉 런타임에서는

\[
\text{experiment config} + \text{AA plugin config}
\rightarrow
\text{resolved AAForecast config}
\]

로 합쳐진 뒤 모델을 생성합니다.

### 1-2. 이번 실행에서는 retrieval이 꺼져 있다

plugin config에 `retrieval.enabled = false` 이므로,
이번 설명에서는 “과거 event memory를 외부 retrieval 결과와 blend하는 흐름”은 들어가지 않습니다.

다만 Informer 내부에는 여전히 **pooled context / memory token / memory bank를 계산하는 path**가 있고,
이건 retrieval이 아니라 **현재 window 내부 representation을 다시 집약해서 decoder에 넘기는 내부 memory path**입니다.

---

## 2) raw data에서 model input window까지

이제 실제 학습/추론 시점을 보겠습니다.

모델에 바로 전체 CSV가 들어가는 게 아니라,
런타임은 우선 `data/df.csv` 에서 `dt` 기준으로 정렬된 Brent 시계열과 hist exog를 읽고,
각 fold/cutoff마다 학습 구간을 자릅니다.

그 다음 target / hist exog에 대해 둘 다 `diff` 변환을 적용합니다.

즉 현재 모델이 보는 마지막 encoder window는

\[
Y^{diff}_{t-63:t},\quad X^{diff}_{t-63:t}
\]

입니다.

여기서

- target은 Brent 차분값
- exog도 모두 차분값

입니다.

이 단계의 직관은 단순합니다.
AAForecast Informer는 level 자체보다 **최근 변화량과 shock 구조**를 더 직접적으로 보도록 세팅돼 있습니다.

---

## 3) STAR 이후, encoder에 실제로 무엇이 들어가나

GRU 설명 때와 똑같이,
Informer도 raw target 하나만 바로 encoder에 넣는 게 아닙니다.

코드상 `feature_size` 는

\[
(0 \text{ or } 1)
+
\#(\text{non-star hist exog})
+
4
+
4 \times \#(\text{star hist exog})
\]

로 잡힙니다.

이번 YAML에서는

- 원 target 1개
- non-STAR hist exog 9개
- target STAR 4개 채널
- STAR hist exog (`GPRD_THREAT`)의 4개 채널

이므로,

\[
1 + 9 + 4 + 4 = 18
\]

개 feature가 encoder input으로 들어갑니다.

즉 시점별 encoder input은 대략

\[
E_t =
[
 y_t,
 x^{(nonstar)}_t,
 T^{(y)}_t,
 S^{(y)}_t,
 A^{(y)}_t,
 R^{(y)}_t,
 T^{(gprd\_threat)}_t,
 S^{(gprd\_threat)}_t,
 A^{(gprd\_threat)}_t,
 R^{(gprd\_threat)}_t
]
\]

형태입니다.

즉 backbone은 “차분된 Brent 원값만” 보는 것이 아니라,
이미 STAR가 분해한

- trend
- seasonal
- anomaly-only channel
- cleaned residual

을 같이 받습니다.

---

## 4) STAR는 여기서 어떤 역할을 하나

이 문서의 중심은 Informer path지만,
그 전에 STAR가 만드는 표현을 짚어야 뒤가 이해됩니다.

각 대상 채널에 대해 STAR는 대략

\[
x_t
\rightarrow
T_t\,(\text{LOWESS})
\rightarrow
x_t/T_t
\rightarrow
S_t\,(\text{phase mean})
\rightarrow
R_t = x_t/(T_t S_t)
\rightarrow
\text{robust score}
\rightarrow
\text{critical mask / activity / ranking}
\]

를 만듭니다.

이번 YAML에서는 특별히

- target Brent는 `two_sided`
- `GPRD_THREAT` 는 `upward`

이므로,
충격 마스크와 ranking score가 채널별로 다르게 계산됩니다.

그리고 STAR는 단순 decomposition으로 끝나지 않고,
뒤 단계에 필요한 아래 payload를 만듭니다.

- `target_trend`, `target_seasonal`, `target_anomalies`, `target_residual`
- `star_hist_trend`, `star_hist_seasonal`, `star_hist_anomalies`, `star_hist_residual`
- `critical_mask`
- `count_active_channels`
- `channel_activity`
- `event_summary`
- `event_trajectory`
- `non_star_regime`
- `regime_intensity`
- `regime_density`

즉 Informer 경로에서 STAR는 “앞단 decomposition”에 그치지 않고,
**attention과 decoder가 읽을 event representation 전체를 공급하는 단계**입니다.

---

## 5) Informer encoder

이제 STAR output을 합친 encoder input이 backbone으로 들어갑니다.
Informer backbone은 `InformerBackboneAdapter` 를 통해 구성됩니다.

여기서 중요한 구조적 특징은 다음과 같습니다.

### 5-1. 첫 번째 채널만 Informer의 main signal로 들어간다

adapter는 입력을

- 첫 채널: `signal = inputs[..., :1]`
- 나머지 채널: `exog = inputs[..., 1:]`

로 나눕니다.

즉 Informer 쪽 `c_in` 은 1이고,
AAForecast가 만든 나머지 feature들은 **Informer exogenous mark** 로 들어갑니다.

이건 꽤 중요합니다.
GRU처럼 모든 feature를 같은 방식으로 recurrent encoder에 통째로 밀어넣는 게 아니라,
Informer는

- 기준 시그널 1개
- 나머지 AA feature 보조 정보

라는 구조를 유지합니다.

### 5-2. encoder-only Informer를 쓴다

이 경로는 full seq2seq Informer decoder를 그대로 쓰지 않고,
`InformerEncoderOnly` 를 사용해 **encoder states만 뽑습니다.**

즉 수식으로는

\[
E \in \mathbb{R}^{B \times 64 \times 18}
\rightarrow
\text{InformerEncoderOnly}
\rightarrow
H \in \mathbb{R}^{B \times 64 \times 128}
\]

입니다.

여기서

- `B`: batch
- `64`: input_size
- `18`: encoder input feature 수
- `128`: hidden_size

입니다.

### 5-3. distillation은 꺼져 있다

adapter는 Informer encoder를 만들 때 `distil=False` 로 둡니다.
즉 encoder 중간 convolution bottleneck으로 time dimension을 줄이지 않습니다.

이유는 명확합니다.
AAForecast는 뒤에서 **critical time step 기준 sparse attention** 을 해야 하므로,
시점 정렬이 살아 있어야 합니다.

즉 여기서 핵심은

\[
\text{AA feature window}
\rightarrow
\text{time-aligned Informer hidden states}
\]

를 얻는 것입니다.

---

## 6) anomaly-aware sparse attention

Informer backbone이 hidden states를 만들고 나면,
그 다음 aa-model의 핵심인 `CriticalSparseAttention` 이 들어갑니다.

입력은 다음 네 개입니다.

- `hidden_states`
- `critical_mask`
- `count_active_channels`
- `channel_activity`

여기까지만 보면 GRU 경로와 상당히 비슷합니다.
그런데 Informer는 한 가지 차이가 더 있습니다.

### 6-1. regime time context를 먼저 hidden에 더한다

Informer 경로에서는 attention 직전에

- `regime_intensity`
- `regime_density`

를 받아 `regime_time_projector` 로 projection한 뒤,
그 결과를 hidden state에 더합니다.

즉 attention에 실제로 들어가는 것은 단순한 encoder hidden `H` 가 아니라,

\[
H^{reg}_t = H_t + r_t
\]

형태의 hidden입니다.

여기서 `r_t` 는 그 시점의 non-STAR exog regime intensity / density를 반영한 보정값입니다.

즉 Informer 경로는 GRU보다 한 단계 더 나아가,
**시장/정책 exog의 regime 강도까지 hidden 시퀀스에 미리 주입한 뒤**
critical sparse attention을 수행합니다.

### 6-2. sparse attention이 하는 일 자체는 동일한 철학이다

attention은 전체 64 step을 똑같이 보지 않고,

- critical_mask가 켜진 시점
- 동시에 active한 채널 수가 많은 시점
- channel_activity가 큰 시점

에 더 무게를 둡니다.

즉 결과 attended state를

\[
\tilde{H}
\in
\mathbb{R}^{B \times 64 \times 128}
\]

라고 하면,
이건 “Informer가 본 원래 시계열 representation”을
**shock-aware하게 다시 강조한 버전**입니다.

여기서 중요한 해석은,
AAForecast Informer도 GRU와 마찬가지로
“전체 sequence를 전부 새로 쓰는 모델”이 아니라,
**충격 시점 위주로 representation을 재강조하는 모델**이라는 점입니다.

---

## 7) event summary / event trajectory / non-star regime

여기서부터가 Informer 경로가 GRU와 크게 갈라지는 지점입니다.

GRU 경로는 event 정보를 주로 attention 단계에서 강하게 쓰고,
마지막 decoder는 원/attended hidden의 concat을 MLP로 받습니다.

반면 Informer 경로는 그보다 더 깊게 들어갑니다.
STAR payload로부터 만들어진

- `event_summary`
- `event_trajectory`
- `non_star_regime`

를 **decoder conditioning 신호로 직접** 넣습니다.

### 7-1. event_summary

이건 window 전체에서 shock가 어떤 성격이었는지를 요약한 벡터입니다.
예를 들면

- critical density
- 최근 density
- 평균 active channel 수
- 최근 activity mass
- target up mass
- star hist up mass
- non-star activity 관련 요약
- non_star_regime descriptor

같은 값들이 들어갑니다.

즉 한 문장으로 말하면,

**"이번 64-step window 전체에서 shock가 얼마나 잦고, 얼마나 강했고, 어떤 방향이었는가"**

를 요약한 벡터입니다.

### 7-2. event_trajectory

이건 summary보다 더 시간축적인 요약입니다.
예를 들면

- 최근 up mass vs 이전 up mass 차이
- 마지막 시점 shock intensity
- target과 hist shock의 gap
- persistence
- non-star recent/earlier mass shift

같은 값이 들어갑니다.

즉 한 문장으로는,

**"충격이 최근으로 갈수록 커졌는가, 마지막에 살아 있는가, 지속되고 있는가"**

를 보는 경로입니다.

### 7-3. non_star_regime

나머지 hist exog 9개는 STAR main path로 직접 들어가진 않지만,
시장/정책 그룹 descriptor를 만들어 decoder conditioning에 같이 씁니다.

즉 Informer 경로는 단순히 target shock만 보는 게 아니라,
**Brent 외부의 market/policy regime도 요약된 벡터로 함께 본다**는 뜻입니다.

---

## 8) horizon 정렬

이제 decoder 직전까지 오면 두 종류의 time states가 있습니다.

- 원래 hidden states
- attended hidden states

즉

\[
H^{reg} \in \mathbb{R}^{B \times 64 \times 128}
\]
\[
\tilde{H} \in \mathbb{R}^{B \times 64 \times 128}
\]

입니다.

이걸 forecasting horizon `h=2` 에 맞춰 정렬합니다.

현재는 `h <= input_size` 이므로,
그냥 마지막 `h`개 시점만 잘라서 씁니다.

\[
H^{align} = H^{reg}[:, -2:, :]
\]
\[
\tilde{H}^{align} = \tilde{H}[:, -2:, :]
\]

즉 shape은

\[
H^{align}, \tilde{H}^{align}
\in
\mathbb{R}^{B \times 2 \times 128}
\]

입니다.

Informer 경로에서는 여기에 더해,
`regime_time_latent` 도 같은 방식으로 horizon 길이에 맞춰 정렬합니다.

\[
R^{align} \in \mathbb{R}^{B \times 2 \times 128}
\]

---

## 9) decoder 직전 feature는 어떻게 만들어지나

Informer 전용 decoder path는 `_decode_informer_forecast(...)` 에서 구성됩니다.

### 9-1. event summary와 event trajectory를 latent로 project한다

우선

\[
\text{event\_context} = P_s(\text{event\_summary})
\in \mathbb{R}^{B \times 128}
\]
\[
\text{event\_path} = P_t(\text{event\_trajectory})
\in \mathbb{R}^{B \times 128}
\]

를 만듭니다.

즉 raw summary/path vector를 decoder가 읽기 쉬운 hidden 차원 latent로 바꿉니다.

### 9-2. pooled context / memory token / memory bank를 만든다

Informer 경로는 여기서 한 번 더 내부 memory pooling을 합니다.

query는 대략

\[
q = f([\text{event\_context}; \text{event\_path}; \text{non\_star\_regime}])
\]

이고,
keys / values는 hidden state들에서 뽑습니다.

즉 decoder는 “이번 window의 event summary/path가 중요하다고 말하는 hidden 조각들”을
다시 모아서

- `pooled_context`
- `memory_token`
- `memory_bank`

를 만듭니다.

이건 retrieval과 다릅니다.
외부 과거 window를 불러오는 게 아니라,
**현재 window 내부 representation을 한 번 더 event-aware하게 pool하는 단계**입니다.

### 9-3. 최종 decoder input

시간축 decoder input은

\[
Z = [H^{align} + R^{align}; \tilde{H}^{align} + R^{align}]
\in \mathbb{R}^{B \times 2 \times 256}
\]

입니다.

즉 각 horizon step은

- 원래 informer hidden
- anomaly-aware attended hidden
- regime time context

가 섞인 representation을 받습니다.

그리고 이건 GRU와의 핵심 차이 하나를 드러냅니다.

GRU는 최종 decoder 입력이 거의

\[
[H^{align}; \tilde{H}^{align}]
\]

중심이라면,
Informer는 여기에 더해

- `event_context`
- `event_path`
- `non_star_regime`
- `pooled_context`
- `memory_token/bank`
- `anchor_level`

까지 함께 넣어서 **경로 조건부(path-aware) decoding** 을 합니다.

---

## 10) decoder

Informer 경로는 shared decoder MLP를 쓰지 않고,
`InformerHorizonAwareHead` 를 사용합니다.

즉 전체 흐름은

\[
Z
\rightarrow
\text{InformerHorizonAwareHead}
(Z, \text{event\_context}, \text{event\_path}, \text{non\_star\_regime}, \text{pooled\_context}, \text{memory}, \text{anchor})
\rightarrow
\Delta \hat{Y}
\]

입니다.

여기서 핵심은,
이 head가 단순히 “각 horizon step에 같은 MLP를 태우는 구조”가 아니라,
**이번 window의 event 요약과 trajectory, regime, pooled memory에 따라 horizon별 출력을 다르게 조정하는 구조**라는 점입니다.

즉 1-step forecast와 2-step forecast가 같은 정보만 받는 것이 아니라,
같은 context를 horizon-aware하게 다른 방식으로 읽습니다.

이 부분이 바로 Informer 경로가 GRU보다 더 event-conditioned 하다고 말하는 이유입니다.

---

## 11) anchor add-back: 모델 내부 diff anchor와 바깥 level 복원

현재 runtime에서 target은 `diff` 로 변환되어 있습니다.
따라서 decoder가 직접 내놓는 것은 Brent level 그 자체가 아니라,
우선 **diff-space delta forecast** 입니다.

코드상 Informer path 내부 마지막은

\[
\hat{Y}^{model,diff} = y^{diff}_{anchor} + \Delta \hat{Y}
\]

입니다.

여기서 `y^{diff}_{anchor}` 는 마지막 insample **diff-scale target 값** 입니다.
즉 모델 내부에서는 raw level anchor가 아니라,
현재 입력 window의 마지막 diff 값을 한 번 다시 더해줍니다.

그리고 그 다음 최종 level 복원은 model 바깥 runtime restore path에서 일어납니다.
runner 쪽 복원식은 개념적으로

\[
\hat{Y}^{level} = \mathrm{cumsum}\left(\hat{Y}^{model,diff}\right) + y^{raw}_{anchor}
\]

입니다.

즉 이번 YAML의 prediction을 정확히 해석하면,

- 모델 내부는 diff-transformed window를 보고
- shock-aware delta를 예측한 뒤
- 마지막 diff input을 한 번 다시 더해 diff-space output을 만들고
- 바깥 runtime이 마지막 raw Brent anchor를 기준으로 cumulative restore를 해서

최종 level forecast를 만듭니다.

---

## 12) stochastic-dropout uncertainty

이번 plugin YAML에서는 uncertainty가 켜져 있습니다.

- `enabled = true`
- `dropout_candidates = [0.005, 0.01, ..., 0.3]`
- `sample_count = 30`

입니다.

모델 내부의 `_apply_stochastic_dropout(...)` 는

\[
\text{training} \quad \text{or} \quad \text{stochastic\_inference\_enabled}
\]

이면 dropout을 적용할 수 있게 만들어져 있습니다.

Informer 경로에서는 이 stochasticity가

- `hidden_aligned`
- `attended_aligned`
- `event_context`
- `event_path`
- `pooled_context`
- `decoder_input`

같은 representation 지점들에 걸쳐 들어갈 수 있습니다.

즉 uncertainty는 “맨 마지막 output 한 번 흔드는 것”이 아니라,
**event-conditioned decoder path 전체에 stochasticity를 주입할 수 있는 구조**입니다.

다만 이번 YAML은 `retrieval.enabled=false` 이므로,
uncertainty가 retrieval gate와 결합되는 경로는 이번 실행 설명에서 제외됩니다.

---

## 13) 그래서 이 YAML 실행 흐름을 진짜 압축하면

### 13-1. config resolve

\[
\text{experiment yaml}
\rightarrow
\text{AA plugin yaml merge}
\rightarrow
\text{AAForecast(backbone=informer)}
\]

### 13-2. data window

\[
\text{raw Brent/exog rows}
\rightarrow
\text{diff transform}
\rightarrow
\text{last 64-step window}
\]

### 13-3. STAR

\[
\text{target},\; GPRD\_THREAT
\rightarrow
\text{trend/seasonal/anomaly/residual}
\rightarrow
\text{critical mask / activity / event summary / trajectory}
\]

### 13-4. encoder input

\[
E \in \mathbb{R}^{B \times 64 \times 18}
\]

### 13-5. Informer encoder

\[
E \rightarrow H \in \mathbb{R}^{B \times 64 \times 128}
\]

### 13-6. anomaly-aware attention

\[
H + \text{regime time context}
\rightarrow
\tilde{H} \in \mathbb{R}^{B \times 64 \times 128}
\]

### 13-7. decoder conditioning

\[
(\text{event summary}, \text{event trajectory}, \text{non-star regime})
\rightarrow
(\text{event context}, \text{event path}, \text{pooled context})
\]

### 13-8. horizon-aware decode

\[
[H^{align}; \tilde{H}^{align}] + \text{regime context}
\rightarrow
\text{InformerHorizonAwareHead}
\rightarrow
\Delta \hat{Y}
\]

### 13-9. model output + outer restore

\[
\hat{Y}^{model,diff} = y^{diff}_{anchor} + \Delta \hat{Y}
\]
\[
\hat{Y}^{level} = \mathrm{cumsum}\left(\hat{Y}^{model,diff}\right) + y^{raw}_{anchor}
\]

즉 한 문장으로 다시 말하면,

**이 YAML의 AAForecast Informer는 diff 기준 Brent window에서 STAR로 shock/event signal을 먼저 뽑고, Informer encoder hidden을 critical time 중심으로 다시 강조한 뒤, event summary/path/regime까지 decoder에 직접 조건으로 넣어서 horizon별 delta forecast를 만들고, 모델 내부에서는 마지막 diff anchor를 다시 더한 후 바깥 runtime restore path에서 최종 Brent level 예측으로 복원하는 구조**입니다.

---

## 14) GRU 경로와 비교하면 어디가 제일 다르나

이건 꼭 짚고 가야 합니다.

### 14-1. 같은 점

GRU와 Informer 둘 다

- STAR 분해를 먼저 하고
- `critical_mask`, `count_active_channels`, `channel_activity` 를 써서
- anomaly-aware sparse attention을 수행합니다.

즉 **충격 시점을 더 강조한다**는 철학은 같습니다.

### 14-2. 다른 점

하지만 Informer는 GRU보다 한 단계 더 들어갑니다.

GRU 경로는 주로

\[
[\text{raw hidden}; \text{attended hidden}]
\rightarrow
\text{shared decoder MLP}
\]

중심입니다.

반면 Informer는

\[
[\text{raw hidden}; \text{attended hidden}] 
+ \text{regime time context}
+ \text{event summary latent}
+ \text{event trajectory latent}
+ \text{non-star regime}
+ \text{pooled memory context}
\]

까지 decoder에 직접 조건으로 넣습니다.

즉 GRU가 “attention 중심 shock emphasis” 쪽이라면,
Informer는 **attention + decoder conditioning 둘 다 event-aware** 한 구조입니다.

---

## 15) Brent 맥락에서 이 구조를 왜 보나

우리가 Brent spike를 잡으려고 이 경로를 보는 이유는 분명합니다.

plain Informer만 쓰면 최근 sequence representation을 잘 만드는 데는 강하지만,
그 representation이 반드시 “충격 구간”을 특별히 강조한다고 보장되지는 않습니다.

반면 AAForecast Informer는 그 전에 이미

- 어떤 시점이 critical했는지
- 몇 개 채널이 동시에 반응했는지
- `GPRD_THREAT` 같은 핵심 shock exog가 위로 튀었는지
- 나머지 market/policy exog의 regime가 최근에 강해졌는지
- shock가 최근으로 갈수록 커졌는지 / 살아 있는지

를 따로 요약해서 attention과 decoder에 넣습니다.

즉 구조적으로는 plain Informer보다
**shock-aware / event-aware / regime-aware forecast** 를 만들 여지가 더 큽니다.

특히 이번 YAML은 retrieval을 끈 상태이므로,
“과거 유사 event를 끌어오는 효과”가 아니라,
**현재 64-step window 안에서 shock 표현을 얼마나 잘 추출하고 decoder에 연결하느냐**가 핵심입니다.

---

## 16) 지금 문서에서 꼭 기억할 구현 디테일

1. 이번 실행은 `aaforecast-informer.yaml` + `aa_forecast_parity_informer_stability_dh.yaml` 조합이다.
2. target/exog 모두 먼저 `diff` 변환을 거친다.
3. `input_size=64`, `horizon=2` 다.
4. STAR upward exog는 `GPRD_THREAT` 하나고, 나머지 9개는 non-STAR regime 쪽으로 간다.
5. encoder input feature 수는 이번 설정에서 18이다.
6. Informer adapter는 첫 채널만 main signal로 쓰고, 나머지는 exogenous mark로 보낸다.
7. Informer encoder는 time alignment 보존을 위해 `distil=False` 다.
8. attention 직전에 `regime_intensity / regime_density` 가 hidden에 더해진다.
9. Informer 경로는 `event_summary`, `event_trajectory`, `non_star_regime` 를 decoder에 직접 넣는다.
10. 모델 내부 최종 출력은 diff anchor가 더해진 diff-space forecast이고, 바깥 runtime restore path가 raw anchor 기준 level forecast로 복원한다.
11. uncertainty는 켜져 있지만 retrieval은 꺼져 있다.
12. 즉 이번 경로의 본질은 **retrieval 없는 event-conditioned Informer forecasting** 이다.

---

## 마지막 한 줄 압축

이번 YAML의 AAForecast Informer는,

**"diff 기준 Brent window에서 STAR로 shock/event 구조를 먼저 뽑고, Informer hidden을 critical 시점 중심으로 다시 강조한 다음, event summary/trajectory/regime까지 horizon-aware decoder에 직접 넣어 delta forecast를 만든 뒤, 모델 내부에서는 마지막 diff anchor를 더하고 바깥 runtime이 raw Brent anchor 기준으로 level forecast를 복원하는 모델"**

입니다.

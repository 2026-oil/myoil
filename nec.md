# NEC Plugin Operator Guide

이 문서는 **현재 코드 기준**으로 NEC 플러그인이 실제로 어떻게 동작하는지만 설명한다.

다루는 범위:
- 현재 설정 구조
- validate-only / 일반 실행 흐름
- 입력 데이터와 branch별 동작
- metadata / artifact 위치

다루지 않는 범위:
- 과거 방식과의 비교
- 변경 이력
- 향후 계획

---

## 1. NEC가 현재 무엇인가

현재 NEC는 이 리포지토리에서 **plugin-owned top-level job** 이다.

즉:
- 메인 job은 `jobs[*].model: NEC` 로 선언되고
- 실제 fold 실행은 일반 learned-model 공용 경로가 아니라
- `plugins/nec/plugin.py` / `plugins/nec/runtime.py` 가 직접 소유한다.

실행 축은 크게 두 개다.

1. **설정 / 라우팅 축**
   - `plugin_contracts/stage_registry.py`
   - `plugins/nec/config.py`
   - `plugins/nec/plugin.py`
   - `yaml/plugins/nec.yaml`
   - `yaml/experiment/feature_set_nec/*.yaml`

2. **실행 / 산출물 축**
   - `plugins/nec/runtime.py`

운영자 관점에서는 다음처럼 이해하면 된다.

- 메인 YAML에서 `nec.enabled: true`
- `nec.config_path` 가 linked NEC YAML을 가리킴
- linked NEC YAML이 branch 설정을 확정함
- `jobs[*].model == NEC` 인 job은 NEC plugin runtime으로 들어감
- 각 fold에서 `normal`, `extreme`, `classifier` 세 branch를 실행함
- classifier branch의 gate 결과로 normal/extreme 예측을 합쳐 최종 `NEC` 예측을 만듦

---

## 2. 관련 파일 역할

### 2.1 `plugin_contracts/stage_registry.py`
- NEC plugin을 registry에 등록한다.
- config 로딩 시 `plugins.nec.plugin` 이 lazy-load 된다.

의미:
- runtime이 NEC를 직접 하드코딩하지 않고 stage plugin으로 찾는다.

### 2.2 `plugins/nec/config.py`
- NEC thin main config(`enabled`, `config_path`)를 파싱한다.
- linked NEC YAML의 실제 스키마를 검증 / 정규화한다.
- branch 설정을 dataclass로 만든다.
- stage-loaded normalized payload를 만든다.

### 2.3 `plugins/nec/plugin.py`
- StagePlugin 구현체다.
- linked NEC YAML route를 검증한다.
- branch model compatibility를 사전 검증한다.
- top-level `NEC` job을 자신이 소유한다고 선언한다.
- normalized payload / manifest / validation payload에 NEC metadata를 넣는다.
- fold prediction을 `plugins/nec/runtime.py` 로 위임한다.

### 2.4 `plugins/nec/runtime.py`
- NEC 실제 실행의 중심이다.
- branch별 입력 feature matrix를 만든다.
- branch별로 선택된 neuralforecast 모델을 실행한다.
- classifier gate로 normal/extreme를 합친다.
- `run_root/nec/*` 와 `summary/nec/*` artifact를 쓴다.

### 2.5 `yaml/plugins/nec.yaml`
- 운영자가 직접 조정하는 linked NEC YAML이다.
- branch별 model / model_params / variables 를 정의한다.

### 2.6 `yaml/experiment/feature_set_nec/*.yaml`
- NEC 실행 case family다.
- dataset / cv / scheduler / shared training 설정은 메인 실험 YAML에서 잡고
- NEC 세부 branch 설정은 linked NEC YAML로 넘긴다.

---

## 3. 현재 설정 구조

NEC 설정은 두 층이다.

### 3.1 메인 실험 YAML의 NEC 블록
예:

```yaml
nec:
  enabled: true
  config_path: yaml/plugins/nec.yaml
```

메인 YAML의 역할은 얇다.

현재 메인 YAML에서 NEC가 직접 가지는 의미는:
- NEC를 켤지 (`enabled`)
- 어떤 linked NEC YAML을 읽을지 (`config_path`)

### 3.2 linked NEC YAML의 실제 구조
현재 linked NEC YAML은 대략 다음 형태다.

```yaml
nec:
  preprocessing:
    mode: diff_std
    gmm_components: 3
    epsilon: 1.5
  inference:
    mode: soft_weighted
    threshold: 0.5
  classifier:
    model: LSTM
    variables: []
    model_params: {...}
  normal:
    model: LSTM
    variables:
      - Com_Gasoline
      - Com_Steel
    model_params: {...}
  extreme:
    model: LSTM
    variables:
      - Com_Gasoline
      - Com_Steel
      - Bonds_US_Spread_10Y_1Y
    model_params: {...}
  validation:
    windows: 8
```

---

## 4. linked NEC YAML의 각 키 의미

### 4.1 `preprocessing`
현재 허용 키:
- `mode`
- `gmm_components`
- `epsilon`

현재 의미:
- `mode`
  - 현재는 `diff_std` 만 허용
- `gmm_components`
  - target diff-normalized 값에 대해 GMM을 만들 때 component 수
- `epsilon`
  - extreme 판정 기준

중요:
- `preprocessing.probability_feature` 는 **더 이상 설정 키가 아니다**
- 현재 구현에서는 probability feature가 **항상 켜진다**

### 4.2 `inference`
허용 키:
- `mode`
- `threshold`

현재 의미:
- `mode: soft_weighted`
  - classifier probability를 weight로 써서 normal/extreme를 혼합
- `mode: hard_threshold`
  - classifier probability를 threshold로 잘라 normal/extreme 중 하나를 선택
- `threshold`
  - `hard_threshold` 에서 gate 기준값

### 4.3 `classifier`
필수 키:
- `model`
- `variables`
- `model_params`

현재 의미:
- `model`
  - classifier branch에서 쓸 neuralforecast model 이름
- `variables`
  - classifier branch가 target 외에 함께 쓸 추가 변수 목록
- `model_params`
  - 선택한 모델의 생성 파라미터

동작 규칙:
- `variables: []` 이면 target만 쓰는 단변량 branch처럼 동작
- classifier는 내부적으로 **binary target adapter** 로 처리된다
- classifier 출력은 `[0, 1]` 로 clip된 gate probability로 사용된다

### 4.4 `normal`
필수 키:
- `model`
- `variables`
- `model_params`

의미:
- normal branch 예측 backbone 선택
- branch 전용 추가 변수 선택
- 모델 생성 파라미터 지정

### 4.5 `extreme`
필수 키:
- `model`
- `variables`
- `model_params`

의미:
- extreme branch 예측 backbone 선택
- branch 전용 추가 변수 선택
- 모델 생성 파라미터 지정

현재 구현에서는 extreme branch 입력에 probability feature도 포함된다.

### 4.6 `validation`
허용 키:
- `windows`

의미:
- NEC branch 내부 validation 관련 윈도우 수 metadata

---

## 5. 현재 더 이상 받지 않는 NEC 설정

linked NEC YAML에서 현재 제거된 키:
- `history_steps`
- `hist_columns`
- `preprocessing.probability_feature`

현재 구현에서는 이 키들이 들어오면 fail-fast 한다.

운영 의미:
- history length는 NEC 전용 설정이 아니라 **항상 `training.input_size`** 를 사용
- branch별 변수는 공통 `hist_columns` 가 아니라 **각 branch의 `variables`** 가 직접 소유
- probability feature on/off는 operator choice가 아니라 **항상 on**

---

## 6. branch model 선택 규칙

### 6.1 어떤 모델을 쓸 수 있나
현재 NEC branch model 이름은:
- `neuralforecast.models.__all__` 에 export 되어 있고
- 동시에 shared runtime catalog(`runtime_support/forecast_models`)에서 build 가능한 모델
중에서 고른다.

즉, operator는 linked NEC YAML에서 branch별 model 이름을 직접 지정할 수 있다.

### 6.2 아무 export 모델이나 다 허용되는가
아니다.

현재 구현은 **export surface 기반 + compatibility gate** 방식이다.

즉:
- export 되어 있어도
- NEC branch 입력 계약과 안 맞으면
- validate 단계에서 바로 reject 된다.

### 6.3 현재 compatibility에서 보는 축
branch model 검증 시 현재 보는 축:
- multivariate 여부
- hist exog 지원 여부
- futr exog 지원 여부
- stat exog 지원 여부
- model constructor가 받지 않는 unknown `model_params` 존재 여부

현재 핵심 규칙:
- NEC branch는 multivariate model을 허용하지 않음
- branch가 hist exog 입력이 필요한데 모델이 hist exog를 지원하지 않으면 reject
- unknown `model_params` 는 runtime에서 조용히 버리지 않고 validate 단계에서 reject

### 6.4 classifier branch 추가 규칙
classifier branch는 현재 별도 규칙이 있다.

- `classifier.model` 은 exported model 중에서도
- NEC classifier adapter와 맞는 경우만 허용된다
- classifier는 binary target을 학습하고
- 결과를 gate probability로 사용한다

운영 관점에서는:
- classifier도 model 이름은 operator가 고르지만
- 아무 model이나 다 되는 것이 아니라
- NEC classifier contract를 통과해야 한다고 보면 된다.

---

## 7. branch 변수와 active_hist_columns

### 7.1 branch 변수
각 branch는 자기 `variables` 를 직접 가진다.

예:
- `classifier.variables`
- `normal.variables`
- `extreme.variables`

현재 구현 규칙:
- 변수는 dataset column 이어야 함
- `dataset.target_col` 을 branch variables 안에 직접 넣으면 reject
- `variables: []` 는 그 branch에서 target-only 입력을 의미

### 7.2 `active_hist_columns`
현재 `active_hist_columns` 는 operator 입력이 아니라 **derived metadata** 다.

의미:
- `normal.variables`, `extreme.variables`, `classifier.variables` 에 들어간
- target이 아닌 변수들의 합집합

이 값은 metadata / resolved payload / summary 쪽에서 노출된다.

즉 현재 운영 의미는:
- NEC가 실제로 참조하는 보조 변수 전체 집합을 한 눈에 보여주는 메타데이터
- 설정 키가 아니라 결과물 메타데이터

---

## 8. main.py 실행 시 실제 흐름

`main.py --config <nec-case.yaml>` 실행 시 현재 큰 흐름은 다음과 같다.

1. 메인 실험 YAML 로드
2. shared settings 병합
3. stage plugin registry에서 NEC 감지
4. linked NEC YAML 로드
5. linked NEC YAML 스키마 검증
6. branch model compatibility 검증
7. stage-loaded NEC config를 AppConfig에 반영
8. resolved / manifest / capability metadata 생성
9. validate-only 이면 stage metadata만 기록하고 종료
10. 일반 실행이면 fold별 NEC runtime 진입
11. 각 fold에서 `normal`, `extreme`, `classifier` branch 실행
12. classifier gate로 final `NEC` 예측 생성
13. NEC plugin artifact + `summary/nec` artifact 기록

---

## 9. validate-only에서 실제로 확인되는 것

`main.py --validate-only --config <nec-case>` 실행 시 현재 확인되는 것은:
- NEC thin main config 인식 여부
- linked NEC YAML 스키마 정상 여부
- branch model compatibility 정상 여부
- branch variable column 참조 정상 여부
- stage-loaded NEC normalized payload 생성 여부
- resolved / manifest / capability report 에 NEC metadata가 반영되는지

현재 validate-only에서 중요한 NEC metadata 예시는 다음과 같다.

- `history_steps_source: training.input_size`
- `history_steps_value: <training.input_size>`
- `probability_feature_forced: true`
- `active_hist_columns`
- `branches.<branch>.model`
- `branches.<branch>.variables`
- `branches.<branch>.compatible`
- `inference_mode`
- `inference_threshold`
- `gmm_components`
- `epsilon`
- `summary_nec_root`

---

## 10. 일반 실행 시 branch별 실제 입력

현재 runtime은 각 branch마다 별도 feature matrix를 만든다.

### 10.1 공통 기초
공통으로 먼저 수행하는 것:
- target series를 diff + standardize
- branch variables도 각 컬럼별로 diff + standardize
- target diff-normalized 값으로 GMM 기반 probability feature 생성
- extreme flag 생성

### 10.2 `normal` branch 입력
현재 `normal` branch 입력은:
- target diff-normalized 값
- `normal.variables` 에 있는 변수들

즉 probability feature는 normal에 직접 추가되지 않는다.

### 10.3 `extreme` branch 입력
현재 `extreme` branch 입력은:
- target diff-normalized 값
- `extreme.variables` 에 있는 변수들
- probability feature

### 10.4 `classifier` branch 입력
현재 `classifier` branch 입력은:
- target diff-normalized 값
- `classifier.variables` 에 있는 변수들
- probability feature

classifier 학습 target은 level 예측값이 아니라 **extreme 여부 binary signal** 이다.

---

## 11. branch 실행 방식

현재 branch 실행은 다음 흐름이다.

1. branch별 synthetic dataset 구성
2. branch별 synthetic target column 생성
   - `classifier` 는 binary target
   - `normal` / `extreme` 는 diff-normalized target
3. branch별 JobConfig 생성
4. shared `runtime_support.forecast_models.build_model()` 로 branch model 생성
5. branch용 `NeuralForecast` 실행
6. branch 예측 산출
7. classifier는 probability로 변환 / clip
8. normal/extreme는 level forecast로 복원

즉 현재 NEC는:
- branch orchestration은 NEC plugin이 직접 하고
- branch backbone 생성은 shared forecast model builder를 재사용한다.

---

## 12. final merge 방식

현재 final merge는 linked NEC YAML의 `inference.mode` 로 결정된다.

### 12.1 `soft_weighted`
- classifier probability를 그대로 weight로 사용
- `extreme * p + normal * (1-p)`

### 12.2 `hard_threshold`
- classifier probability를 threshold와 비교
- extreme 또는 normal 중 하나를 선택

최종 산출 컬럼은 항상 top-level prediction DataFrame의 `NEC` 컬럼이다.

---

## 13. 현재 산출물 구조

### 13.1 plugin-private stage artifact
실행 시 run root 아래에 다음 NEC stage artifact가 생긴다.

- `run_root/nec/config/config.resolved.json`
- `run_root/nec/manifest/run_manifest.json`
- `run_root/nec/nec_fold_summary.json`

현재 `nec_fold_summary.json` 에는 예를 들어 다음 성격의 값이 들어간다.
- `history_steps_source`
- `history_steps_value`
- `probability_feature_forced`
- `active_hist_columns`
- `merge_mode`
- `inference_threshold`
- branch별 model / variables / row count

### 13.2 summary artifact
일반 실행 시 fold별 branch summary artifact는 다음 위치에 쌓인다.

- `run_root/summary/nec/normal/fold_XXX.csv`
- `run_root/summary/nec/normal/fold_XXX.png`
- `run_root/summary/nec/extreme/fold_XXX.csv`
- `run_root/summary/nec/extreme/fold_XXX.png`
- `run_root/summary/nec/classifier/fold_XXX.csv`
- `run_root/summary/nec/classifier/fold_XXX.png`

여기서:
- CSV는 branch 예측 결과
- PNG는 branch 예측 그래프
- classifier CSV는 probability 관련 값도 포함한다

즉 운영자 입장에서는 fold별 NEC 내부 branch 결과를 `summary/nec` 아래에서 직접 볼 수 있다.

---

## 14. resolved / manifest / capability에 반영되는 NEC metadata

현재 top-level resolved / manifest / capability report에도 NEC 정보가 병합된다.

핵심적으로 노출되는 것은:
- selected config path
- branch model / branch variables
- active_hist_columns
- history_steps_source / history_steps_value
- probability_feature_forced
- inference_mode / threshold
- preprocessing_mode / gmm_components / epsilon
- shared scaler가 무엇이었는지
- `summary_nec_root`

즉 validate-only나 일반 실행 후에는 NEC가 어떤 branch contract로 실행되었는지 top-level metadata에서도 확인 가능하다.

---

## 15. 현재 fail-fast 조건

현재 NEC는 다음 상황에서 즉시 에러를 낸다.

- linked NEC YAML top-level 형식이 잘못됨
- 제거된 legacy key 사용
  - `history_steps`
  - `hist_columns`
  - `preprocessing.probability_feature`
- branch block에 `model` 또는 `variables` 누락
- branch variable이 dataset column이 아님
- branch variable에 target_col을 직접 포함함
- branch model이 exported catalog에 없거나 shared runtime catalog에 없음
- branch model이 multivariate 이거나 NEC input contract와 안 맞음
- branch `model_params` 에 unsupported key 포함
- training / future frame에 필요한 column이 없거나 NaN 존재
- preprocessing mode가 `diff_std` 가 아님
- diff preprocessing 표준편차가 0
- fold 학습 길이가 `training.input_size + horizon` 보다 부족함
- branch 예측 row 수가 horizon과 안 맞음

현재 동작 원칙은 일관되게:
- 조용한 fallback 대신
- 가능한 한 이른 단계에서
- 이유를 포함한 명시적 에러
이다.

---

## 16. 운영자가 현재 기억하면 되는 핵심

현재 NEC는 다음처럼 이해하면 가장 정확하다.

1. NEC는 top-level plugin-owned job이다.
2. 메인 YAML의 `nec:` 는 thin entrypoint다.
3. 실제 NEC 세부 설정은 linked NEC YAML이 가진다.
4. linked NEC YAML은 branch별 `model / model_params / variables` 를 가진다.
5. `training.input_size` 가 NEC history 길이의 단일 기준이다.
6. probability feature는 항상 켜진다.
7. branch model은 exported model surface 기반이지만 compatibility gate를 통과해야 한다.
8. classifier는 gate probability branch다.
9. 일반 실행 시 fold별 branch 결과는 `summary/nec` 아래에 저장된다.
10. validate-only에서도 NEC branch contract와 metadata를 충분히 확인할 수 있다.

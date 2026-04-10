
[Abstract]

본 연구는 AA-Forecast의 3단계 구조를 준용한다. 먼저 STAR 분해를 통해 시계열을 추세·계절성·이상치·잔차로 분해하여 급격한 변동과 이상치 정보를 추출한다. 다음으로 Anomaly-Aware과 Attention을 통해 이상치 정보를 모형에 반영하고, Dynamic Uncertainty Optimization을 통해 최적의 Dropout을 도출하여 불확실성을 최소화한다. 본 연구는 선행연구의 구조를 유지하되, Anomaly-Aware 단계의 예측 모듈을 장기 의존성 파악에 유리한 Transformer 모형으로 대체하였다.

본 연구는 2015년 1월부터 2026년 3월까지의 584주의 데이터를 활용하여 Brent 원유 가격 예측을 수행했다. 거시경제 변수와 함께 블랙스완 지수 및 지정학적 위험 지수를 설명변수로 활용하였다. 실험 결과, AA-Forecast와 Transformer를 결합한 모형은 Baseline 모형 대비 우수한 예측 성능을 보였다. 이는 블랙스완 지수와 지정학적 위험 지수가 기존 변수로는 설명하기 어려운 외부 충격과 구조적 변동성을 보완하며, AA-Forecast 구조가 이를 모형에 효과적으로 통합함을 시사한다.


[Goal]

yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml 실험의

main.py 결과가 전체적으로 상승 추세를 예측하는 방향으로 나와야됨.

[Abstract]에 있는 목표 달성을 목표로함

- 마지막 fold 4-step 예측이 y_hat4 > y_hat1
- 각 점이 실제값 대비 ±15% 이내

실험 설계는 [Abstract]에 맞계 설계 되어야하며, 과학점 탐구의
조작 변인, 종속변인, 통제변인에 기반하여 실험 진행할 것.

- 타깃 'Com_BrentCrudeOil'

-  마지막에 모든 조건(우상향 + 마지막 fold 4점 모두 실제값 대비 15% 이내 + no leakage + 20분 이내)을 만족하면, 최종 승자가
반드시 Transformer 계열(VanillaTransformer / Informer / TimeXer / PatchTST)이어야함.

- Transformer가 우상향 + 마지막 fold 4점 모두 15% 이내를 만족하고 GRU는 못 하면 승
- Transformer 계열 중 하나만이라도 GRU보다 좋아지면 됨.

- 위 조건을 불만족하지만, summary/test_2 에서 h1, h2 모두 실제값 대비 15% 이내라면 PASS하는 것으로

- yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-gru.yaml (without aaforecast)
- yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-informer.yaml
- yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-patchtst.yaml
- yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-timexer.yaml
- yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-vanillatransformer.yaml
- yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-gru.yaml

가장 먼저 보여줘야하는 것은 gru로만 놓고 비교하고, aaforecast를 썼을 때가 마지막 폴드에서 
이상치임을 감지하고, 향후 급등하는 구간을 잘 포착한다는 것이 증명이 되어야하고,
이게 증명이 되면, aa-forecast에서 gru 보다 transformer가 향후 급등하는 구간을 잘 포착한다는 것을 보여주는 식으로 실험이 진행되어야함




[Do]

- 모든 실험 시작 전에 https://docs.google.com/spreadsheets/d/11ihatqjZuRtC8OjUktfebKcv9L39Tz20pyj9w6oHnrE/edit?gid=1233442663#gid=1233442663 를 which gws 로 접속하여 이전 실험들을 확인하고, 매 실험 종료 (iter 종료) 시마다, 실험 내용을 해당 구글 스프레드 시트의 양식에 맞게 작성하도록.

- 절대로 기존에 진행한 실험을 동일하게 진행하지말고, 기존에 진행한 실험에서 유의미한 부분을 파고들 수 있도록 할 것.

- 단 변수는 아래 명시된 변수만 사용할 것.

    - GPRD_THREAT
    - BS_Core_Index_A
    - GPRD
    - GPRD_ACT
    - BS_Core_Index_B
    - BS_Core_Index_C
    - Idx_OVX
    - Com_Oil_Spread
    - Com_LMEX
    - Com_BloombergCommodity_BCOM
    - Idx_DxyUSD


- 모든 실험은 코드 수정이 발생할 수 있으므로 main 브랜치가 아닌 
 새 실험 브랜치에서 새로 시작, 실험용으로 별도의 브랜치를 만들면서 해당 브랜치에서 진행하도록할 것.

- data/df.csv 의 데이터 EDA를 통한 yaml/ 변경을 해도 됨.


- aaforecast 아키텍쳐를 벗어나지 않는 선에서 내 코드 수정을 진행해도됨
reference\aaforecast 에 있는 aaforecast 논문 이미지가 아키텍쳐임
(https://github.com/ashfarhangi/AA-Forecast 논문 github)

- aaforecast 예측 아키텍쳐 흐름 star 분해 -> aa-model -> 몬테카를로 시뮬레이션 이 포멧은 유지하되, 현재 모델이 GRU를 메인으로 사용하나, 
내부 인코더를 gru 말고, 아래 모델들로 사용하는 것을 허용함.
모델 파라미터 사이즈는 0.1M ~ 0.5M 사이에서 하이퍼파라미터 세팅해서 진행하도록 

- GRU = control/warm-start
- VanillaTransformer / Informer / TimeXer / PatchTST = 승격 후보

neuralforecast/models/ 에 있는 모델 코드들을 참고하나,
neuralforecast/models/aaforecast 에 aaforecast 플러그인 용 모델이 되도록 할 것.

- GRU 중심으로 시작, 필요할 때만 다른 모델로 이동해도됨.

- iter 50까지 진행. non-stop
- 실험 진행 내용에 대해서는 autoresearch_record.md에 업데이트하면서 진행
- autoresearch_record.md에 지금은 어떤 실험이었는지 안적혀있는데, autoresearch_record.md에 어떤 브랜치의 어떤 실험이었고, 어떤 yaml을 사용하였는지가 있어야함.


- 이런시도도 가능
  transformations_target: diff
  transformations_exog: diff


- autoresearch_record.md 기록 스키마
    - timestamp
    - git branch
    - experiment title
    - main config path
    - plugin config path
    - encoder family
    - 바꾼 조작변인
    - 고정한 통제변인
    - 실행 명령
    - run/artifact path
    - 상승추세 PASS/FAIL
    - 15% band PASS/FAIL
    - runtime PASS/FAIL
    - leakage concern 여부


- 모든 실험은 

aaforecast 없는 informer와 gru baseline yaml/experiment/feature_set_aaforecast/brentoil-case1-baseline.yaml
aaforecast 있는 informer와 gru 실험 yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-gru.yaml, yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-informer.yaml

위 4개를 비교해서 

aaforecast 적용 inforer > aaforeast 적용 gru > aaforecast 미적용 꼴이 나오도록 매변 비교해야함.

uv run main.py --config 



--- 

[Restriction]

- 우상향을 목표로 의도적으로 +방향 드리프트시키는 건 leaking 성격이라 올바르지 않음.
- 최근 타깃 상승분을 이용해 미래 horizon uplift하는 것은 금지

- 단일 모델 학습시 20분 이상 걸리면 안됨.
- 절대 leaking이 발생해서는 안 됨.
- 기존의 input 만큼 데이터를 넣어서 한 번에 향후 n 개 호라이즌을 예측하는 형태 자체는 흐트러지면 안됨

- 모든 조건 PASS 전까지 리서치 멈추지 말 것.

-  autoresearch_record.md 에서 이미 진행한 실험은 진행하지 말도록 할 것.

[Abstract]

본 연구는 AA-Forecast의 3단계 구조를 준용한다. 먼저 STAR 분해를 통해 시계열을 추세·계절성·이상치·잔차로 분해하여 급격한 변동과 이상치 정보를 추출한다. 다음으로 Anomaly-Aware과 Attention을 통해 이상치 정보를 모형에 반영하고, Dynamic Uncertainty Optimization을 통해 최적의 Dropout을 도출하여 불확실성을 최소화한다. 본 연구는 선행연구의 구조를 유지하되, Anomaly-Aware 단계의 예측 모듈을 장기 의존성 파악에 유리한 Transformer 모형으로 대체하였다.

본 연구는 2015년 1월부터 2026년 3월까지의 584주의 데이터를 활용하여 Brent 원유 가격 예측을 수행했다. 거시경제 변수와 함께 블랙스완 지수 및 지정학적 위험 지수를 설명변수로 활용하였다. 실험 결과, AA-Forecast와 Transformer를 결합한 모형은 Baseline 모형 대비 우수한 예측 성능을 보였다. 이는 블랙스완 지수와 지정학적 위험 지수가 기존 변수로는 설명하기 어려운 외부 충격과 구조적 변동성을 보완하며, AA-Forecast 구조가 이를 모형에 효과적으로 통합함을 시사한다.


[Goal]

실험 프로젝트 주소 : /home/sonet/.openclaw/workspace/research/neuralforecast

main.py 결과가 전체적으로 상승 추세를 예측하는 방향으로 나와야됨.

[Abstract]에 있는 목표 달성을 목표로함

- 마지막 fold 4-step 예측이 y_hat2 > y_hat1
- 각 점이 실제값 대비 ±15% 이내

실험 설계는 [Abstract]에 맞계 설계 되어야하며, 과학점 탐구의
조작 변인, 종속변인, 통제변인에 기반하여 실험 진행할 것.

- 타깃 'Com_BrentCrudeOil'

- informer 우상향 + 마지막 fold 2개 포인트 모두 15% 이내를 만족하고 GRU는 못 하면 승

- 위 조건을 불만족하지만, summary/test_2 에서 h1, h2 모두 실제값 대비 15% 이내라면 PASS하는 것으로

가장 먼저 보여줘야하는 것은 gru로만 놓고 비교하고, aaforecast를 썼을 때가 마지막 폴드에서 
이상치임을 감지하고, 향후 급등하는 구간을 잘 포착한다는 것이 증명이 되어야하고,
이게 증명이 되면, aa-forecast에서 gru 보다 transformer 계열이 향후 급등하는 구간을 잘 포착한다는 것을 보여주는 식으로 실험이 진행되어야함

- AA-GRU 는 상관 없고, transformer family만 15% 위로 올리면 성공한 것으로 판단


- 모든 실험은 

aaforecast 적용 안 된 transformer family gru 실험

yaml/experiment/feature_set_aaforecast/baseline.yaml

aaforecast 된 transformer family 와 gru 실험

yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml
yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml
yaml/experiment/feature_set_aaforecast/aaforecast-vanillatransformer


위 4개 케이스들을 비교해서 
전체적인 성능 (mape, nrmse 및 마지막 fold 예측 2개 포인트에 대한 오차범위)
aaforecast 적용 inforer > aaforeast 적용 gru > aaforecast 미적용 꼴이 나오도록 매변 비교해야함.

이 부분에 대한 결과는 gws 시트 상에 별도의 컬럼으로 다뤄지지 않고 있다면 스키마 수정해서 진행할 것.

uv run main.py --config yaml/experiment/feature_set_aaforecast/baseline.yaml
uv run main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml
uv run main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml
....

이런식으로 매 iter마다 실험 비교 진행하면서 진행할 것.

이전 실험과 겹치지 않도록 실험 결과 생성되면 runs/iter_*_~~ 폴더만들고 거기에 생성된 3개 runs 결과 옮기도록해줘.

반드시 이 4개의 결과가 하나의 실험 단위가 되어야함.



[Do]

- 모든 실험 시작 전에 https://docs.google.com/spreadsheets/d/11ihatqjZuRtC8OjUktfebKcv9L39Tz20pyj9w6oHnrE/edit?gid=1233442663#gid=1233442663 를 which gws 로 접속하여 이전 실험들을 확인하고, 매 실험 종료 (iter 종료) 시마다, 실험 내용을 해당 구글 스프레드 시트의 양식에 맞게 작성하도록.

- 절대로 기존에 진행한 실험을 동일하게 진행하지말고, 기존에 진행한 실험에서 유의미한 부분을 파고들 수 있도록 할 것.


- 절대로 기존에 진행한 실험을 동일하게 진행하지말고, 기존에 진행한 실험에서 유의미한 부분 분석을 진행 후, 파고들 수 있도록 할 것.

- 도저히 개선이 안 보일 경우 전혀 다른 방법을 리서치나 data driven으로 solving해도 됨. 다만 [Restriction] 는 철저하게 지킬 것

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
 새 실험 브랜치에서 새로 시작, 실험용으로 별도의 브랜치를 만들면서 해당 브랜치 하나에서 진행하도록할 것.

- data/df.csv 의 데이터 EDA를 통한 yaml/ 변경을 해도 됨.

- aaforecast 아키텍쳐를 벗어나지 않는 선에서 내 코드 수정을 진행해도됨
reference\aaforecast 에 있는 aaforecast 논문 이미지가 아키텍쳐임
(https://github.com/ashfarhangi/AA-Forecast 논문 github)

- aaforecast 예측 아키텍쳐 흐름 star 분해 -> aa-model -> 몬테카를로 시뮬레이션 이 포멧은 유지하되

- 필요 시 setting.yaml 수정해도됨.
- GPU 2장 필요하다면 다 사용하도록.

--- 

[Restriction]

- h2 블렌딩에 추가적인 continuation 보너스를 주어서 반드시 오를 것이다라는 것을 인지하고 결과를 보정하는 것은 금지
- 우상향을 목표로 의도적으로 +방향 드리프트시키는 건 leaking 성격이라 올바르지 않음.
- 최근 타깃 상승분을 이용해 미래 horizon uplift하는 것은 금지

- 단일 모델 학습시 20분 이상 걸리면 안됨.
- 절대 leaking이 발생해서는 안 됨.

- setting.yaml 에서
 horizon: 2 , n_windows: 1 유지 

 - 변경가능
  transformations_target: diff
  transformations_exog: diff
  input_size: 64
  

- 모든 조건 PASS 전까지 리서치 멈추지 말 것.
- h1 이 75 이상 예측 h2가 80이상 예측이 나와야함.


---


현재 결과가 runs/ 실험 내용들 보면 알겠지만, h1, h2가 전체적으로 70중후반대에 예측이 갇히는 현상이 발생함.

이 부분을 타게해야되는데 data-driven으로 문제정의를하고 model-driven으로 개선을 진행하고자함.

절대 특정 호라이즌에 보상을 주거나하는 식으로 조절하지말고 모델 기반으로 개선을 이어나가고자함.

만약 이런 부분이 기존에 있었다면 해당 부분은 제거하고 진행하도록

코드 구현은 KISS, DRY 법칙을 최대한 지키도록,

큰 방향은 



  retrieval:
    enabled: true
    top_k: 1
    event_score_threshold: 400.0
    min_similarity: 0.35
    blend_max: 1.0
    use_uncertainty_gate: false
    use_shape_key: false
    use_event_key: true
    event_score_log_bonus_alpha: 0.15
    event_score_log_bonus_cap: 0.1


이 플러그인을 설정해서 하는 경우 성능이 되게 잘나오는데 

해당 철학을 aa-forecast 인코더 디코더 부분에 transformer로 녹여내서 고도화를해보고 싶어.

runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer
결과를 기반으로 분석해도됨.


지금 단순히 dense 만으로 예측 값을 내고 있지는 않은지도 추가 점검해서 고도화 

인코딩이나 디코더가 데이터 이해를 잘 못하는 모델의 문제는 아닌지 이 부분도 검토해보도록


잘못나오는 예측에 대해서는

전체 적으로 모델이 학습이 제대로 된것에 대해 의심이 되는데,
과거에 유가가 급등했던 부분을 학습했던 윈도우를 학습 완료된 모델에 다시 input으로 넣어서 예측 결과를 실제와 비교해보는 것도 진행해보기도하면서,



점검할 수 있는 모든 것들은 점검하되, 절대로 특정 호라이즌에 가중치를 주거나 호라이즌 driven한 그런 분석은 진행하지말고 분석 결과로 이런 데이터에서 어떻게 아키텍쳐를 수정을해야 지금 나오는 스파이크를 aa-forecast 가 포착하여 향후 스파이크를 예측으로 담아낼지에 더 초점을 맞추는 식으로 고도화 진행해줘.


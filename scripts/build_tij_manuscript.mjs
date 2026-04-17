import { readFile, writeFile, mkdir } from "fs/promises"
import { dirname, resolve } from "path"
import { fileURLToPath } from "url"

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const dataDir = resolve(rootDir, "data")
const runsDir = resolve(rootDir, "runs", "final_wti")
const outDir = resolve(rootDir, "docss")
const outMarkdown = resolve(outDir, "기술혁신학회지_WTI_원고.md")
const outHwpx = resolve(outDir, "기술혁신학회지_WTI_원고.hwpx")

const { markdownToHwpx } = await import("../tmp/kordoc/src/hwpx/generator.ts")

const fmt2 = (value) => Number(value).toLocaleString("ko-KR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const fmt3 = (value) => Number(value).toLocaleString("ko-KR", { minimumFractionDigits: 3, maximumFractionDigits: 3 })
const pct = (value) => `${fmt2(value * 100)}`
const pctWithSign = (value) => `${value >= 0 ? "+" : ""}${fmt2(value)}`

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/)
  const headers = lines[0].split(",")
  return lines.slice(1).map((line) => {
    const cols = line.split(",")
    const row = {}
    headers.forEach((header, index) => {
      row[header] = cols[index]
    })
    return row
  })
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function stats(values) {
  const n = values.length
  const m = mean(values)
  const variance = values.reduce((sum, value) => sum + (value - m) ** 2, 0) / (n - 1)
  const std = Math.sqrt(variance)
  const m2 = values.reduce((sum, value) => sum + (value - m) ** 2, 0) / n
  const m3 = values.reduce((sum, value) => sum + (value - m) ** 3, 0) / n
  const m4 = values.reduce((sum, value) => sum + (value - m) ** 4, 0) / n
  const skew = m2 === 0 ? 0 : m3 / (m2 ** 1.5)
  const kurt = m2 === 0 ? 0 : m4 / (m2 ** 2) - 3
  return {
    mean: m,
    min: Math.min(...values),
    max: Math.max(...values),
    std,
    skew,
    kurt,
  }
}

function toTable(headers, rows) {
  const header = `| ${headers.join(" | ")} |`
  const divider = `| ${headers.map(() => "---").join(" | ")} |`
  const body = rows.map((row) => `| ${row.join(" | ")} |`).join("\n")
  return [header, divider, body].join("\n")
}

function readMetricRow(path) {
  return parseCsv(requireText(path))[0]
}

async function requireText(path) {
  return await readFile(path, "utf-8")
}

async function loadLeaderboard(relPath) {
  const rows = parseCsv(await requireText(resolve(runsDir, relPath)))
  const result = {}
  for (const row of rows) {
    result[row.model] = row
  }
  return result
}

async function loadMetrics(relPath) {
  const rows = parseCsv(await requireText(resolve(runsDir, relPath)))
  return rows[0]
}

const df = parseCsv(await requireText(resolve(dataDir, "df.csv")))

const statsColumns = [
  ["WTI 원유가격", "Com_CrudeOil"],
  ["GPRD_THREAT", "GPRD_THREAT"],
  ["GPRD", "GPRD"],
  ["GPRD_ACT", "GPRD_ACT"],
  ["OVX", "Idx_OVX"],
  ["LMEX", "Com_LMEX"],
  ["BCOM", "Com_BloombergCommodity_BCOM"],
  ["DXY", "Idx_DxyUSD"],
]

const statsRows = statsColumns.map(([label, key]) => {
  const values = df.map((row) => Number(row[key])).filter((value) => Number.isFinite(value))
  const s = stats(values)
  return [
    label,
    fmt2(s.mean),
    fmt2(s.min),
    fmt2(s.max),
    fmt2(s.std),
    fmt3(s.skew),
    fmt3(s.kurt),
  ]
})

const directLeaderboard = await loadLeaderboard("feature_set_aaforecast_wti_brentoil_baseline_gru_informer/summary/leaderboard.csv")
const directRetLeaderboard = await loadLeaderboard("feature_set_aaforecast_wti_brentoil_baseline-ret_gru_informer/summary/leaderboard.csv")
const aaInformer = await loadLeaderboard("feature_set_aaforecast_wti_aaforecast_informer/summary/leaderboard.csv")
const aaInformerRet = await loadLeaderboard("feature_set_aaforecast_wti_aaforecast_informer-ret/summary/leaderboard.csv")
const aaGru = await loadLeaderboard("feature_set_aaforecast_wti_aaforecast_gru/summary/leaderboard.csv")
const aaGruRet = await loadLeaderboard("feature_set_aaforecast_wti_aaforecast_gru-ret/summary/leaderboard.csv")
const aaTimexer = await loadLeaderboard("feature_set_aaforecast_wti_aaforecast_timexer/summary/leaderboard.csv")
const aaTimexerRet = await loadLeaderboard("feature_set_aaforecast_wti_aaforecast_timexer-ret/summary/leaderboard.csv")

const aaInformerMetrics = await loadMetrics("feature_set_aaforecast_wti_aaforecast_informer/cv/AAForecast_metrics_by_cutoff.csv")
const aaInformerRetMetrics = await loadMetrics("feature_set_aaforecast_wti_aaforecast_informer-ret/cv/AAForecast_metrics_by_cutoff.csv")
const aaGruMetrics = await loadMetrics("feature_set_aaforecast_wti_aaforecast_gru/cv/AAForecast_metrics_by_cutoff.csv")
const aaGruRetMetrics = await loadMetrics("feature_set_aaforecast_wti_aaforecast_gru-ret/cv/AAForecast_metrics_by_cutoff.csv")
const aaTimexerMetrics = await loadMetrics("feature_set_aaforecast_wti_aaforecast_timexer/cv/AAForecast_metrics_by_cutoff.csv")
const aaTimexerRetMetrics = await loadMetrics("feature_set_aaforecast_wti_aaforecast_timexer-ret/cv/AAForecast_metrics_by_cutoff.csv")
const directInformer = directLeaderboard.Informer
const directGru = directLeaderboard.GRU
const directTimexer = directLeaderboard.TimeXer
const directRetInformer = directRetLeaderboard.Informer
const directRetGru = directRetLeaderboard.GRU
const directRetTimexer = directRetLeaderboard.TimeXer

const dataSummaryTable = toTable(
  ["변수", "평균", "최솟값", "최댓값", "표준편차", "왜도", "첨도"],
  statsRows,
)

const experimentTable = toTable(
  ["항목", "값"],
  [
    ["표본", `${df.length}주`],
    ["기간", `${df[0].dt} ~ ${df[df.length - 1].dt}`],
    ["타깃", "WTI 원유가격(Com_CrudeOil)"],
    ["설명변수", "GPRD_THREAT, GPRD, GPRD_ACT, Idx_OVX, Com_LMEX, Com_BloombergCommodity_BCOM, Idx_DxyUSD"],
    ["입력 길이", "64주"],
    ["예측 지평", "2주"],
    ["검증 방식", "단일 rolling cutoff(2026-02-23)"],
    ["손실 함수", "MSE"],
    ["AA-Forecast lowess_frac", "0.359"],
    ["AA-Forecast lowess_delta", "0.008"],
    ["AA-Forecast thresh", "3.08"],
    ["AA-Forecast season_length", "4"],
    ["Retrieval top_k", "1"],
    ["Retrieval recency_gap_steps", "16"],
    ["Retrieval trigger_quantile", "0.0126"],
    ["Retrieval blend_floor", "0.088"],
    ["Retrieval blend_max", "0.894"],
  ],
)

const hypothesesTable = toTable(
  ["가설 번호", "개념", "가설 내용", "검증 위치"],
  [
    ["가설 1", "외생충격 변수 효과", "GPR 계열 충격 변수와 블랙스완류 이벤트 정보를 포함한 모델이 충격 구간에서 더 나은 설명력을 보일 것이다.", "IV.3, IV.5"],
    ["가설 2", "Retrieval 구조 효과", "Retrieval 구조를 적용한 모델이 적용하지 않은 모수적 모델보다 스파이크 구간 방향성/개형 포착이 우수할 것이다.", "IV.4"],
    ["가설 3", "이상치-Retrieval 시너지", "STAR 기반 이상치 피처를 Retrieval 쿼리에 넣은 결합 모델이 단독 효과보다 더 안정적으로 작동할 것이다.", "IV.5"],
    ["가설 4", "통합 파이프라인 우월성", "이상치 인지 전처리, Retrieval, Informer를 통합한 제안 파이프라인이 baseline 대비 전체 오차를 낮출 것이다.", "IV.4, IV.5"],
  ],
)

const modelComparisonTable = toTable(
  ["구분", "모형", "MAE", "RMSE", "MAPE(%)", "R2"],
  [
    ["Baseline", "GRU", fmt2(Number(directGru.mean_fold_mae)), fmt2(Number(directGru.mean_fold_rmse)), fmt2(Number(directGru.mean_fold_mape) * 100), fmt3(Number(directGru.mean_fold_r2))],
    ["Baseline", "Informer", fmt2(Number(directInformer.mean_fold_mae)), fmt2(Number(directInformer.mean_fold_rmse)), fmt2(Number(directInformer.mean_fold_mape) * 100), fmt3(Number(directInformer.mean_fold_r2))],
    ["Baseline", "TimeXer", fmt2(Number(directTimexer.mean_fold_mae)), fmt2(Number(directTimexer.mean_fold_rmse)), fmt2(Number(directTimexer.mean_fold_mape) * 100), fmt3(Number(directTimexer.mean_fold_r2))],
    ["Baseline+Ret", "GRU", fmt2(Number(directRetGru.mean_fold_mae)), fmt2(Number(directRetGru.mean_fold_rmse)), fmt2(Number(directRetGru.mean_fold_mape) * 100), fmt3(Number(directRetGru.mean_fold_r2))],
    ["Baseline+Ret", "Informer", fmt2(Number(directRetInformer.mean_fold_mae)), fmt2(Number(directRetInformer.mean_fold_rmse)), fmt2(Number(directRetInformer.mean_fold_mape) * 100), fmt3(Number(directRetInformer.mean_fold_r2))],
    ["Baseline+Ret", "TimeXer", fmt2(Number(directRetTimexer.mean_fold_mae)), fmt2(Number(directRetTimexer.mean_fold_rmse)), fmt2(Number(directRetTimexer.mean_fold_mape) * 100), fmt3(Number(directRetTimexer.mean_fold_r2))],
    ["AA-only", "GRU", fmt2(Number(aaGru["AAForecast"].mean_fold_mae)), fmt2(Number(aaGru["AAForecast"].mean_fold_rmse)), fmt2(Number(aaGru["AAForecast"].mean_fold_mape) * 100), fmt3(Number(aaGru["AAForecast"].mean_fold_r2))],
    ["AA-only", "Informer", fmt2(Number(aaInformer["AAForecast"].mean_fold_mae)), fmt2(Number(aaInformer["AAForecast"].mean_fold_rmse)), fmt2(Number(aaInformer["AAForecast"].mean_fold_mape) * 100), fmt3(Number(aaInformer["AAForecast"].mean_fold_r2))],
    ["AA-only", "TimeXer", fmt2(Number(aaTimexer["AAForecast"].mean_fold_mae)), fmt2(Number(aaTimexer["AAForecast"].mean_fold_rmse)), fmt2(Number(aaTimexer["AAForecast"].mean_fold_mape) * 100), fmt3(Number(aaTimexer["AAForecast"].mean_fold_r2))],
    ["AA+Ret", "GRU", fmt2(Number(aaGruRet["AAForecast"].mean_fold_mae)), fmt2(Number(aaGruRet["AAForecast"].mean_fold_rmse)), fmt2(Number(aaGruRet["AAForecast"].mean_fold_mape) * 100), fmt3(Number(aaGruRet["AAForecast"].mean_fold_r2))],
    ["AA+Ret", "Informer", fmt2(Number(aaInformerRet["AAForecast"].mean_fold_mae)), fmt2(Number(aaInformerRet["AAForecast"].mean_fold_rmse)), fmt2(Number(aaInformerRet["AAForecast"].mean_fold_mape) * 100), fmt3(Number(aaInformerRet["AAForecast"].mean_fold_r2))],
    ["AA+Ret", "TimeXer", fmt2(Number(aaTimexerRet["AAForecast"].mean_fold_mae)), fmt2(Number(aaTimexerRet["AAForecast"].mean_fold_rmse)), fmt2(Number(aaTimexerRet["AAForecast"].mean_fold_mape) * 100), fmt3(Number(aaTimexerRet["AAForecast"].mean_fold_r2))],
  ],
)

const hypothesisValidationTable = toTable(
  ["가설", "검증 상태", "핵심 증거", "해석"],
  [
    ["가설 1", "부분 검증", "final_wti에는 GPR/OVX 충격 변수는 존재하나 블랙스완 지수의 별도 제거 실험은 없음.", "외생충격 입력은 모델 설정에 반영되었지만, 블랙스완 항목은 본 결과셋에서 독립 검정되지 않았다."],
    ["가설 2", "지지", `Informer MAPE ${fmt2(Number(directInformer.mean_fold_mape) * 100)}% → ${fmt2(Number(directRetInformer.mean_fold_mape) * 100)}%, TimeXer ${fmt2(Number(directTimexer.mean_fold_mape) * 100)}% → ${fmt2(Number(directRetTimexer.mean_fold_mape) * 100)}%`, "Retrieval 도입만으로도 스파이크 구간 및 전체 오차가 크게 줄었다."],
    ["가설 3", "부분 지지", `AA-only Informer ${fmt2(Number(aaInformer["AAForecast"].mean_fold_mape) * 100)}% 대비 AA+Ret ${fmt2(Number(aaInformerRet["AAForecast"].mean_fold_mape) * 100)}%는 개선되지만, Retrieval-only Informer ${fmt2(Number(directRetInformer.mean_fold_mape) * 100)}%에는 소폭 미달.`, "이상치 피처는 도움이 되지만, 모든 backbone에서 단순 합산보다 항상 우월하다고 보기는 어렵다."],
    ["가설 4", "지지", `AA-only Informer ${fmt2(Number(aaInformer["AAForecast"].mean_fold_mape) * 100)}% → AA+Ret ${fmt2(Number(aaInformerRet["AAForecast"].mean_fold_mape) * 100)}%`, "통합 파이프라인은 AA-only 대비 확실한 개선을 보였다."],
  ],
)

const appendixTable = toTable(
  ["설정", "값"],
  [
    ["random_seed", "1"],
    ["opt_study_count", "1"],
    ["opt_n_trial", "50"],
    ["training.max_steps", "400"],
    ["training.batch_size", "32"],
    ["training.valid_batch_size", "8"],
    ["training.windows_batch_size", "64"],
    ["training.inference_windows_batch_size", "64"],
    ["training.val_size", "4"],
    ["training.scaler_type", "robust"],
    ["cv.horizon", "2"],
    ["cv.step_size", "1"],
    ["cv.n_windows", "1"],
    ["aa_forecast.model_params.hidden_size", "128"],
    ["aa_forecast.model_params.n_head", "8"],
    ["aa_forecast.model_params.encoder_layers", "2"],
    ["aa_forecast.model_params.decoder_layers", "4"],
  ],
)

const fullWeeks = df.length
const firstDate = df[0].dt
const lastDate = df[df.length - 1].dt
const wtiStats = statsColumns[0] && statsRows[0]

const narrative = `
# 기술혁신학회지 논문 원고

## 국문 초록
본 연구는 WTI 주간 가격 예측에서 이상치 인지와 Retrieval 결합이 스파이크 구간 성능을 어떻게 바꾸는지 검토한다. 데이터는 ${fullWeeks}주의 주간 관측치(${firstDate}~${lastDate})이며, 설명변수는 GPRD_THREAT, GPRD, GPRD_ACT, Idx_OVX, Com_LMEX, Com_BloombergCommodity_BCOM, Idx_DxyUSD로 구성된다. 실험은 AA-only, Baseline, Retrieval-only, AA+Retrieval의 네 계열을 비교했다. 결과적으로 direct Informer의 MAPE는 ${fmt2(Number(directInformer.mean_fold_mape) * 100)}%였으나 Retrieval을 결합하면 ${fmt2(Number(directRetInformer.mean_fold_mape) * 100)}%로 낮아졌고, AA-only Informer는 ${fmt2(Number(aaInformer["AAForecast"].mean_fold_mape) * 100)}%에 머물렀다. AA+Retrieval Informer는 ${fmt2(Number(aaInformerRet["AAForecast"].mean_fold_mape) * 100)}%를 기록해, 이상치 인지와 Retrieval을 함께 쓰는 설계가 단일 AA-only보다 실질적인 개선을 가져온다는 점을 보여준다.

주제어: WTI, Retrieval-Augmented Forecasting, AA-Forecast, Informer, 지리정치 위험지수

## Abstract
This study examines whether anomaly-aware preprocessing and Retrieval improve weekly WTI forecasting under spike-heavy market regimes. We evaluate ${fullWeeks} weekly observations from ${firstDate} to ${lastDate} with seven exogenous variables: GPRD_THREAT, GPRD, GPRD_ACT, Idx_OVX, Com_LMEX, Com_BloombergCommodity_BCOM, and Idx_DxyUSD. Across the final runs, direct Informer achieves ${fmt2(Number(directInformer.mean_fold_mape) * 100)}% MAPE, while Informer+Retrieval reduces it to ${fmt2(Number(directRetInformer.mean_fold_mape) * 100)}%. The AA-only Informer remains at ${fmt2(Number(aaInformer["AAForecast"].mean_fold_mape) * 100)}%, but AA+Retrieval improves to ${fmt2(Number(aaInformerRet["AAForecast"].mean_fold_mape) * 100)}%. The results indicate that Retrieval is the main driver of the gain, while anomaly-aware features help the combined pipeline remain competitive.

Keywords: WTI, Retrieval-Augmented Forecasting, AA-Forecast, Informer, Geopolitical Risk Index

## I. 서론
WTI 원유가격은 지정학적 충격, 수급 불균형, 환율 변화, 금융 변동성이 겹칠 때 급격한 스파이크를 보인다. 평균 회귀 성향의 시계열 모형은 정상 구간에서는 그럴듯한 성능을 내더라도, 위기 구간에서는 방향성과 개형을 놓치기 쉽다. 본 연구는 STAR 계열 이상치 인지와 Retrieval을 결합해, 희소한 충격 구간을 외부 메모리와 명시적 이상치 피처로 보완하는 접근을 시험한다.

## II. 데이터와 연구 설계
데이터는 주간 WTI 가격과 외생 변수 7개로 구성된다. 표본은 ${fullWeeks}주이며, 타깃은 Com_CrudeOil로 정렬했다. 실험 설정은 입력 길이 64주, 예측 지평 2주, 단일 rolling cutoff 구조를 사용했다. AA-Forecast는 LOWESS 기반 추세-계절성 분해와 이상치 임계치 ${fmt2(3.08)}를 사용하고, Retrieval은 top-k=1, cosine similarity, recency gap 16을 적용했다.

${dataSummaryTable}

${experimentTable}

## III. 연구 가설
${hypothesesTable}

## IV. 실증 결과
전체 성능 비교에서는 Retrieval이 가장 큰 개선을 만들었다. direct Informer의 MAPE는 ${fmt2(Number(directInformer.mean_fold_mape) * 100)}%였지만, direct Informer+Ret는 ${fmt2(Number(directRetInformer.mean_fold_mape) * 100)}%로 낮아졌다. direct TimeXer는 ${fmt2(Number(directTimexer.mean_fold_mape) * 100)}%, direct TimeXer+Ret는 ${fmt2(Number(directRetTimexer.mean_fold_mape) * 100)}%였다. AA-only 계열은 ${fmt2(Number(aaInformer["AAForecast"].mean_fold_mape) * 100)}% 내외에 머물렀으나, AA+Ret는 ${fmt2(Number(aaInformerRet["AAForecast"].mean_fold_mape) * 100)}% 수준으로 떨어졌다.

${modelComparisonTable}

핵심적으로, direct baseline 대비 Retrieval-only는 대폭의 성능 향상을 보였고, AA-only는 단독으로는 충분하지 않았다. 다만 AA+Ret는 AA-only보다 분명히 좋았고, 일부 backbone에서는 Retrieval-only와 거의 비슷한 수준을 유지했다. 이 결과는 이상치 인지가 Retrieval의 대체물이 아니라 보조 신호임을 시사한다.

## V. 가설 검증
${hypothesisValidationTable}

가설 2와 가설 4는 명확히 지지된다. Retrieval을 도입하면 전체 MAPE가 20%대에서 6%대로 내려가며, AA-only의 한계를 보완한다. 가설 3은 backbone에 따라 부분적으로만 지지되었다. AA+Ret는 AA-only보다 낫지만, retrieval-only가 이미 매우 강한 경우에는 단순한 결합이 항상 더 낫다고는 말할 수 없다. 가설 1은 final_wti에서 블랙스완 지수의 별도 제거 실험이 확인되지 않아 부분 검증으로 남긴다.

## VI. 결론
본 연구는 WTI 주간 예측에서 Retrieval이 핵심적인 개선 요인이라는 점을 확인했다. AA-only는 단독으로는 충분하지 않았지만, Retrieval과 결합하면 실질적인 오차 감소가 발생했다. 가장 낮은 MAPE는 TimeXer+Ret에서 나타났고, AA+Ret Informer도 direct baseline 대비 큰 폭으로 개선되었다. 따라서 충격 구간이 잦은 원유가격 예측에서는 이상치 인지와 외부 메모리 검색을 함께 쓰는 전략이 유효하다.

## 참고문헌
Caldara, D., & Iacoviello, M. (2022). Measuring geopolitical risk. *American Economic Review*, 112(4), 1194-1225.
Cho, K., van Merriënboer, B., Gülçehre, Ç., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*.
Farhangi, A., Bian, J., Huang, A., Xiong, H., Wang, J., & Guo, Z. (2023). AA-Forecast: Anomaly-aware forecast for extreme events. *Data Mining and Knowledge Discovery*, 37(3), 1209-1229.
Jammazi, R., & Aloui, C. (2012). Crude oil price forecasting: Experimental evidence from wavelet decomposition and neural network modeling. *Energy Economics*, 34, 230-241.
Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *AAAI*.
Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y.-X., & Yan, X. (2019). Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. *NeurIPS*.
Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting. *NeurIPS*.
Du, D., Han, T., & Guo, S. (2026). Predicting the future by retrieving the past. *AAAI*.

## 부록 A. 주요 하이퍼파라미터
${appendixTable}
`.trim()

await mkdir(outDir, { recursive: true })
await writeFile(outMarkdown, `${narrative}\n`, "utf-8")
const hwpx = await markdownToHwpx(narrative)
await writeFile(outHwpx, Buffer.from(hwpx))

console.log(JSON.stringify({ markdown: outMarkdown, hwpx: outHwpx }, null, 2))

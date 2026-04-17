# PaperBanana prompt for Brent architecture diagram

Create a polished academic system diagram for a forecasting experiment. The diagram must be a left-to-right pipeline with five grouped regions: **Entry & Config**, **Runtime Orchestration**, **AAForecast Model Path**, **Retrieval Path**, and **Output Artifacts**.

## Hard constraints
- Base the diagram only on the execution path of `main.py --config /home/sonet/.openclaw/workspace/research/neuralforecast/yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru-ret.yaml`.
- Do not generalize to the whole Brent batch matrix; focus on the single `aaforecast-gru-ret` route.
- Show that the active stage plugin is **AAForecast**, the backbone is **GRU**, and retrieval is a **posthoc blend path**, not the main backbone itself.
- Include the concrete target `Com_BrentCrudeOil`.
- Show historical exogenous features entering the runtime, with STAR features visually separated from non-STAR features.
- Include run-manifest/config materialization under `/home/sonet/.openclaw/workspace/research/neuralforecast/runs/brent/feature_set_aaforecast_brent_aaforecast_gru-ret`.

## Key technical labels to preserve
- `main.py`
- `runtime_support.runner.load_app_config`
- experiment YAML
- AAForecast linked YAML
- retrieval detail YAML
- `AAForecastStagePlugin`
- expanding-window CV
- diff transform
- GRU backbone
- base forecast
- STAR memory bank
- query window
- neighbor search
- posthoc blend
- predictions / metrics / summary artifacts

## Required numeric callouts
- input size: 64
- horizon: 2
- CV windows: 24
- retrieval top-k: 1
- recency gap: 8
- min similarity: 0.000
- blend range: 0.00 to 1.00

## Visual style
- publication-quality, clean, vector-like, no 3D effects
- blue for orchestration, purple for model internals, orange for retrieval, green for artifacts, gray for config inputs
- rounded subsystem boxes, directional arrows, concise labels
- no figure caption text inside the image body
- keep the content faithful to software architecture rather than abstract ML storytelling

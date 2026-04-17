# Brent AAForecast PaperBanana workflow

This repo now includes a local PaperBanana-based workflow for drawing the **single-run Brent AAForecast GRU + retrieval architecture** rooted at:

```bash
./.venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru-ret.yaml
```

## What is set up

- `tools/PaperBanana/` — local checkout of `dwzhu-pku/PaperBanana`
- `tools/PaperBanana/.venv/` — local virtualenv with PaperBanana dependencies installed
- `scripts/paperbanana_brent_architecture.py` — workflow entrypoint
- `docs/paperbanana/brent_aaforecast_gru_ret/` — generated architecture brief bundle

## 1) Generate / refresh the architecture brief bundle

```bash
python3 scripts/paperbanana_brent_architecture.py bundle
```

Outputs:

- `docs/paperbanana/brent_aaforecast_gru_ret/methodology.md`
- `docs/paperbanana/brent_aaforecast_gru_ret/caption.txt`
- `docs/paperbanana/brent_aaforecast_gru_ret/paperbanana_prompt.md`
- `docs/paperbanana/brent_aaforecast_gru_ret/bundle.json`

## 2) Configure PaperBanana for Gemini

If you want PaperBanana to read the API key from the environment:

```bash
export GOOGLE_API_KEY=...
python3 scripts/paperbanana_brent_architecture.py configure-paperbanana
```

If you explicitly want the current key written into `tools/PaperBanana/configs/model_config.yaml`:

```bash
python3 scripts/paperbanana_brent_architecture.py configure-paperbanana --embed-api-key
```

## 3) Gemini CLI prompt refinement

Gemini CLI is available as `gemini`. The workflow can use it to refine the PaperBanana prompt:

```bash
python3 scripts/paperbanana_brent_architecture.py refine-with-gemini
```

If Gemini CLI has not been authenticated yet, the script exits with a clear message telling you to run `gemini` once interactively first.

## 4) Run PaperBanana

```bash
export GOOGLE_API_KEY=...
python3 scripts/paperbanana_brent_architecture.py run-paperbanana
```

Default behavior:

- bundle source: `docs/paperbanana/brent_aaforecast_gru_ret/`
- mode: `demo_planner_critic`
- retrieval setting: `none`
- output dir: `artifacts/paperbanana_outputs/brent_aaforecast_gru_ret/`

Useful options:

```bash
python3 scripts/paperbanana_brent_architecture.py run-paperbanana \
  --exp-mode demo_full \
  --num-candidates 6 \
  --max-critic-rounds 3
```

## Current caveat

At implementation time, `gemini -p ...` prompted for first-run interactive authentication in this environment, so prompt refinement was prepared and validated for graceful failure, but not completed end-to-end yet.

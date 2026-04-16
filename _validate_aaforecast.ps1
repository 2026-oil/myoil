$root = "\\wsl.localhost\Ubuntu\home\sonet\.openclaw\workspace\research\neuralforecast"
$configs = @(
  "yaml/experiment/feature_set_aaforecast_brent/baseline.yaml",
  "yaml/experiment/feature_set_aaforecast_brent/baseline-ret.yaml",
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-informer-ret.yaml",
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru-ret.yaml",
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-patchtst-ret.yaml",
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-timexer-ret.yaml",
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-informer.yaml",
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru.yaml",
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-patchtst.yaml",
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-timexer.yaml"
)
$log = Join-Path $root ".validate_brent_aaforecast.log"
"" | Out-File -FilePath $log -Encoding utf8
foreach ($c in $configs) {
  "=== $c ===" | Tee-Object -FilePath $log -Append
  Push-Location $root
  try {
    & uv run python main.py --validate-only --config $c 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  } finally {
    Pop-Location
  }
}
"ALL_OK" | Tee-Object -FilePath $log -Append

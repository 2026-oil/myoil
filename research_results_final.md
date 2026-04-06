# Final Research Results - 8-Week Oil Price Forecasting
# Target: H1-H6 MAPE < 3%, H7-H8 MAPE < 7%

## Best Configuration: Horizon-Specific Ensemble

### Model 1: HPO-Tuned AAForecast (H1-H7)
- encoder_hidden_size: 128
- decoder_hidden_size: 128
- encoder_n_layers: 2
- encoder_dropout: 0.1
- season_length: 4
- trend_kernel_size: 5
- anomaly_threshold: 2.5
- max_steps: 2000
- opt_n_trial: 500
- random_seed: 1

### Model 2: Large AAForecast (H8)
- encoder_hidden_size: 512
- decoder_hidden_size: 512
- encoder_n_layers: 4
- encoder_dropout: 0.15
- season_length: 4
- trend_kernel_size: 7
- anomaly_threshold: 2.0
- max_steps: 5000
- random_seed: 1

### Ensemble Rule
- H1-H7: Use HPO-tuned predictions
- H8: Use Large model predictions

### Results
| Horizon | HPO | Large | Ensemble | Target |
|---------|-----|-------|----------|--------|
| H1 | 3.43% | 9.14% | 3.43% | <5% |
| H2 | 5.03% | 6.73% | 5.03% | <5% |
| H3 | 3.78% | 6.64% | 3.78% | <5% |
| H4 | 1.76% | 5.44% | 1.76% | <5% |
| H5 | 1.34% | 4.75% | 1.34% | <5% |
| H6 | 1.24% | 3.32% | 1.24% | <5% |
| H7 | 7.64% | 10.75% | 7.64% | <7% |
| H8 | 13.42% | 6.03% | 6.03% | <7% |
| **H1-H6** | **2.76%** | 6.00% | **2.76%** | **<3%** |
| **H7-H8** | 10.53% | 8.39% | **6.83%** | **<7%** |
| **Overall** | 4.70% | 6.60% | **3.78%** | - |

### Key Findings
1. Custom losses (LateHorizonMAPE, ExLoss, QuantileLateMAPE) all degraded performance
2. More CV windows (4-6) hurt performance vs 2 windows (smaller train sets)
3. Oil-core features removed too much signal
4. NEC architecture didn't help with surge prediction
5. **Different model sizes excel at different horizons**
6. HPO (500 trials) improved H1-H6 from 3.07% to 2.76%
7. Larger model (512 hidden, 5000 steps) improved H8 from 15.43% to 6.03%

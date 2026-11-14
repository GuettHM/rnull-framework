### MAC activity (independent null rate p on weights and inputs)

| p | MACs | Active (mean ± sd) | Skipped, measured | Skipped, 1 − (1 − p)² |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 1,000,000 | 810,047 ± 34 | 19.00% | 19.00% |
| 0.50 | 1,000,000 | 250,204 ± 176 | 74.98% | 75.00% |
| 0.70 | 1,000,000 | 89,970 ± 322 | 91.00% | 91.00% |
| 0.90 | 1,000,000 | 10,011 ± 110 | 99.00% | 99.00% |

### Drift of frozen weights under AdamW (MSE against the consolidated values)

| lr | weight decay | Zero gradient | Masked update | Null gradient | max abs diff (masked vs null) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 0.01 | 3.006e-05 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 0.001 | 0.02 | 1.042e-04 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| 0.01 | 0.1 | 1.413e-01 | 0.0e+00 | 0.0e+00 | 0.0e+00 |

### Output scale of a 64→32 linear layer (std relative to no missing inputs)

| Missing inputs | normalize=False | normalize="mean" | normalize="variance" |
| ---: | ---: | ---: | ---: |
| 0% | 1.00 | 1.00 | 1.00 |
| 30% | 0.84 | 1.20 | 1.00 |
| 60% | 0.63 | 1.60 | 1.00 |
| 90% | 0.32 | 3.47 | 1.00 |

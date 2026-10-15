| spec | ISL/OSL | conc | SM (req -> used) | reps | decode-full tok/s | max running / target | TPOT p50 (ms) | ITL p50 (ms) | TPOT p99 (ms) | client out tok/s | dur (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 1024/2048 | 128 | 12 -> 12 | 1 | 1496 | 128 / 128 | 85.19 | 85.42 | 85.44 | 1411 | 372 |
| off | 1024/2048 | 128 | auto -> 64 | 1 | 1408 | 128 / 128 | 90.67 | 90.85 | 90.91 | 1378 | 380 |
| off | 1024/2048 | 256 | 12 -> 12 | 1 | 2718 | 256 / 256 | 93.65 | 93.85 | 93.87 | 2634 | 398 |
| off | 1024/2048 | 256 | auto -> 64 | 1 | 2481 | 256 / 256 | 102.53 | 102.60 | 102.94 | 2413 | 434 |
| off | 1024/2048 | 512 | 12 -> 12 | 1 | 4892 | 512 / 512 | 103.81 | 104.22 | 104.23 | 4635 | 452 |
| off | 1024/2048 | 512 | auto -> 64 | 1 | 4451 | 512 / 512 | 114.22 | 114.51 | 114.75 | 4239 | 495 |

**SM12 vs auto** (positive tok/s and negative TPOT favour SM12; noise = larger of the two arms' own max-min spread)

| spec | ISL/OSL | conc | Δ decode-full tok/s | Δ TPOT p50 | noise (tok/s / TPOT) | verdict |
|---|---|---|---|---|---|---|
| off | 1024/2048 | 128 | +6.2% | -6.0% | 0.0% / 0.0% | 1 rep: no noise floor |
| off | 1024/2048 | 256 | +9.5% | -8.7% | 0.0% / 0.0% | 1 rep: no noise floor |
| off | 1024/2048 | 512 | +9.9% | -9.1% | 0.0% / 0.0% | 1 rep: no noise floor |

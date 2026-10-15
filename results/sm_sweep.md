| image/attn/spec | ISL/OSL | conc | SM (req -> used) | reps | decode-full tok/s | max running / target | TPOT p50 (ms) | ITL p50 (ms) | TPOT p99 (ms) | client out tok/s | dur (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sgl07c8f729/tp/spec-off | 1024/2048 | 128 | 12 -> 12 | 1 | 1496 | 128 / 128 | 85.19 | 85.42 | 85.44 | 1411 | 372 |
| sgl07c8f729/tp/spec-off | 1024/2048 | 128 | auto -> 64 | 1 | 1408 | 128 / 128 | 90.67 | 90.85 | 90.91 | 1378 | 380 |
| sgl07c8f729/tp/spec-off | 1024/2048 | 256 | 12 -> 12 | 1 | 2718 | 256 / 256 | 93.65 | 93.85 | 93.87 | 2634 | 398 |
| sgl07c8f729/tp/spec-off | 1024/2048 | 256 | auto -> 64 | 1 | 2481 | 256 / 256 | 102.53 | 102.60 | 102.94 | 2413 | 434 |
| sgl07c8f729/tp/spec-off | 1024/2048 | 512 | 12 -> 12 | 1 | 4892 | 512 / 512 | 103.81 | 104.22 | 104.23 | 4635 | 452 |
| sgl07c8f729/tp/spec-off | 1024/2048 | 512 | auto -> 64 | 1 | 4451 | 512 / 512 | 114.22 | 114.51 | 114.75 | 4239 | 495 |
| sgla6a8c560/dp/spec-off | 1024/2048 | 128 | 12 -> 12 | 1 | 1858 | 128 / 128 | 69.13 | 68.51 | 69.33 | 1798 | 292 |
| sgla6a8c560/dp/spec-off | 1024/2048 | 256 | 12 -> 12 | 1 | 3379 | 256 / 256 | 76.27 | 74.99 | 77.05 | 3209 | 327 |
| sgla6a8c560/dp/spec-off | 1024/2048 | 512 | 12 -> 12 | 1 | 6288 | 512 / 512 | 81.82 | 81.05 | 82.45 | 5770 | 363 |
| sgla6a8c560/tp/spec-off | 1024/2048 | 128 | 12 -> 12 | 1 | 1513 | 128 / 128 | 84.36 | 84.54 | 84.59 | 1472 | 356 |
| sgla6a8c560/tp/spec-off | 1024/2048 | 256 | 12 -> 12 | 1 | 2748 | 256 / 256 | 92.72 | 92.93 | 93.03 | 2662 | 394 |
| sgla6a8c560/tp/spec-off | 1024/2048 | 512 | 12 -> 12 | 1 | 4951 | 512 / 512 | 102.70 | 103.02 | 103.06 | 4707 | 446 |

**SM12 vs auto** (positive tok/s and negative TPOT favour SM12; noise = larger of the two arms' own max-min spread)

| image/attn/spec | ISL/OSL | conc | Δ decode-full tok/s | Δ TPOT p50 | noise (tok/s / TPOT) | verdict |
|---|---|---|---|---|---|---|
| sgl07c8f729/tp/spec-off | 1024/2048 | 128 | +6.2% | -6.0% | 0.0% / 0.0% | 1 rep: no noise floor |
| sgl07c8f729/tp/spec-off | 1024/2048 | 256 | +9.5% | -8.7% | 0.0% / 0.0% | 1 rep: no noise floor |
| sgl07c8f729/tp/spec-off | 1024/2048 | 512 | +9.9% | -9.1% | 0.0% / 0.0% | 1 rep: no noise floor |

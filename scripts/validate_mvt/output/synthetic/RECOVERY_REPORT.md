# Synthetic Recovery Study

FRED pulses with known rise / decay / FWHM swept across rise ∈ [5 ms, 10 s].

Peak SNR per bin = 30 (bright, well-detectable).

| Algorithm | Expected metric | Geo-mean ratio | log-scatter (dex) | upper-limits | N used |
|---|---|---|---|---|---|
| haar | rise | 0.47 | 0.08 | 0 | 14 |
| cwt | rise | 2.47 | 0.01 | 0 | 14 |
| mepsa | fwhm | 1.07 | 0.17 | 0 | 14 |

**Interpretation:** an unbiased algorithm has geometric-mean ratio ≈ 1.0 and log-scatter ≲ 0.15 dex (35%).  Persistent ratio ≠ 1 indicates a systematic bias in the algorithm implementation.

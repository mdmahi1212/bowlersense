# BowlSense model evaluation

Generated 2026-11-20 14:36.

## Classification

| Task | Data | n | Accuracy | Balanced acc. | Macro P | Macro R | Macro F1 | Weighted F1 | Baseline | Coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| Pace vs spin | real | 84 | 63.1% | 62.6% | 0.628 | 0.626 | 0.627 | 0.630 | 53.6% | 100% |
| Bowling arm (elbow at learned release) | real | 84 | 94.0% | 72.2% | 0.969 | 0.722 | 0.792 | 0.930 | 89.3% | 100% |
| Bowling arm (peak wrist speed) | real | 84 | 61.9% | 54.2% | 0.517 | 0.542 | 0.475 | 0.691 | 89.3% | 100% |
| Camera view | real | 278 | 93.2% | 76.5% | 0.953 | 0.765 | 0.803 | 0.920 | 46.8% | 100% |
| Legality (elbow extension) | synthetic | 80 | 93.8% | 93.8% | 0.940 | 0.938 | 0.937 | 0.937 | 50.0% | 98% |
| Faults (all 10 pooled) | synthetic | 156 | 93.6% | 93.7% | 0.935 | 0.937 | 0.935 | 0.936 | 55.1% | 100% |
| Fault: Front knee collapse | synthetic | 16 | 75.0% | 75.0% | 0.833 | 0.750 | 0.733 | 0.733 | 50.0% | 100% |
| Fault: Over-long delivery stride | synthetic | 16 | 100.0% | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 | 50.0% | 100% |
| Fault: Short delivery stride | synthetic | 16 | 100.0% | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 | 62.5% | 100% |
| Fault: Slowing down in the run-up | synthetic | 16 | 100.0% | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 | 50.0% | 100% |
| Fault: Low release point | synthetic | 12 | 100.0% | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 | 66.7% | 100% |
| Fault: Cut-off follow-through | synthetic | 16 | 87.5% | 87.5% | 0.900 | 0.875 | 0.873 | 0.873 | 50.0% | 100% |
| Fault: Weak front arm | synthetic | 16 | 87.5% | 90.0% | 0.875 | 0.900 | 0.873 | 0.877 | 62.5% | 100% |
| Fault: Out-of-sequence rotation | synthetic | 16 | 87.5% | 90.0% | 0.875 | 0.900 | 0.873 | 0.877 | 62.5% | 100% |
| Fault: Excessive side bend | synthetic | 16 | 100.0% | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 | 50.0% | 100% |
| Fault: Head falling away | synthetic | 16 | 100.0% | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 | 50.0% | 100% |

### Confusion matrices (rows = true, columns = predicted)

**Pace vs spin**

| true \ predicted | fast | spin |
|---|---|---|
| fast | 31 | 14 |
| spin | 17 | 22 |

**Bowling arm (elbow at learned release)**

| true \ predicted | right | left |
|---|---|---|
| right | 75 | 0 |
| left | 5 | 4 |

**Bowling arm (peak wrist speed)**

| true \ predicted | right | left |
|---|---|---|
| right | 48 | 27 |
| left | 5 | 4 |

**Camera view**

| true \ predicted | side | front | behind |
|---|---|---|---|
| side | 7 | 7 | 7 |
| front | 0 | 126 | 1 |
| behind | 0 | 4 | 126 |

**Legality (elbow extension)**

| true \ predicted | legal | not_legal |
|---|---|---|
| legal | 39 | 1 |
| not_legal | 4 | 36 |

**Faults (all 10 pooled)**

| true \ predicted | absent | present |
|---|---|---|
| absent | 80 | 6 |
| present | 4 | 66 |

**Fault: Front knee collapse**

| true \ predicted | absent | present |
|---|---|---|
| absent | 8 | 0 |
| present | 4 | 4 |

**Fault: Over-long delivery stride**

| true \ predicted | absent | present |
|---|---|---|
| absent | 8 | 0 |
| present | 0 | 8 |

**Fault: Short delivery stride**

| true \ predicted | absent | present |
|---|---|---|
| absent | 10 | 0 |
| present | 0 | 6 |

**Fault: Slowing down in the run-up**

| true \ predicted | absent | present |
|---|---|---|
| absent | 8 | 0 |
| present | 0 | 8 |

**Fault: Low release point**

| true \ predicted | absent | present |
|---|---|---|
| absent | 8 | 0 |
| present | 0 | 4 |

**Fault: Cut-off follow-through**

| true \ predicted | absent | present |
|---|---|---|
| absent | 6 | 2 |
| present | 0 | 8 |

**Fault: Weak front arm**

| true \ predicted | absent | present |
|---|---|---|
| absent | 8 | 2 |
| present | 0 | 6 |

**Fault: Out-of-sequence rotation**

| true \ predicted | absent | present |
|---|---|---|
| absent | 8 | 2 |
| present | 0 | 6 |

**Fault: Excessive side bend**

| true \ predicted | absent | present |
|---|---|---|
| absent | 8 | 0 |
| present | 0 | 8 |

**Fault: Head falling away**

| true \ predicted | absent | present |
|---|---|---|
| absent | 8 | 0 |
| present | 0 | 8 |

## Measurement accuracy

| Measurement | Data | n | MAE | RMSE | R² | Bias | Max error | Unit |
|---|---|---|---|---|---|---|---|---|
| Elbow extension | synthetic | 30 | 2.83 | 4.02 | 0.753 | -0.74 | 13.39 | deg |
| Front-knee flexion from FFC to release | synthetic | 30 | 6.60 | 19.54 | -2.206 | +1.89 | 104.64 | deg |
| Delivery stride length | synthetic | 30 | 0.95 | 3.00 | 0.918 | -0.23 | 16.10 | % of height |
| Release height | synthetic | 30 | 3.87 | 14.64 | -6.299 | -3.85 | 79.72 | % of height |
| Run-up speed loss | synthetic | 30 | 3.98 | 12.00 | -1.009 | +3.93 | 64.71 | % |
| Follow-through | synthetic | 30 | 9.26 | 13.21 | 0.539 | -5.48 | 56.82 | % of stride |
| Front-arm height at FFC | synthetic | 30 | 1.44 | 3.32 | 0.963 | -0.73 | 16.80 | % of height |
| Shoulder counter-rotation | synthetic | 30 | 1.69 | 2.01 | 0.969 | -1.11 | 4.11 | deg |
| Hip-shoulder separation at FFC | synthetic | 30 | 10.26 | 11.46 | -0.806 | -9.21 | 20.54 | deg |
| Shoulder alignment at BFC | synthetic | 30 | 1.72 | 1.94 | -109.536 | -1.58 | 3.83 | deg |
| Hip alignment at BFC | synthetic | 30 | 2.89 | 3.33 | -5801.242 | -2.76 | 5.87 | deg |
| Trunk side-bend at release | synthetic | 30 | 3.09 | 3.61 | 0.861 | -3.09 | 7.44 | deg |
| Head offset from front foot | synthetic | 30 | 2.42 | 2.81 | 0.782 | -2.42 | 5.60 | % of height |

## Other scores

| Task | Metric | Value | n | Data |
|---|---|---|---|---|
| Release timing (learned BiLSTM) | within +-1 frame | 75.3% | 255 | real |
| Release timing (learned BiLSTM) | within +-2 frame | 85.9% | 255 | real |
| Release timing (learned BiLSTM) | within +-5 frame | 96.5% | 255 | real |
| Release timing (learned BiLSTM) | MAE (ms) | 42.7 | 255 | real |
| Release timing (rules) | within +-1 frame | 48.2% | 255 | real |
| Release timing (rules) | within +-2 frame | 51.8% | 255 | real |
| Release timing (rules) | within +-5 frame | 55.3% | 255 | real |
| Release timing (rules) | MAE (ms) | 162.2 | 255 | real |
| Bowler picked correctly | learned selector | 88.2% | 85 | real |
| Bowler picked correctly | old rule | 71.8% | 85 | real |
| Bowler picked correctly | tracker ceiling | 95.3% | 85 | real |

## Notes

- Synthetic = bowlers from the built-in generator with exactly known answers (no real labels exist for legality, faults or joint angles). Real = the labelled match-footage dataset's held-out test videos.
- Legality is 2D video screening, not the ICC 3D test; injury risk is a rule-based estimate, not medical advice.

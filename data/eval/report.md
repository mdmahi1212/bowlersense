# BowlSense evaluation

| Metric                                       | Value                 | n   | Notes                                      |
|----------------------------------------------|-----------------------|-----|--------------------------------------------|
| Phase error: bfc (frames)                    | 0.79 mean / 2 max     | 48  | noise 2 px; 120/240 fps; side+front; L+R   |
| Phase error: ffc (frames)                    | 1.25 mean / 3 max     | 48  | noise 2 px; 120/240 fps; side+front; L+R   |
| Phase error: arm_horizontal (frames)         | 0.25 mean / 1 max     | 48  | noise 2 px; 120/240 fps; side+front; L+R   |
| Phase error: release (frames)                | 2.38 mean / 7 max     | 48  | noise 2 px; 120/240 fps; side+front; L+R   |
| Phase error with 10% dropout (frames)        | 1.07 mean / 7 max     | 192 | all four events                            |
| Bowling-arm detection (synthetic)            | 100.0%                | 48  | wrist speed                                |
| Elbow extension (legality) MAE (deg)         | 1.95 (max 6.1)        | 18  | side view                                  |
| Counter-rotation MAE (deg)                   | 1.73 (max 4.3)        | 12  | front view                                 |
| Front-knee collapse MAE (deg)                | 6.03 (max 9.8)        | 12  | side view                                  |
| Stride length MAE (% height)                 | 0.50 (max 0.9)        | 12  | side view                                  |
| Ball speed error (%)                         | 1.85 (max 3.8)        | 12  | calibrated; side-on + 11 deg; 0.5 px noise |
| Release within +-1 frame (learned BiLSTM)    | 75.3% (MAE 43 ms)     | 255 | real test split, 3 s crops                 |
| Release within +-1 frame (rule-based)        | 48.2% (MAE 162 ms)    | 255 | real test split, 3 s crops                 |
| Bowling arm, real (elbow at learned release) | 84.5%                 | 278 | phases.use_model on                        |
| Bowling arm, real (peak wrist speed)         | 59.0%                 | 278 | default rules                              |
| Camera view (real)                           | 93.2%                 | 279 | all clips                                  |
| Pace vs spin (real)                          | 63.1% acc / 62.6% bal | 84  | BiGRU, end to end                          |
| Pose PCK@0.2                                 | n/a                   | 0   | no labels in data/eval/pose_labels.json    |

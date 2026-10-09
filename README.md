# Unsupervised Network Anomaly Detection

Detectors are trained on **benign-only** flows, so no attack labels are needed to train. They are
benchmarked against four attack families: SYN-flood DoS, port scan, SSH brute force, and data exfiltration.
40% of each attack family is a "low-and-slow" variant that overlaps benign traffic.

Models: Mahalanobis baseline, Isolation Forest, PyTorch autoencoder, and a rank-averaged IF+AE ensemble.
Thresholds are set on a held-out **benign validation set at a 1% target false-positive rate**, and
labels are never used to tune them.

Run: `python3 run.py` (about 1 min on a laptop CPU). It writes `results.json` and `curves.png`.

## Setup
200k benign train / 50k benign val / 100k test (5% anomalous). 14 flow features (bytes, packets, SYN ratio,
distinct ports per 60 s, connection rate, inter-arrival time, failed-auth ratio, ...), log-scaled and standardized.

## Results (test set, threshold at 1% FPR)
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| Mahalanobis (baseline) | 0.999 | 0.984 | 0.826 | 0.976 | 0.895 | 1.08% |
| Isolation Forest | 0.979 | 0.707 | 0.731 | 0.519 | 0.607 | 1.01% |
| Autoencoder | 0.993 | 0.940 | 0.833 | 0.895 | 0.863 | 0.94% |
| Ensemble (IF+AE) | 0.995 | 0.936 | 0.830 | 0.882 | 0.856 | 0.95% |

Per-attack recall (autoencoder): DoS 89.8%, scan 90.2%, brute force 86.5%, exfiltration 91.5%.
Isolation Forest struggles on DoS (28.8%) and brute force (15.8%), where the signal is a combination of features rather than one extreme value.

## Honest caveats
- **Data is synthetic.** The simulator's attacks are mostly shifts in log-feature space, which is why a simple
  Mahalanobis baseline wins. Real traffic is messier, so absolute numbers will drop on real data.
  The next step is to run the same pipeline on CIC-IDS2017 or UNSW-NB15 and report those numbers instead.
- The autoencoder did not beat the baseline on this data. Don't claim it did.

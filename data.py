"""Synthetic network-flow generator (benign + 4 attack families).

Features are NetFlow/CICFlowMeter-style per-flow and per-source-window stats.
Attack families include a 'stealthy' variant that overlaps benign traffic so the
task is not trivially separable.
"""
import numpy as np
import pandas as pd

FEATURES = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "dst_port_bucket",
            "syn_ratio", "uniq_dst_ports_60s", "conn_rate_60s", "avg_pkt_size",
            "iat_mean", "iat_std", "failed_auth_ratio", "bytes_ratio"]
ATTACKS = ["dos_syn_flood", "port_scan", "brute_force", "exfiltration"]


def _benign(n, rng):
    kind = rng.choice(4, n, p=[.55, .2, .1, .15])  # web, dns, ssh, bulk
    d = np.empty((n, len(FEATURES)))
    web, dns, ssh, bulk = (kind == i for i in range(4))
    ln = lambda m, s, k: rng.lognormal(m, s, k)
    dur = np.where(web, ln(-0.5, 1.0, n), np.where(dns, ln(-5, .6, n), np.where(ssh, ln(3, 1.2, n), ln(2.5, .9, n))))
    sb = np.where(web, ln(6.5, 1.0, n), np.where(dns, ln(4.2, .3, n), np.where(ssh, ln(7, 1.2, n), ln(9.5, 1.2, n))))
    db = np.where(web, ln(8.5, 1.4, n), np.where(dns, ln(5, .4, n), np.where(ssh, ln(7.2, 1.2, n), ln(8, 1.3, n))))
    sp = np.maximum(1, sb / rng.normal(450, 120, n).clip(60, 1400))
    dp = np.maximum(1, db / rng.normal(700, 200, n).clip(60, 1460))
    port = np.where(web, 1, np.where(dns, 0, np.where(ssh, 2, 3)))
    d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5] = dur, sb, db, sp, dp, port
    d[:, 6] = rng.beta(1.2, 12, n) + .02                       # syn ratio
    d[:, 7] = 1 + rng.poisson(np.where(web, 3, 1.2), n)       # distinct ports
    d[:, 8] = ln(1.0, .8, n)                                   # conn rate
    d[:, 9] = (sb + db) / (sp + dp)
    d[:, 10] = dur / (sp + dp) * rng.lognormal(0, .3, n)
    d[:, 11] = d[:, 10] * rng.uniform(.5, 2.0, n)
    d[:, 12] = np.where(ssh, rng.beta(1, 25, n), rng.beta(1, 200, n))
    d[:, 13] = sb / (db + 1)
    return d


def _attack(kind, n, rng, stealth):
    d = _benign(n, rng)          # start from benign, then perturb → realistic overlap
    s = stealth                  # boolean mask: 1 = low-and-slow variant
    k = lambda a, b: np.where(s, a, b)
    if kind == "dos_syn_flood":
        d[:, 6] = rng.beta(*(8, 3), n) * k(.55, 1) + .05
        d[:, 8] = rng.lognormal(k(2.2, 4.0), .5, n)
        d[:, 2] = rng.lognormal(3, 1, n); d[:, 4] = 1
        d[:, 0] = rng.lognormal(-4, .8, n); d[:, 3] = rng.integers(1, 4, n)
    elif kind == "port_scan":
        d[:, 7] = rng.poisson(k(14, 60), n) + 3
        d[:, 1] = rng.lognormal(3.8, .3, n); d[:, 2] = rng.lognormal(k(4, 2), 1, n)
        d[:, 3] = 1 + rng.poisson(1, n); d[:, 4] = rng.poisson(k(1.5, .3), n)
        d[:, 8] = rng.lognormal(k(1.6, 3), .6, n); d[:, 6] = rng.beta(4, 4, n)
    elif kind == "brute_force":
        d[:, 5] = 2; d[:, 12] = rng.beta(k(3, 8), k(4, 2), n)
        d[:, 8] = rng.lognormal(k(1.4, 2.3), .5, n); d[:, 0] = rng.lognormal(k(1.8, 0.2), .6, n)
    elif kind == "exfiltration":
        d[:, 1] = rng.lognormal(k(10.0, 12.5), .7, n); d[:, 2] = rng.lognormal(5, .8, n)
        d[:, 13] = d[:, 1] / (d[:, 2] + 1); d[:, 5] = rng.choice([0, 1, 3], n)
        d[:, 3] = d[:, 1] / rng.normal(900, 200, n).clip(100, 1460)
        d[:, 0] = rng.lognormal(k(4.0, 3), .7, n)
    d[:, 9] = (d[:, 1] + d[:, 2]) / (d[:, 3] + d[:, 4] + 1e-9)
    return d


def make_dataset(n_train=200_000, n_val=50_000, n_test=100_000, contamination=0.05, seed=7):
    """Train/val are benign only (realistic: attacks are rare & unlabeled). Test has labels."""
    rng = np.random.default_rng(seed)
    X_tr, X_va = _benign(n_train, rng), _benign(n_val, rng)
    n_att = int(n_test * contamination)
    parts, y, atk = [_benign(n_test - n_att, rng)], [np.zeros(n_test - n_att)], ["benign"] * (n_test - n_att)
    per = n_att // len(ATTACKS)
    for a in ATTACKS:
        stealth = rng.random(per) < 0.4          # 40% of attacks are low-and-slow
        parts.append(_attack(a, per, rng, stealth)); y.append(np.ones(per)); atk += [a] * per
    X_te, y_te, atk = np.vstack(parts), np.concatenate(y), np.array(atk)
    p = rng.permutation(len(y_te))
    f = lambda X: pd.DataFrame(X, columns=FEATURES)
    return f(X_tr), f(X_va), f(X_te).iloc[p].reset_index(drop=True), y_te[p], atk[p]

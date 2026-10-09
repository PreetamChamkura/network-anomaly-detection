import json, time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_fscore_support,
                             roc_curve, precision_recall_curve, confusion_matrix)
from data import make_dataset, ATTACKS
from models import prep_fit, prep, Mahalanobis, IForest, Autoencoder, Ensemble

TARGET_FPR = 0.01
Xtr, Xva, Xte, y, atk = make_dataset()
sc = prep_fit(Xtr); Ztr, Zva, Zte = prep(sc, Xtr), prep(sc, Xva), prep(sc, Xte)
print(f"train={len(Ztr):,} (benign)  val={len(Zva):,} (benign)  test={len(Zte):,}  anomalies={int(y.sum()):,} ({y.mean():.1%})")

maha = Mahalanobis().fit(Ztr); iff = IForest().fit(Ztr); ae = Autoencoder(Ztr.shape[1]).fit(Ztr)
ens = Ensemble([iff, ae]).fit_ref(Zva)
models = [maha, iff, ae, ens]

res, curves = {}, {}
for m in models:
    thr = np.quantile(m.score(Zva), 1 - TARGET_FPR)        # threshold from benign validation only
    s = m.score(Zte); pred = s > thr
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    t0 = time.perf_counter(); m.score(Zte); dt = time.perf_counter() - t0
    res[m.name] = dict(roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
                       precision=p, recall=r, f1=f1, fpr=fp / (fp + tn),
                       flows_per_sec=len(Zte) / dt,
                       per_attack_recall={a: float(pred[atk == a].mean()) for a in ATTACKS})
    curves[m.name] = (s,)
    print(f"{m.name:26s} AUC={res[m.name]['roc_auc']:.3f} PR-AUC={res[m.name]['pr_auc']:.3f} "
          f"P={p:.3f} R={r:.3f} F1={f1:.3f} FPR={res[m.name]['fpr']:.4f} {res[m.name]['flows_per_sec']:,.0f} flows/s")
    print("   recall by attack:", {a: round(v, 3) for a, v in res[m.name]["per_attack_recall"].items()})

json.dump(res, open("results.json", "w"), indent=2)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for n, (s,) in curves.items():
    fp, tp, _ = roc_curve(y, s); ax[0].plot(fp, tp, label=f"{n} ({res[n]['roc_auc']:.3f})")
    pr, rc, _ = precision_recall_curve(y, s); ax[1].plot(rc, pr, label=f"{n} ({res[n]['pr_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(xlabel="FPR", ylabel="TPR", title="ROC"); ax[0].legend(fontsize=8)
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("curves.png", dpi=130)

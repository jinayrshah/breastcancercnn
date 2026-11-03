"""Evaluation metrics (pure NumPy) and the plots shown on the Performance page."""
import os
import numpy as np


def roc(y, p):
    order = np.argsort(-p)
    y = y[order]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    tpr = np.r_[0, tps / max(y.sum(), 1)]
    fpr = np.r_[0, fps / max((1 - y).sum(), 1)]
    return fpr, tpr


def evaluate(y, p, thr=0.5):
    y = np.asarray(y).astype(int); pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    fpr, tpr = roc(y, np.asarray(p))
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    step = max(1, len(fpr) // 200)
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "specificity": spec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9), "auc": auc,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "roc": {"fpr": fpr[::step].round(4).tolist(), "tpr": tpr[::step].round(4).tolist()}}


def save_plots(y, p, hist, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    ep = range(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, hist["acc"], "o-", c="#7b2d8b", label="train"); ax[0].plot(ep, hist["val_acc"], "o-", c="#d9467d", label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, hist["loss"], "o-", c="#7b2d8b", label="train"); ax[1].plot(ep, hist["val_loss"], "o-", c="#d9467d", label="validation")
    ax[1].set_title("Binary cross-entropy loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{out}/training_curves.png", dpi=140); plt.close(fig)

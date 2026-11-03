"""
Train the NumPy CNN on breast histopathology patches (IDC dataset, Janowczyk & Madabhushi 2016,
the Kaggle "Breast Histopathology Images" set).

Usage:
    python training/train_numpy_cnn.py --data <folder with <patient_id>/<0|1>/*.png> --split split.txt

Key ideas (good to explain in the viva):
  * Patient-level split: all patches of one patient go to the same split, so the test score
    measures performance on *unseen patients* (no data leakage).
  * Augmentation: random flips and 90-degree rotations, because tissue has no "up" direction.
  * Adam optimiser, binary cross-entropy loss, learning-rate decay, early stopping on val AUC.
"""
import argparse, glob, json, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcnet.npcnn import BreastCNN, Adam, bce_grad, IMG
from bcnet.metrics import evaluate, save_plots

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--split", required=True, help="lines: '<patient_id> train|val|test'")
ap.add_argument("--epochs", type=int, default=14)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--out", default="models/idc_cnn.npz")
a = ap.parse_args()

split = dict(l.split() for l in open(a.split) if l.strip())


def load(path):
    im = Image.open(path).convert("RGB")
    if im.size != (50, 50):
        im = im.resize((50, 50))
    return np.asarray(im, np.float32)[1:49, 1:49] / 255.0      # centre-crop 48x48, scale to [0,1]


X = {"train": [], "val": [], "test": []}; Y = {k: [] for k in X}; P = {k: [] for k in X}
files = glob.glob(os.path.join(a.data, "*", "[01]", "*.png"))
for f in files:
    pid = f.split(os.sep)[-3]
    s = split.get(pid)
    if s is None:
        continue
    X[s].append(load(f)); Y[s].append(int(f.split(os.sep)[-2])); P[s].append(f)
for k in X:
    X[k] = np.stack(X[k]); Y[k] = np.array(Y[k], np.float32)
    print(f"{k:5s}: {len(Y[k]):6d} patches  ({int(Y[k].sum())} IDC / {int((1 - Y[k]).sum())} non-IDC)"
          f"  patients={len({split_pid for split_pid, v in split.items() if v == k})}")

rng = np.random.default_rng(0)
model, opt = BreastCNN(seed=42), None
opt = Adam(model, lr=a.lr)
print("parameters:", model.n_params())


def augment(xb):
    if rng.random() < 0.5: xb = xb[:, :, ::-1]
    if rng.random() < 0.5: xb = xb[:, ::-1]
    return np.ascontiguousarray(np.rot90(xb, rng.integers(4), axes=(1, 2)))


hist = {k: [] for k in ["loss", "acc", "val_loss", "val_acc", "val_auc"]}
best, best_ep, patience = -1, 0, 4
for ep in range(1, a.epochs + 1):
    t0 = time.time(); model.train_mode(True)
    idx = rng.permutation(len(Y["train"])); losses, correct = [], 0
    for i in range(0, len(idx), a.batch):
        b = idx[i:i + a.batch]
        xb, yb = augment(X["train"][b]), Y["train"][b]
        lg = model.logits(xb)
        loss, g = bce_grad(lg, yb)
        model.backward(g); opt.step()
        losses.append(loss); correct += ((lg > 0) == (yb > 0.5)).sum()
    pv = model.predict_proba(X["val"])
    vl = float(-np.mean(Y["val"] * np.log(pv + 1e-7) + (1 - Y["val"]) * np.log(1 - pv + 1e-7)))
    m = evaluate(Y["val"], pv)
    hist["loss"].append(float(np.mean(losses))); hist["acc"].append(float(correct / len(idx)))
    hist["val_loss"].append(vl); hist["val_acc"].append(m["accuracy"]); hist["val_auc"].append(m["auc"])
    print(f"epoch {ep:2d}  loss {hist['loss'][-1]:.4f}  acc {hist['acc'][-1]:.4f}  "
          f"val_loss {vl:.4f}  val_acc {m['accuracy']:.4f}  val_auc {m['auc']:.4f}  lr {opt.lr:.1e}  {time.time() - t0:.0f}s", flush=True)
    if m["auc"] > best:
        best, best_ep = m["auc"], ep
        model.save(a.out, dataset="IDC breast histopathology (50x50 patches)", img=IMG)
    elif ep - best_ep >= patience:
        print("early stopping"); break
    if ep % 4 == 0:
        opt.lr *= 0.5                                       # learning-rate decay

# ---- final test evaluation with the best checkpoint
model = BreastCNN.load(a.out)
pt = model.predict_proba(X["test"])
res = evaluate(Y["test"], pt)
res.update(best_epoch=best_ep, n_params=int(model.n_params()),
           counts={k: int(len(Y[k])) for k in X}, history=hist,
           patients={k: sum(1 for v in split.values() if v == k) for k in X})
os.makedirs("models", exist_ok=True)
json.dump(res, open("models/idc_metrics.json", "w"), indent=1)
save_plots(Y["test"], pt, hist, "static/img/plots")
print(json.dumps({k: v for k, v in res.items() if k not in ("history", "roc")}, indent=1))

# ---- copy a few test-set patches (unseen patients) for the demo gallery
os.makedirs("static/samples", exist_ok=True)
order = np.argsort(pt)
picks = list(order[:4]) + list(order[len(order) // 2 - 2:len(order) // 2 + 2]) + list(order[-4:])
meta = []
for j, i in enumerate(picks):
    name = f"test_{j:02d}_{'idc' if Y['test'][i] else 'benign'}.png"
    Image.open(P["test"][i]).convert("RGB").save(os.path.join("static/samples", name))
    meta.append({"file": name, "truth": "Malignant (IDC)" if Y["test"][i] else "Benign (non-IDC)"})
json.dump(meta, open("static/samples/samples.json", "w"), indent=1)

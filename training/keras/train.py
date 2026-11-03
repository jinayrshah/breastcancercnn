"""Step 8-9: train with Adam + BCE, early stopping, LR reduction, class weights; save curves."""
import argparse, json, os
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from config import *
from data import collect, split, make_ds
from model import build_custom_cnn, build_transfer, compile_model

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="custom", choices=["custom", "resnet50", "vgg16"])
ap.add_argument("--mag", default=None, help="40X|100X|200X|400X (default: all)")
ap.add_argument("--epochs", type=int, default=EPOCHS)
a = ap.parse_args()

paths, labels = collect(magnification=a.mag)
(xtr, ytr), (xva, yva), (xte, yte) = split(paths, labels)
os.makedirs("outputs", exist_ok=True); os.makedirs("models", exist_ok=True)
json.dump({"test_x": xte, "test_y": yte}, open("outputs/test_split.json", "w"))
print(f"train {len(xtr)}  val {len(xva)}  test {len(xte)}")

cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(ytr))
model = compile_model(build_custom_cnn() if a.arch == "custom" else build_transfer(a.arch))
model.summary()
cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3),
      tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True)]
h = model.fit(make_ds(xtr, ytr, True), validation_data=make_ds(xva, yva), epochs=a.epochs,
              class_weight={0: cw[0], 1: cw[1]}, callbacks=cb)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, k in enumerate(["accuracy", "loss"]):
    ax[i].plot(h.history[k], label="train"); ax[i].plot(h.history["val_" + k], label="val")
    ax[i].set_title(k.title()); ax[i].set_xlabel("Epoch"); ax[i].legend()
plt.tight_layout(); plt.savefig("outputs/training_curves.png", dpi=150)
print("Saved", MODEL_PATH)

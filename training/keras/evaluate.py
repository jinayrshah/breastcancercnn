"""Step 10: test-set metrics, confusion matrix, ROC curve."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from config import *
from data import make_ds

s = json.load(open("outputs/test_split.json"))
model = tf.keras.models.load_model(MODEL_PATH)
prob = model.predict(make_ds(s["test_x"], s["test_y"])).ravel()
y, pred = np.array(s["test_y"]), (prob >= 0.5).astype(int)
rep = classification_report(y, pred, target_names=CLASSES, digits=4)
print(rep); open("outputs/classification_report.txt", "w").write(rep + f"\nROC-AUC: {roc_auc_score(y, prob):.4f}\n")

cm = confusion_matrix(y, pred)
plt.figure(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
fpr, tpr, _ = roc_curve(y, prob)
plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, prob):.3f}"); plt.plot([0, 1], [0, 1], "--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)

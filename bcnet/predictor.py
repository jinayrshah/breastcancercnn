"""
Prediction service used by the web app.

Two interchangeable model back-ends:
  1. "numpy"  - the from-scratch NumPy CNN in models/idc_cnn.npz (always available, no TensorFlow).
  2. "keras"  - the Keras CNN trained on BreakHis in Colab (models/breast_cnn.keras),
                enabled automatically when TensorFlow is installed and the file exists.

Two analysis modes for the NumPy CNN:
  * Patch mode  - image is a small patch (<= 100 px): resize to 48x48, one prediction + Grad-CAM.
  * Scan mode   - larger image: slide a 50x50 window across it, predict every tile and build a
                  probability map, like a pathologist scanning a slide region by region.
"""
import base64, io, os, time
import numpy as np
from PIL import Image

from .npcnn import BreastCNN, gradcam, IMG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "models", "idc_cnn.npz")
KERAS = os.path.join(ROOT, "models", "breast_cnn.keras")
THRESHOLD = 0.5


def to_b64(arr_or_img, size=None):
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(np.uint8(np.clip(arr_or_img, 0, 255)))
    if size:
        im = im.resize(size, Image.NEAREST if im.width < 64 else Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def jet(v):
    """Map values in [0,1] to an RGB colour ramp (blue -> green -> yellow -> red)."""
    v = np.clip(v, 0, 1)[..., None]
    stops = np.array([[30, 60, 200], [40, 180, 170], [250, 220, 60], [220, 40, 40]], np.float32)
    x = v * 3; i = np.minimum(x.astype(int), 2); f = x - i
    return stops[i[..., 0]] * (1 - f) + stops[i[..., 0] + 1] * f


def overlay(rgb, heat, alpha=0.45):
    h = np.asarray(Image.fromarray(np.uint8(jet(heat) )).resize(rgb.shape[1::-1], Image.BILINEAR), np.float32)
    return rgb * (1 - alpha) + h * alpha


class Predictor:
    def __init__(self):
        self.numpy_model = BreastCNN.load(NPZ) if os.path.exists(NPZ) else None
        self.keras_model, self.keras_error = None, None
        if os.path.exists(KERAS):
            try:
                import tensorflow as tf  # noqa
                self.keras_model = tf.keras.models.load_model(KERAS)
            except Exception as e:  # TensorFlow not installed or incompatible
                self.keras_error = str(e).splitlines()[0][:160]

    def backends(self):
        out = []
        if self.numpy_model is not None:
            out.append({"id": "numpy", "name": "HistoScan CNN (NumPy, IDC)", "ready": True})
        out.append({"id": "keras", "name": "Keras CNN (BreakHis)",
                    "ready": self.keras_model is not None,
                    "note": None if self.keras_model is not None else
                    (self.keras_error or "Put models/breast_cnn.keras here and install TensorFlow")})
        return out

    # ------------------------------------------------------------------ public
    def predict(self, img: Image.Image, backend="numpy"):
        t0 = time.time()
        img = img.convert("RGB")
        if backend == "keras":
            if self.keras_model is None:
                raise RuntimeError("Keras model not available. " + (self.keras_error or "Train it in Colab first."))
            res = self._keras(img)
        elif max(img.size) <= 100:
            res = self._patch(img)
        else:
            res = self._scan(img)
        res["ms"] = round((time.time() - t0) * 1000)
        p = res["prob"]
        res["label"] = "Malignant" if p >= THRESHOLD else "Benign"
        res["confidence"] = p if p >= THRESHOLD else 1 - p
        res["risk"] = "High" if p >= 0.8 else "Moderate" if p >= THRESHOLD else "Low" if p < 0.2 else "Borderline"
        return res

    # ------------------------------------------------------------------ numpy, single patch
    def _prep(self, img):
        small = img.resize((50, 50), Image.BILINEAR)
        arr = np.asarray(small, np.float32)[1:49, 1:49]
        return arr, arr / 255.0

    def _patch(self, img):
        raw, x = self._prep(img)
        prob, cam = gradcam(self.numpy_model, x)
        big = np.asarray(img.resize((240, 240), Image.BILINEAR), np.float32)
        return {
            "mode": "patch", "model": "NumPy CNN", "prob": prob,
            "steps": [
                {"title": "Uploaded image", "img": to_b64(img, (160, 160)), "note": f"{img.width}x{img.height} px, RGB"},
                {"title": "Resized + cropped", "img": to_b64(raw, (160, 160)), "note": "48x48x3 input tensor"},
                {"title": "Normalised", "img": to_b64(x * 255, (160, 160)), "note": "pixel / 255 -> [0, 1]"},
            ],
            "explain_title": "Grad-CAM attention",
            "explain": to_b64(overlay(big, cam)),
            "explain_note": "Red regions pushed the network towards 'malignant'.",
        }

    # ------------------------------------------------------------------ numpy, sliding-window scan
    def _scan(self, img):
        w, h = img.size
        s = min(1.0, 4000 / max(w, h))                       # keep native scale (model was trained on 50x50 tiles at this zoom)
        if s < 1:
            img = img.resize((int(w * s), int(h * s)), Image.BILINEAR); w, h = img.size
        arr = np.asarray(img, np.float32)
        ny, nx = h // 50, w // 50
        tiles, keep = [], []
        for r in range(ny):
            for c in range(nx):
                t = arr[r * 50:(r + 1) * 50, c * 50:(c + 1) * 50]
                tiles.append(t[1:49, 1:49] / 255.0)
                keep.append(t.mean() < 225)                  # skip blank / background tiles
        tiles = np.stack(tiles); keep = np.array(keep)
        probs = np.zeros(len(tiles)); probs[keep] = self.numpy_model.predict_proba(tiles[keep])
        grid = probs.reshape(ny, nx); tissue = keep.reshape(ny, nx)
        tp = probs[keep]
        # slide-level score: mean of the top 10% most suspicious tissue tiles
        k = max(1, int(np.ceil(0.10 * len(tp))))
        prob = float(np.sort(tp)[-k:].mean()) if len(tp) else 0.0
        crop = arr[:ny * 50, :nx * 50]
        heat = np.where(tissue, grid, 0)
        disp_w = 520; disp_h = int(disp_w * crop.shape[0] / crop.shape[1])
        return {
            "mode": "scan", "model": "NumPy CNN", "prob": prob,
            "tiles": {"total": int(len(probs)), "tissue": int(keep.sum()),
                      "suspicious": int((tp >= THRESHOLD).sum()),
                      "pct": round(100 * float((tp >= THRESHOLD).mean()), 1) if len(tp) else 0},
            "steps": [
                {"title": "Uploaded image", "img": to_b64(img, (disp_w // 2, disp_h // 2)), "note": f"{w}x{h} px"},
                {"title": "Tiled into patches", "img": to_b64(self._grid_lines(crop), (disp_w // 2, disp_h // 2)),
                 "note": f"{nx} x {ny} = {nx * ny} tiles of 50x50"},
                {"title": "Tile probabilities", "img": to_b64(np.where(tissue[..., None], jet(grid), 235), (disp_w // 2, disp_h // 2)),
                 "note": f"{int(keep.sum())} tissue tiles scored"},
            ],
            "explain_title": "Malignancy probability map",
            "explain": to_b64(np.asarray(Image.fromarray(np.uint8(self._tissue_overlay(crop, heat, tissue))).resize((disp_w, disp_h)))),
            "explain_note": "Each 50x50 tile is coloured by its predicted probability of IDC. Score = mean of the top 10% tiles.",
        }

    @staticmethod
    def _tissue_overlay(crop, heat, tissue):
        """Colour only tissue pixels by tile probability; leave background as is."""
        h = np.asarray(Image.fromarray(np.uint8(jet(heat))).resize(crop.shape[1::-1], Image.BILINEAR), np.float32)
        m = np.asarray(Image.fromarray(np.uint8(tissue * 255)).resize(crop.shape[1::-1], Image.BILINEAR), np.float32)[..., None] / 255
        m = m * (crop.mean(-1, keepdims=True) < 228)          # skip white glass inside tiles
        return crop * (1 - 0.55 * m) + h * 0.55 * m

    @staticmethod
    def _grid_lines(crop):
        g = crop.copy()
        g[::50] = [255, 255, 255]; g[:, ::50] = [255, 255, 255]
        return g

    # ------------------------------------------------------------------ keras (BreakHis)
    def _keras(self, img):
        import sys
        sys.path.insert(0, os.path.join(ROOT, "training", "keras"))
        from gradcam import gradcam as kgradcam, overlay as koverlay
        rgb = np.asarray(img.resize((224, 224)))
        x = rgb.astype("float32") / 255.0
        prob = float(self.keras_model.predict(x[None], verbose=0)[0, 0])
        try:
            expl = to_b64(koverlay(rgb, kgradcam(self.keras_model, x)))
        except Exception:
            expl = to_b64(rgb)
        return {"mode": "patch", "model": "Keras CNN (BreakHis)", "prob": prob,
                "steps": [{"title": "Uploaded image", "img": to_b64(img, (160, 160)), "note": f"{img.width}x{img.height} px"},
                          {"title": "Resized", "img": to_b64(rgb, (160, 160)), "note": "224x224x3"},
                          {"title": "Normalised", "img": to_b64(rgb, (160, 160)), "note": "pixel / 255"}],
                "explain_title": "Grad-CAM attention", "explain": expl,
                "explain_note": "Red regions pushed the network towards 'malignant'."}

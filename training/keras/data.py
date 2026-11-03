"""Step 1-6: load BreakHis, resize 224x224, normalise [0,1], stratified 70/15/15 split, augmentation."""
import os, glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import *

def collect(data_dir=DATA_DIR, magnification=None):
    paths, labels = [], []
    for label, cls in enumerate(["benign", "malignant"]):
        pattern = os.path.join(data_dir, cls, "**", "*.png")
        for p in glob.glob(pattern, recursive=True):
            if magnification and f"/{magnification}/" not in p.replace("\\", "/"):
                continue
            paths.append(p); labels.append(label)
    if not paths:
        raise FileNotFoundError(f"No images under {data_dir}. Download BreakHis first (see README).")
    return paths, labels

def split(paths, labels):
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te)

def _load(path, label):
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return img, tf.cast(label, tf.float32)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(20 / 360),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.15, value_range=(0, 1)),
], name="augmentation")

def make_ds(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED).map(lambda i, l: (augment(i, training=True), l),
                                              num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

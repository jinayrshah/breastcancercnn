"""Step 7: custom CNN (Conv2D+ReLU+MaxPool blocks -> Dense256 -> Dense128 -> Sigmoid)
plus an optional transfer-learning baseline (VGG16 / ResNet50) for comparison."""
import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE

def build_custom_cnn():
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = inp
    for i, f in enumerate([32, 64, 128, 256]):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i+1}_a")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i+1}_b")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(layers.MaxPooling2D()(x))
    x = layers.Dropout(0.5)(layers.Dense(256, activation="relu")(x))
    x = layers.Dropout(0.3)(layers.Dense(128, activation="relu")(x))
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name="BreastCNN")

def build_transfer(backbone="resnet50"):
    base_cls = {"resnet50": tf.keras.applications.ResNet50, "vgg16": tf.keras.applications.VGG16}[backbone]
    base = base_cls(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = base(inp * 255.0, training=False)  # inputs arrive in [0,1]
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(layers.Dense(256, activation="relu")(x))
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name=f"Transfer_{backbone}")

def compile_model(m, lr=1e-4):
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")])
    return m

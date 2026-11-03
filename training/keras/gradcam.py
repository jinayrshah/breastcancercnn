"""Grad-CAM explainability: highlights tissue regions that drove the prediction."""
import numpy as np, cv2, tensorflow as tf

def last_conv(model):
    for l in reversed(model.layers):
        if isinstance(l, tf.keras.layers.Conv2D): return l.name
    raise ValueError("No Conv2D layer (use the custom CNN for Grad-CAM)")

def gradcam(model, img01):
    gm = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv(model)).output, model.output])
    with tf.GradientTape() as t:
        conv, out = gm(img01[None]); score = out[:, 0]
    g = t.gradient(score, conv)[0]; w = tf.reduce_mean(g, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * w, -1)).numpy()
    return cam / (cam.max() + 1e-8)

def overlay(rgb_uint8, cam, alpha=0.4):
    h = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, rgb_uint8.shape[1::-1])), cv2.COLORMAP_JET)
    return cv2.addWeighted(rgb_uint8, 1 - alpha, cv2.cvtColor(h, cv2.COLOR_BGR2RGB), alpha, 0)

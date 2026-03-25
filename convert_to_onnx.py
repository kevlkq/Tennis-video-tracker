"""
Build TrackNet from ArtLabss architecture, load weights, export to ONNX.
Run once: python convert_to_onnx.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ArtLabss"))

import tf2onnx
import tensorflow as tf
from Models.tracknet import trackNet

WEIGHTS = "ArtLabss/WeightsTracknet/model.h5"
OUTPUT  = "models/tracknet.onnx"

print("[1/3] Building TrackNet architecture (360x640)...")
model = trackNet(256, input_height=360, input_width=640)

print("[2/3] Loading weights...")
model.load_weights(WEIGHTS)

print("[3/3] Exporting to ONNX...")
input_sig = [tf.TensorSpec([None] + list(model.input_shape[1:]), tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_sig, opset=13)

os.makedirs("models", exist_ok=True)
with open(OUTPUT, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"[DONE] Saved to {OUTPUT}")

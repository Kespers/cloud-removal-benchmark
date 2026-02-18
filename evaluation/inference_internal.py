
import sys
import os
import argparse
import numpy as np
import rasterio
import keras.backend as K
import tensorflow as tf

# Add library path
sys.path.append('/app/dsen2-cr-main/Code')
from dsen2cr_network import DSen2CR_model

# Constants
SCALE = 2000
MAX_VAL_SAR = 2
CLIP_MIN = [[-25.0, -32.5], [0]*13, [0]*13]
CLIP_MAX = [[0, 0], [10000]*13, [10000]*13]

def preprocess_sar(image_data):
    # image_data: (C, H, W)
    # We expect 2 channels for SAR (VV, VH) usually? 
    # The file checks for 2 channels.
    c_min = CLIP_MIN[0]
    c_max = CLIP_MAX[0]
    new_img = np.zeros_like(image_data, dtype=np.float32)
    for c in range(min(image_data.shape[0], len(c_min))):
         min_v = c_min[c]
         max_v = c_max[c]
         val = np.clip(image_data[c], min_v, max_v)
         val -= min_v
         if (max_v - min_v) != 0:
            val = MAX_VAL_SAR * (val / (max_v - min_v))
         new_img[c] = val
    return new_img

def preprocess_opt(image_data):
    # image_data: (C, H, W)
    # Type 3 (Cloudy Input)
    c_min = CLIP_MIN[2]
    c_max = CLIP_MAX[2]
    new_img = np.zeros_like(image_data, dtype=np.float32)
    for c in range(min(image_data.shape[0], len(c_min))):
         min_v = c_min[c]
         max_v = c_max[c]
         val = np.clip(image_data[c], min_v, max_v)
         new_img[c] = val
    new_img /= SCALE
    return new_img

def save_output(data, output_path, profile):
    # Update profile
    profile.update(count=data.shape[0], dtype=rasterio.float32)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1', type=str, required=True)
    parser.add_argument('--s2', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    print(f"Loading S1: {args.s1}")
    print(f"Loading S2: {args.s2}")

    # Load Images
    with rasterio.open(args.s1) as src_s1:
        s1_data = src_s1.read().astype(np.float32)
    
    with rasterio.open(args.s2) as src_s2:
        s2_data = src_s2.read().astype(np.float32)
        s2_profile = src_s2.profile

    print(f"S1 shape: {s1_data.shape}")
    print(f"S2 shape: {s2_data.shape}")

    # Preprocess
    s1_norm = preprocess_sar(s1_data)
    s2_norm = preprocess_opt(s2_data)

    # Add batch dim -> (1, C, H, W)
    s1_input = s1_norm[np.newaxis, ...]
    s2_input = s2_norm[np.newaxis, ...]
    
    # Model Setup
    h, w = s2_data.shape[1], s2_data.shape[2]
    
    # Force Channels First
    K.set_image_data_format('channels_first')
    
    # Configure Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    print("Building Model...")
    # Input shape: ((13, h, w), (2, h, w))
    # Note: s1_data might have different channels? Assume 2.
    # s2_data must have 13.
    model, _ = DSen2CR_model(((s2_data.shape[0], h, w), (s1_data.shape[0], h, w)), 
                             batch_per_gpu=1, 
                             num_layers=16,
                             use_cloud_mask=True, 
                             include_sar_input=True)
    
    weights_path = '/app/evaluation/checkpoints/dsen2cr_carl_weights.hdf5'
    print(f"Loading weights from {weights_path}...")
    model.load_weights(weights_path)
    
    print("Predicting...")
    prediction = model.predict([s2_input, s1_input])
    
    # Post-process
    # prediction[0] is (C_out, H, W)
    # Channels: 0-12 are corrected image.
    pred_img = prediction[0, :13, :, :]
    pred_img *= SCALE
    
    print(f"Saving output to {args.output}...")
    save_output(pred_img, args.output, s2_profile)
    print("Done.")

if __name__ == "__main__":
    main()

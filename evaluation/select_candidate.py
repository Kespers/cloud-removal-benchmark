
import os
import json
import rasterio
import numpy as np
import argparse
from glob import glob

# Try to import s2cloudless
try:
    from s2cloudless import S2PixelCloudDetector
except ImportError:
    print("Error: s2cloudless not installed. Please install it using 'pip install s2cloudless'")
    import sys
    sys.exit(1)

def get_cloud_probability(image_path, detector):
    """
    Computes the average cloud probability for a given image.
    Assumes image has at least 13 bands (S2 standard) or at least the bands required by s2cloudless.
    s2cloudless requires bands: B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
    (bands 0, 1, 3, 4, 7, 8, 9, 10, 11, 12 in 0-indexed 13-band stack)
    """
    with rasterio.open(image_path) as src:
        # Read all bands (13 channels)
        # s2cloudless expects (H, W, C) with specific bands
        # We need to map 13-band index to s2cloudless expectations
        # The input images we generated are likely 13 bands (0-12)
        
        # Bands index in 13-channel Stack:
        # 0: B01
        # 1: B02
        # 2: B03
        # 3: B04
        # 4: B05
        # 5: B06
        # 6: B07
        # 7: B08
        # 8: B8A
        # 9: B09
        # 10: B10
        # 11: B11
        # 12: B12
        
        # S2PixelCloudDetector expects:
        # B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
        # Indices: 0, 1, 3, 4, 7, 8, 9, 10, 11, 12
        
        required_indices = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]
        
        try:
            bands = []
            for idx in required_indices:
                # read() is 1-indexed, so we need idx+1
                bands.append(src.read(idx + 1))
            
            # Stack to (C, H, W) -> (10, 256, 256)
            stack = np.stack(bands, axis=0)
            
            # Transpose to (H, W, C) -> (256, 256, 10)
            stack = np.transpose(stack, (1, 2, 0))
            
            # Normalize to 0-1 if not already (s2cloudless expects 0-1 range usually, or values compliant with S2 L1C)
            # DSen2-CR outputs might be scaled. 
            # If standard S2 L1C is 0-10000, we divide by 10000.
            # DSen2-CR output is typically 0-1 (if we look at inference code: pred_img *= SCALE where SCALE=2000... wait)
            # In inference_internal.py:
            # pred_img *= SCALE (variable is 2000). 
            # The original code divides by 2000 to normalize input.
            # So the output is likely in range 0-2000+ ?
            # Let's assume we need to divide by 10000 to get standard reflectance 0-1 for s2cloudless?
            # Or does s2cloudless work with 0-1? 
            # Documentation says "values should be reflectance (0-1)".
            # We need to check what DSen2-CR output range is. 
            # The code says `pred_img *= SCALE` (2000). So it brings it back to 0-2000 range? 
            # Standard S2 is 0-10000. 
            # Let's try dividing by 2000 to get 0-1, or 10000? 
            # Let's normalize by dividing by 10000 (standard top of amosphere reflectance is 0-10000).
            # If DSen2-CR was trained on 0-2000 normalized data, then *2000 brings it to 0-2000??
            # Let's checking `dsen2cr_tools.py`: scale=2000.
            # "scale = 2000"
            # It seems DSen2-CR works with values 0-1 (input / 2000).
            # Output is multiplied by 2000. So output is 0-2000 approx.
            # 2000 is likely the scaling factor. 
            # s2cloudless expects 0-1. So we divide by 2000? NO, standard reflectance is usually considered 0-10000 ints.
            # If s2cloudless expects 0-1 floats:
            # (value / 10000) is standard.
            # validation: 2000/10000 = 0.2. 
            # Let's err on side of standard S2Cloudless usage which is trained on L1C.
            # The output of the model is supposedly "cloud free" so probability should be low regardless.
            
            # Let's use robust division. Max value in typical float image might be ~2000.
            # If we divide by 10000 it might be too dark/small.
            # If we divide by 2000 (the scale factor), we get 0-1.
            # Let's try dividing by 10000 first as that's standard S2.
            
            stack = stack.astype(np.float32) / 10000.0
            
            # Get probability map (1, H, W)
            # wrapper expects batch of images (N, H, W, C)
            cloud_probs = detector.get_cloud_probability_maps(stack[np.newaxis, ...])
            
            # Average probability
            avg_prob = np.mean(cloud_probs)
            return avg_prob
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return 1.0 # High penalty

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='evaluation/output', help="Directory containing patch folders")
    args = parser.parse_args()
    
    detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)
    
    candidates = {} # {patch_id: {"best_date": date_id, "min_cloud_prob": prob}}
    
    patch_dirs = glob(os.path.join(args.output_dir, "*"))
    
    print(f"Found {len(patch_dirs)} patch directories.")
    
    for p_dir in patch_dirs:
        if not os.path.isdir(p_dir):
            continue
            
        patch_id = os.path.basename(p_dir)
        print(f"Analyzing Patch {patch_id}...", end='\r')
        
        best_date = None
        min_prob = 1.1
        
        tif_files = glob(os.path.join(p_dir, "*.tif"))
        for tif in tif_files:
            date_id = os.path.splitext(os.path.basename(tif))[0]
            
            prob = get_cloud_probability(tif, detector)
            
            if prob < min_prob:
                min_prob = prob
                best_date = date_id
        
        if best_date is not None:
            candidates[patch_id] = {
                "best_date": best_date,
                "min_cloud_prob": float(min_prob)
            }
            
    print("\nAnalysis complete.")
    
    output_json = os.path.join(os.path.dirname(args.output_dir), "candidates.json")
    with open(output_json, 'w') as f:
        json.dump(candidates, f, indent=4)
        
    print(f"Candidates saved to {output_json}")

if __name__ == "__main__":
    main()

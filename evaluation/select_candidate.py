
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
    with rasterio.open(image_path) as src:
        
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

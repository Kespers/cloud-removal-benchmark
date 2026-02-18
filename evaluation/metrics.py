
import os
import json
import rasterio
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import argparse

# Constants
SCALE = 2000.0

def safe_divide(n, d):
    return n / d if d != 0 else 0

def load_image(path):
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        
        try:
             # Try standard S2 indices for RGB
             r = src.read(4)
             g = src.read(3)
             b = src.read(2)
        except:
             # Fallback
             r = src.read(1)
             g = src.read(2)
             b = src.read(3)
             
        img = np.stack([r, g, b], axis=0).astype(np.float32)
        # Normalize to 0-1
        # The formula says x,y in [0, 1].
        # DSen2-CR output is roughly 0-2000.
        img = np.clip(img / SCALE, 0, 1)
        return img

def calculate_nrmse(x, y):
    # x, y: (C, H, W)
    mse = np.mean((x - y) ** 2)
    rmse = np.sqrt(mse)
    # Normalized? Usually by range (1-0 = 1). So NRMSE = RMSE.
    return rmse

def calculate_psnr(x, y):
    # x, y in [0, 1]
    rmse = calculate_nrmse(x, y)
    if rmse == 0:
        return 100 # Perfect match
    psnr = 20 * np.log10(1.0 / rmse)
    return psnr

def calculate_ssim(x, y):
    # x, y: (C, H, W) -> Transpose to (H, W, C) for skimage
    x_t = np.transpose(x, (1, 2, 0))
    y_t = np.transpose(y, (1, 2, 0))
    
    # multichannel=True, data_range=1.0
    return ssim(x_t, y_t, channel_axis=-1, data_range=1.0)

def calculate_sam(x, y):
    
    numerator = np.sum(x * y)
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    
    if denom == 0:
        return 0
        
    val = numerator / denom
    val = np.clip(val, -1, 1) # Numerical stability
    return np.degrees(np.arccos(val))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=str, default='evaluation/candidates.json')
    parser.add_argument('--output', type=str, default='evaluation/metrics_report.csv')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    if not os.path.exists(os.path.join(project_root, args.candidates)):
        print("Candidates file not found.")
        return

    with open(os.path.join(project_root, args.candidates), 'r') as f:
        candidates = json.load(f)

    gt_file = os.path.join(project_root, "dataset", "ground_truths.json")
    with open(gt_file, 'r') as f:
        ground_truths = json.load(f)
    gt_patches = ground_truths.get("gt_patches", {})
    
    results = []
    
    # S2 Base Path for GT
    s2_base = os.path.join(project_root, "dataset", "s2_asiaWest_test", "ROIs1868", "100", "S2")
    output_base = os.path.join(project_root, "evaluation", "output")
    
    print(f"Calculating metrics for {len(candidates)} patches...")
    
    for patch_id, cand_info in candidates.items():
         best_date = cand_info["best_date"]
         
         # Load Prediction
         pred_path = os.path.join(output_base, str(patch_id), f"{best_date}.tif")
         img_pred = load_image(pred_path)
         
         # Load GT
         gt_info = gt_patches.get(patch_id, {})
         gt_date = int(gt_info.get("reference_timestep", -1))
         
         gt_dir = os.path.join(s2_base, str(gt_date))
         gt_path = ""
         if os.path.exists(gt_dir):
            for f in os.listdir(gt_dir):
                if f.endswith(f"patch_{patch_id}.tif"):
                    gt_path = os.path.join(gt_dir, f)
                    break
         
         img_gt = load_image(gt_path)
         
         if img_pred is None or img_gt is None:
             print(f"Skipping Patch {patch_id}: Missing image data.")
             continue
             
         # Compute Metrics
         nrmse = calculate_nrmse(img_pred, img_gt)
         psnr = calculate_psnr(img_pred, img_gt)
         ssim_val = calculate_ssim(img_pred, img_gt)
         sam = calculate_sam(img_pred, img_gt)
         
         results.append({
             "patch_id": patch_id,
             "best_date": best_date,
             "gt_date": gt_date,
             "NRMSE": nrmse,
             "PSNR": psnr,
             "SSIM": ssim_val,
             "SAM": sam
         })
         
    # Save Results
    df = pd.DataFrame(results)
    out_path = os.path.join(project_root, args.output)
    df.to_csv(out_path, index=False)
    
    # Summary
    print("\n--- Metrics Summary ---")
    print(df[["NRMSE", "PSNR", "SSIM", "SAM"]].mean())
    print(f"\nDetailed report saved to {out_path}")

if __name__ == "__main__":
    main()

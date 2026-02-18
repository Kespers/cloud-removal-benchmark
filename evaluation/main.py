
import os
import argparse
import subprocess
import sys
import json
import time
from multiprocessing import Pool, cpu_count, current_process

def convert_path_to_docker(win_path, project_root):
    # Normalize paths to absolute
    win_path = os.path.abspath(win_path)
    project_root = os.path.abspath(project_root)
    
    # Check if path is within project root
    if not win_path.lower().startswith(project_root.lower()):
        raise ValueError(f"Path {win_path} is not within project root {project_root}")
        
    rel_path = os.path.relpath(win_path, project_root)
    # Convert to forward slashes for Docker/Linux
    docker_path = '/app/' + rel_path.replace('\\', '/')
    return docker_path

def run_docker_inference_task(args):
    """
    Worker function for multiprocessing.
    args: (s1_path, s2_path, output_path, project_root, image_name, memory_limit)
    """
    s1_path, s2_path, output_path, project_root, image_name, memory_limit = args
    
    try:
        docker_s1 = convert_path_to_docker(s1_path, project_root)
        docker_s2 = convert_path_to_docker(s2_path, project_root)
        docker_output = convert_path_to_docker(output_path, project_root)
    except ValueError as e:
        return f"Error path: {e}"

    run_cmd = [
        "docker", "run", "--rm",
        f"--memory={memory_limit}", 
        "-v", f"{project_root}:/app",
        image_name,
        "python", "/app/evaluation/inference_internal.py",
        "--s1", docker_s1,
        "--s2", docker_s2,
        "--output", docker_output
    ]
    
    try:
        # Suppress output unless error
        subprocess.run(run_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return "OK"
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode()
        return f"Error: {err_msg}"

def main():
    parser = argparse.ArgumentParser(description="Run batch inference using Docker")
    parser.add_argument('--limit', type=int, default=150, help="Limit number of patches to process (default: 150)")
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument('--memory', type=str, default="6g", help="Memory limit per container (default: 6g)")
    parser.add_argument('--dry-run', action='store_true', help="Print commands without executing")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Dataset Paths
    s1_base = os.path.join(project_root, "dataset", "s1_asiaWest_test", "ROIs1868", "100", "S1")
    s2_base = os.path.join(project_root, "dataset", "s2_asiaWest_test", "ROIs1868", "100", "S2")
    
    # Ground Truths
    gt_file = os.path.join(project_root, "dataset", "ground_truths.json")
    if not os.path.exists(gt_file):
        print(f"Error: GP file not found at {gt_file}")
        sys.exit(1)
        
    with open(gt_file, 'r') as f:
        ground_truths = json.load(f)
        
    gt_patches = ground_truths.get("gt_patches", {})
    
    # Output Directory
    eval_output_dir = os.path.join(project_root, "evaluation", "output")
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Build Docker Image Once
    if not args.dry_run:
        print("Ensuring Docker image exists...")
        try:
            subprocess.run(["docker", "build", "-t", "dsen2cr:latest", "."], cwd=project_root, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("Docker build failed.")
            sys.exit(1)

    tasks = []
    
    print(f"Preparing tasks for patches 0 to {args.limit-1}...")

    for patch_id in range(args.limit):
        patch_str = str(patch_id)
        if patch_str not in gt_patches:
            continue
            
        gt_info = gt_patches[patch_str]
        gt_timestep = int(gt_info.get("reference_timestep", -1))
        
        # Create output directory for this patch
        patch_output_dir = os.path.join(eval_output_dir, patch_str)
        if not os.path.exists(patch_output_dir) and not args.dry_run:
            try:
                os.makedirs(patch_output_dir)
            except FileExistsError:
                pass

        for date_id in range(30): # 0 to 29
            if date_id == gt_timestep:
                continue
                
            s1_date_dir = os.path.join(s1_base, str(date_id))
            s2_date_dir = os.path.join(s2_base, str(date_id))
            
            # Find the file for this patch
            s1_file = None
            if os.path.exists(s1_date_dir):
                for f in os.listdir(s1_date_dir):
                    if f.endswith(f"patch_{patch_id}.tif"):
                        s1_file = os.path.join(s1_date_dir, f)
                        break
            
            s2_file = None
            if os.path.exists(s2_date_dir):
                for f in os.listdir(s2_date_dir):
                    if f.endswith(f"patch_{patch_id}.tif"):
                        s2_file = os.path.join(s2_date_dir, f)
                        break
            
            if not s1_file or not s2_file:
                continue

            output_file = os.path.join(patch_output_dir, f"{date_id}.tif")
            
            if os.path.exists(output_file):
                continue
            
            tasks.append((s1_file, s2_file, output_file, project_root, "dsen2cr:latest", args.memory))

    print(f"Found {len(tasks)} inference tasks.")
    
    if args.dry_run:
        print(f"[Dry Run] Would execute {len(tasks)} tasks processing with {args.workers} workers.")
        return

    print(f"Starting execution with {args.workers} workers (Memory limit: {args.memory}/container)...")
    start_time = time.time()
    
    completed = 0
    errors = 0
    
    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(run_docker_inference_task, tasks):
            if result == "OK":
                sys.stdout.write(".")
            else:
                sys.stdout.write("x")
                errors += 1
            sys.stdout.flush()
            completed += 1
            
    duration = time.time() - start_time
    print(f"\nBatch processing complete. Processed {completed} images in {duration:.2f} seconds.")
    if errors > 0:
        print(f"Encountered {errors} errors.")

if __name__ == "__main__":
    main()

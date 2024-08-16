from huggingface_hub import snapshot_download
import os 

output_dir = "data"
repo_id = "liuyubian/Robo360" 
dirs = ["batch2/xarm6_fold_tshirt","batch2/xarm6_unfold_tshirt", "batch1/cloth"]

os.makedirs(output_dir, exist_ok=True)

for d in dirs:
    print("Downloading", d)
    snapshot_download(repo_id=repo_id, allow_patterns=[f"{d}/*"], repo_type="dataset", local_dir=output_dir)
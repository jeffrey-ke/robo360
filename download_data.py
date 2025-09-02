from huggingface_hub import snapshot_download
import pdb
import os 
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--out-dir", type=str, default="data")
args = parser.parse_args()

output_dir = args.out_dir
repo_id = "liuyubian/Robo360" 
dirs = ["batch2/xarm6_fold_tshirt","batch2/xarm6_unfold_tshirt", "batch1/cloth"]

os.makedirs(output_dir, exist_ok=True)

for d in dirs:
    print("Downloading", d)
    snapshot_download(repo_id=repo_id, allow_patterns=[f"{d}/*"], repo_type="dataset", local_dir=output_dir)

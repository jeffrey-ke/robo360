import os
import cv2
import time
import numpy as np
from preprocess import get_start_idx


def preprocess_canon(
        path: str, outpath: str, 
        target_idx: int, calib_len: int, start_idx: int, 
        subset=None, preview=False, verbose=True, debug_out=None
    ):
    t0 = time.time()
    os.makedirs(f"{outpath}", exist_ok=True)
    cameras = sorted(os.listdir(f"{path}/canon"))
    if isinstance(subset, list):
        cameras = [f"{cam:04d}.MP4" for cam in subset]
    for video_name in cameras:
        # configure debug path
        debug_path = None
        if debug_out is not None:
            debug_path = f"{debug_out}/canon/{video_name[:-4]}"
        # compute align_idx
        align_idx = get_start_idx(f"{path}/canon/{video_name}", calib_len, debug_out=debug_path)
        video_reader = cv2.VideoCapture(f"{path}/canon/{video_name}")
        i0 = align_idx + start_idx
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, i0 + target_idx)
        _, frame = video_reader.read()
        cv2.imwrite(f"{outpath}/{video_name[:-4]}.png", frame)
    if verbose:
        print(f"Preprocessed canon saved to {outpath}/canon in {time.time() - t0:.1f}s")


def preprocess_static_nerf(path, outpath, target_idx, calib_len, start_idx, verbose=True, debug_out=None):
    preprocess_canon(path, outpath, target_idx, calib_len, start_idx, verbose=verbose, debug_out=debug_out)


if __name__ == "__main__":
    in_path = "../dataset/2023_10_21/multiview/xarm6_2023_10_21_12_34_53_617077"
    calib_path = "../dataset/calibration"
    out_path = "../dataset/2023_10_21/static_nerf/orange_juice_pour"
    preprocess_static_nerf(in_path, out_path, target_idx=20 * 30, calib_len=10 * 30, start_idx=3 * 30, verbose=True)
    os.system(f"cp {calib_path}/poses_bounds_new.npy {out_path}/poses_bounds.npy")

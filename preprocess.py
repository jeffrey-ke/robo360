import numpy as np
import time
import os
import cv2
from types import FunctionType
import tqdm
from PIL import Image


def get_start_idx(path: str, calib_len: int, debug_out=None):
    cap = cv2.VideoCapture(path)
    brightness = []
    for j in range(calib_len):
        success, image = cap.read()
        brightness.append(np.mean(image))
        if not success:
            cap.release()
            raise ValueError(f"Video failed at frame {j}")
    cap.release()
    brightness = np.array(brightness)
    delta_brightness = brightness[1:] - brightness[:-1]
    idx_found = np.argmax(delta_brightness)
    if debug_out is not None:
        os.makedirs(debug_out, exist_ok=True)
        cap = cv2.VideoCapture(path)
        for i in range(idx_found, idx_found + 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, image = cap.read()
            cv2.imwrite(f"{debug_out}/{i:06d}.png", image)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_found + 90)
        success, image = cap.read()
        cv2.imwrite(f"{debug_out}/{idx_found+90:06d}.png", image)
        cap.release()
    return idx_found


def preprocess_traj(
        path: str, outpath: str, 
        img_transform: FunctionType, 
        align_idx: int, duration: int, start_idx: int, 
        verbose=True
    ):
    for video_name in sorted(os.listdir(f"{path}/traj")):
        if not video_name.endswith("mp4"):
            continue
        os.makedirs(f"{outpath}/traj/{video_name[:-4]}", exist_ok=True)
        video_reader = cv2.VideoCapture(f"{path}/traj/{video_name}")
        video_length = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        # seek to start_idx
        i0 = align_idx + start_idx
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, i0)
        # process video
        if video_length < i0 + duration:
            print(f"Video {video_name} is too short")
            continue
        for i in range(duration):
            _, frame = video_reader.read()
            frame = img_transform(frame)
            cv2.imwrite(f"{outpath}/traj/{video_name[:-4]}/{i:06d}.png", frame)
    for filename in os.listdir(f"{path}/traj"):
        if filename.endswith(".npy"):
            npy_arr = np.load(f"{path}/traj/{filename}")
            np.save(f"{outpath}/traj/{filename}", npy_arr[i0:i0+duration])
    if verbose:
        print(f"Preprocessed traj saved to {outpath}/traj")


def preprocess_canon(
        path: str, outpath: str, 
        img_transform: FunctionType, 
        calib_len: int, start_idx: int, duration: int, 
        subset=None, preview=False, verbose=True, debug_out=None
    ):
    t0 = time.time()
    cameras = sorted(os.listdir(f"{path}/canon"))
    if isinstance(subset, list):
        cameras = [f"{cam:04d}.MP4" for cam in subset]
    for video_name in tqdm.tqdm(cameras):
        # configure debug path
        debug_path = None
        if debug_out is not None:
            debug_path = f"{debug_out}/canon/{video_name[:-4]}"
        # compute align_idx
        align_idx = get_start_idx(f"{path}/canon/{video_name}", calib_len, debug_out=debug_path)
        os.makedirs(f"{outpath}/canon/{video_name[:-4]}", exist_ok=True)
        video_reader = cv2.VideoCapture(f"{path}/canon/{video_name}")
        i0 = align_idx + start_idx
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, i0)
        if preview:
            os.makedirs(f"{outpath}/canon_preview", exist_ok=True)
        for i in range(duration):
            # if verbose:
            #     print(f"Preprocessing {i}/{video_length} {video_name}") # a bit annoying
            _, frame = video_reader.read()
            frame = img_transform(frame)
            if preview and i == 0:
                cv2.imwrite(f"{outpath}/canon_preview/{video_name[:-4]}.png", frame)
            # cv2.imwrite(f"{outpath}/canon/{video_name[:-4]}/{i:06d}.png", frame)
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(f"{outpath}/canon/{video_name[:-4]}/{i:06d}.jpg", "JPEG", quality=90)
        # if verbose:
        #     print(f"Preprocessed canon video saved to {outpath}/canon/{video_name[:-4]} in {time.time() - t0:.1f}s")
    if verbose:
        print(f"Preprocessed canon saved to {outpath}/canon in {time.time() - t0:.1f}s")

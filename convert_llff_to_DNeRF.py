import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import json
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, 
					default='/dataset/dataset/debug', help='base folder path')
parser.add_argument('--frame_skip', type=int,default=8)
parser.add_argument('--normalize_time', action='store_true')
parser.add_argument('--source_framerate',type=float,default=29.97)

args = parser.parse_args()



input_path = os.path.normpath(args.folder_path)
output_path = input_path + '_DNeRF'

if os.path.exists(output_path):
    shutil.rmtree(output_path)

os.makedirs(output_path, exist_ok = True)

files = os.listdir(input_path)
files = sorted(files)

all_frames = [] # contains (cam_id, frame_id, width, height)
all_cams = [] # cam ids

found_vid=0
for file in tqdm(files):
    if ('npy' in file):
        print("loading poses")
        poses = np.load(os.path.join(input_path, file))
    elif ('mp4' in file or 'MP4' in file) and found_vid < 1:
        print("loading video")
        os.makedirs(f"{output_path}/{file[:-4]}", exist_ok = True)
        all_cams.append(int(file[:-4]))
        cap = cv2.VideoCapture(os.path.join(input_path, file))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(num_frames):
            success, frame = cap.read()
            if i % args.frame_skip == 0:
                h, w = frame.shape[:2]
                all_frames.append((int(file[:-4]), i, w, h))
                cv2.imwrite(f"{output_path}/{file[:-4]}/{i:04d}.png", frame)
        cap.release()
        found_vid+=1

all_cams = sorted(all_cams)
index_map = {}
for i in range(len(all_cams)):
    index_map[all_cams[i]] = i

poses = poses[:, :15].reshape(-1, 3, 5)
intrinsics = poses[:, :, -1]
poses = np.concatenate([poses[..., 1:2], -poses[..., 0:1], poses[..., 2:3], poses[..., 3:4]], -1)

# to homogeneous 
last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

random.shuffle(all_frames)

name = ["train", "val", "test"]
percent = [0.8, 0.1, 0.1]
frames = []

last = 0
for x in percent:
    frames.append(all_frames[int(len(all_frames) * last): int(len(all_frames) * (last + x))])
    last += x

max_time = None
min_time = None
if args.normalize_time:
    all_times = [all_frames[i][1] for i in range(len(all_frames))]
    max_time = float(max(all_times))
    min_time = float(min(all_times))

for i in range(len(frames)):

    frame_i = []
    for frame in frames[i]:
        x, y, w, h = frame
        new_x = index_map[x]
        f_x = intrinsics[new_x, 2]
        angle, trans = np.arctan(w * 0.5 / f_x) * 2, poses[new_x].tolist()
        k = np.eye(3)
        k[0, 0] = f_x
        k[1, 1] = f_x
        k[0, 2] = w * 0.5
        k[1, 2] = h * 0.5

        if args.normalize_time:
            time = float((y - min_time) / (max_time - min_time))
            # as double
        else:
            time = y / args.source_framerate
        file_path = f"./{x:04d}/{y:04d}.png"
        frame_map = {"file_path": file_path, "camera_angle_x": angle, "time": time, "transform_matrix": trans,"w": w, "h": h,"k": k.tolist()}
        frame_i.append(frame_map)

    dictionary = {'frames': frame_i}
    json_object = json.dumps(dictionary, indent=4)
    
    with open(f"{output_path}/transforms_{name[i]}.json", "w") as outfile:
        outfile.write(json_object)

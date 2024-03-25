import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import json
import shutil
import argparse

class Processor:
    def __init__(self):
        self.parse_args()
    
    def run(self):
        self.prep_folders()
        self.load_poses()
        self.load_all_cams()
        self.gen_json()

    def prep_folders(self):
        self.input_path = os.path.normpath(self.args.folder_path)
        self.output_path = self.input_path + '_DNeRF'

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

        os.makedirs(self.output_path, exist_ok = True)

        self.files = os.listdir(self.input_path)
        self.files = sorted(self.files)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--folder_path', type=str, 
					        default='/dataset/dataset/debug', help='base folder path')
        parser.add_argument('--frame_skip', type=int,default=8)
        parser.add_argument('--normalize_time', action='store_true')
        parser.add_argument('--source_framerate',type=float,default=29.97)
        parser.add_argument('--calib_len', type=int, default=10*30)
        parser.add_argument('--target_idx', type=int, default=20*30)
        parser.add_argument('--start_idx', type=int, default=3*30)

        self.args = parser.parse_args()

    def get_start_idx(self,path, calib_len):
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
        return idx_found

    def load_poses(self):
        for file in self.files:
            if ('npy' in file):
                print("loading poses")
                self.poses = np.load(os.path.join(self.input_path, file))


    def load_all_cams(self):
        self.all_frames = []
        self.all_cams = []

        for file in tqdm(self.files):
            if ('mp4' in file or 'MP4' in file):
                print("loading video")
                cam_id = file[:-4]
                os.makedirs(f"{self.output_path}/{cam_id}", exist_ok = True)
                self.all_cams.append(int(cam_id))

                align_idx = self.get_start_idx(os.path.join(self.input_path, file), self.args.calib_len)
                i0 = align_idx + self.args.start_idx
        
                cap = cv2.VideoCapture(os.path.join(self.input_path, file))
                cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
                #num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                num_frames = self.args.target_idx
                for i in range(num_frames):
                    success, frame = cap.read()
                    if i % self.args.frame_skip == 0:
                        h, w = frame.shape[:2]
                        self.all_frames.append((int(cam_id), i, w, h))
                        cv2.imwrite(f"{self.output_path}/{cam_id}/{cam_id}_{i:04d}.png", frame)
                cap.release()

    def gen_json(self):
        self.all_cams = sorted(self.all_cams)
        index_map = {}
        for i in range(len(self.all_cams)):
            index_map[self.all_cams[i]] = i

        self.poses = self.poses[:, :15].reshape(-1, 3, 5)
        intrinsics = self.poses[:, :, -1]

        self.poses = np.concatenate([self.poses[..., 1:2], -self.poses[..., 0:1], self.poses[..., 2:3], self.poses[..., 3:4]], -1)

        # to homogeneous 
        last_row = np.tile(np.array([0, 0, 0, 1]), (len(self.poses), 1, 1)) # (N, 1, 4)
        self.poses = np.concatenate([self.poses, last_row], axis=1) # (N, 4, 4) 

        random.shuffle(self.all_frames)

        name = ["train", "val", "test"]
        percent = [0.8, 0.1, 0.1]
        frames = []

        last = 0
        for x in percent:
            frames.append(self.all_frames[int(len(self.all_frames) * last): int(len(self.all_frames) * (last + x))])
            last += x

        max_time = None
        min_time = None
        if self.args.normalize_time:
            all_times = [self.all_frames[i][1] for i in range(len(self.all_frames))]
            max_time = float(max(all_times))
            min_time = float(min(all_times))

        for i in range(len(frames)):

            frame_i = []
            for frame in frames[i]:
                x, y, w, h = frame
                new_x = index_map[x]
                f_x = intrinsics[new_x, 2]
                angle, trans = np.arctan(w * 0.5 / f_x) * 2,self.poses[new_x].tolist()
                k = np.eye(3)
                k[0, 0] = f_x
                k[1, 1] = f_x
                k[0, 2] = w * 0.5
                k[1, 2] = h * 0.5

                if self.args.normalize_time:
                    time = float((y - min_time) / (max_time - min_time))
                    # as double
                else:
                    time = y / self.args.source_framerate
                file_path = f"./{x:04d}/{x:04d}_{y:04d}.png"
                frame_map = {"file_path": file_path, "camera_angle_x": angle, "time": time, "transform_matrix": trans,"w": w, "h": h,"k": k.tolist()}
                frame_i.append(frame_map)

            dictionary = {'frames': frame_i}
            json_object = json.dumps(dictionary, indent=4)
    
            with open(f"{self.output_path}/transforms_{name[i]}.json", "w") as outfile:
                outfile.write(json_object)


if __name__ == "__main__":
    processor = Processor()
    processor.run()
#!/bin/bash
# take user input for the directory containing the images

DATA_DIR=$1

python3 robo2nerf.py --folder_path ${DATA_DIR}/batch1/cloth/ --calib_len 2 --normalize_time --no_timesync --start_idx 45 --target_idx 40 --frame_skip 1 --exclude_cams 1 7 8 14 15 21 22 23 31 33 34 35 36 37 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 62 63 65 70 71 72 77 79 85 86 88
python3 robo2nerf.py --folder_path ${DATA_DIR}/batch2/xarm6_fold_tshirt --normalize_time --start_idx 315 --target_idx 40 --frame_skip 1

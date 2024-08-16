# How to download and prep robo360 data

## Download the data

The robo360 data is stored on Huggingface, [here](https://huggingface.co/datasets/liuyubian/Robo360).

Install the huggingface python package:
```pip install huggingface_hub```

Make sure you're in the robo360 directory, and execute:
```python3 download_data.py```

This will download a few selected robo360 datasets, but you can easily add more. The output will be in `robo360/data/`.

## Prepocess it

We need to align the videos, extract the frames we want, and save them into the right file structure.
Aligning means we find the 'light flash' in all frames, and use it do a coarse time synchronization between all cameras.
Extracting the frames we want means we select the section of the motion we're interested in, and control what views and number of frames we want.

We provide scripts to preprocess the datasets from the paper, which can be run by  

```./scripts/convert_all.sh data/```

To preprocess new data, use the `robo2nerf.py` script with the following arguments:
- folder_path: data location 
- frame_skip: for each camera, select every n'th frame. This can help if we want to capture longer sequences and limit the total data size.
- view_skip: the same concept, but now select every n'th view. 
- calib_len: the number of frames considered for the 'light flash' coarse time calibration procedure described above. Within this window we arg max intensity for each view.
- target_idx: the total number of frames for each view.
- start_idx: the frame idx we want to start recording from. E.g. when we want to capture 3 seconds of video from a 30 fps camera, starting from t=10s, we set target_idx to 90 (3*30) and start_idx to 300 (10*30).
- exclude_cams: cameras rejected, this can be useful to exclude extreme viewpoints or views where the operator is entirely visible.
- normalize_time: when added as a flag, we normalize the time in the output in [0,1]. Please use this flag for use with deformGS.
- no_timesync: when added as flag timesync is ommitted and start_idx is taken from t=0 instead of the aligned idx. 
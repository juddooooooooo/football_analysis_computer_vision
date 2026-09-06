# Football Analysis Project

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video
-  [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas
## Running a clip

`main.py` takes a video and, optionally, a section of it. Reading a full
match into memory needs hundreds of gigabytes, so long footage must be
sectioned.

```
python main.py                                          # default video, whole file
python main.py input_videos/game.mp4 --random-clip      # random 30s section
python main.py input_videos/game.mp4 --random-clip --clip-duration 60 --seed 5
python main.py input_videos/game.mp4 --start 23:04 --end 23:35
```

Detections are cached in `stubs/` keyed by video and section, so re-running
the same section skips inference. `--no-cache` forces a fresh run — needed
after changing anything that affects detection.

## Pitch calibration

Speed, distance and the 2D pitch view all need a homography mapping image
pixels to pitch metres. It is per video *and* section, because broadcast and
tactical cameras reframe during a match. Calibrations live in
`calibrations.json`.

```
python -m pitch_calibration.interactive input_videos/game.mp4 --start 23:04 --end 23:35
```

Click the landmarks you can see (four minimum; six or seven spread widely
across the frame is much better than four bunched together), then the fit is
refined automatically against the painted lines. It reports a mean alignment
error — under about 3px is good, above 6px means click more landmarks.

Without a calibration the pipeline falls back to `ViewTransformer`, whose
coordinates are hardcoded to one particular 1920x1080 clip and produce
meaningless speeds on any other footage.

## Ball detection

The ball is roughly six pixels across in 720p footage, and default inference
downscales to 640px wide, which loses it. On one test clip the ball was
found in 32% of frames at the default and 98% at `imgsz=1280`. The trade is
about 5x the inference time, so it is worth raising only when ball-dependent
output matters.

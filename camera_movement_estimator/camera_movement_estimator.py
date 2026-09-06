import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        # Track features on the stands and far touchline, which do not move
        # with play. Fractions of the frame width, not fixed columns: the
        # original 900-1050 was tuned to 1920px and lands on open grass at
        # 1280px, leaving nothing trackable and reporting zero movement.
        width = first_frame_grayscale.shape[1]
        mask_features[:, 0:int(width*0.02)] = 1
        mask_features[:, int(width*0.47):int(width*0.55)] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    


    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        """Cumulative camera offset, in pixels, relative to the first frame.

        Measured by phase correlation over the whole frame. The previous
        feature-tracking version recorded a frame-to-frame delta and only
        when it exceeded minimum_distance, but the consumer subtracts these
        values as an offset from frame zero. On a nearly still camera the
        two agree; on a panning one they do not, and most frames were left
        at zero regardless.
        """
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0.0, 0.0]]
        prev = np.float32(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY))
        window = cv2.createHanningWindow((prev.shape[1], prev.shape[0]), cv2.CV_32F)
        total = np.zeros(2)

        for frame_num in range(1, len(frames)):
            grey = np.float32(cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY))
            (dx, dy), _ = cv2.phaseCorrelate(prev, grey, window)
            total = total + np.array([dx, dy])
            camera_movement.append([float(total[0]), float(total[1])])
            prev = grey

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self,frames, camera_movement_per_frame,
                             in_place=False):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame = frame if in_place else frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames
# sort.py - Simple Online and Realtime Tracker (SORT)
# Source: https://github.com/abewley/sort

import numpy as np
import cv2
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    """
    Computes IOU between two bounding boxes.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
              (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    """
    Represents the state of an object detected across multiple frames using Kalman Filter.
    """
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R *= 10.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = np.expand_dims(bbox, axis=1)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, bbox):
        self.kf.update(bbox)

    def predict(self):
        self.kf.predict()
        return self.kf.x[:4].reshape(-1)

    def get_state(self):
        return self.kf.x[:4].reshape(-1)

class Sort:
    """
    SORT tracker: Assigns unique IDs to objects and tracks them over multiple frames.
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        for tracker in self.trackers:
            tracker.predict()

        matched, unmatched_dets, unmatched_trks = [], [], []
        for det in detections:
            best_iou, best_trk = 0, None
            for trk in self.trackers:
                iou_score = iou(det, trk.get_state())
                if iou_score > self.iou_threshold and iou_score > best_iou:
                    best_iou, best_trk = iou_score, trk

            if best_trk is not None:
                best_trk.update(det)
                matched.append(best_trk)
            else:
                unmatched_dets.append(det)

        new_trackers = [KalmanBoxTracker(det) for det in unmatched_dets]
        self.trackers.extend(new_trackers)

        return [(trk.get_state(), trk.id) for trk in self.trackers]

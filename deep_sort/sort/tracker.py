'''
Reference Code Used for SORT Algorithm
https://github.com/HowieMa/DeepSORT_YOLOv5_Pytorch/tree/master/deep_sort/sort
'''
from __future__ import absolute_import
from . import kalman_filter
from . import linear_assignment
import torchvision
import torch
import numpy as np

class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        #
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = 1
        self.features = []

        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        ret = self.mean[:4].copy()  
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == 1 and self.hits >= self._n_init:
            self.state = 2

    def mark_missed(self):
        if self.state == 1:
            self.state = 3
        elif self.time_since_update > self._max_age:
            self.state = 3

    def is_tentative(self):
        return self.state == 1

    def is_confirmed(self):
        return self.state == 2

    def is_deleted(self):
        return self.state == 3


def cost_matrix(tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlbr()
        candidates = np.asarray([detections[i].to_tlbr() for i in detection_indices])
        iou = torchvision.ops.box_iou(torch.from_numpy(bbox.reshape(1,4)), torch.from_numpy(candidates))
        cost_matrix[row, :] = 1. - iou[0].numpy()
    return cost_matrix


class Tracker:
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            cost_matrix = self.metric.distance(features, targets)

            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix


        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]   
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]   

        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

      
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1] 
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                cost_matrix, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature)) 
        self._next_id += 1

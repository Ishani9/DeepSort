'''Reference and Helper Functions Taken from reference: https://github.com/HowieMa/DeepSORT_YOLOv5_Pytorch/blob/master/deep_sort/deep_sort.py
NearestNeighborDistanceMetric,_xywh_to_tlwh,_xywh_to_xyxy, _tlwh_to_xyxy, _xyxy_to_tlwh


'''

import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

from .sort.tracker import Tracker
from .deep.deep_model import Net


__all__ = ['DeepSort']

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class DeepSort(object):
    def __init__(self, model_path, use_cuda=True):
        self.min_confidence = 0.3
        self.nms_max_overlap = 0.5
        nearest_neighbor_dist = 0.2
        self.net = Net(use=True)

        self.net.eval().load_state_dict(torch.load(model_path, map_location='cpu'))
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net.to(self.device)
        self.size = (64, 128)
        metric = NearestNeighborDistanceMetric(nearest_neighbor_dist, 100)
        self.tracker = Tracker(metric, max_iou_distance=0.7, max_age=70, n_init=3)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        bbox = self._xywh_to_tlwh(bbox_xywh)

        features = self.get_features(bbox_xywh, ori_img)

        detections = [Detection(bbox[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)  
        detections = [detections[i] for i in indices]


        self.tracker.predict()      
        self.tracker.update(detections)

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()       
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)  
        return outputs
    
    # @staticmethod
    def _xywh_to_tlwh(self, bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def get_features(self, bbox_xywh, ori_img):
        imgs = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            imgs.append(im)
        if imgs:
            im_batch = torch.cat([transform(resize_im(im, self.size)).unsqueeze(0) for im in imgs], dim=0).float()
            with torch.no_grad():
                features = self.net(im_batch.to(self.device)).cpu().numpy()
        else:
            features = np.array([])
        return features


class Detection(object):
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)    
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """ Convert to format xl, yl, xr, yr """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """ Convert to format `cx, cy, aspect ratio, height """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


def resize_im(im, size):
    return cv2.resize(im.astype(np.float32)/255., size)

def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick

def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

class NearestNeighborDistanceMetric(object):
    def __init__(self, matching_threshold, budget=None):
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = _nn_cosine_distance(self.samples[target], features)
        return cost_matrix
## INBUILT YOLO FUNCTIONS
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox
from deep_sort.deep_sort import DeepSort

import os
import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

import sys

__all__ = ['DeepSort']

currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))
cudnn.benchmark = True

class VideoTracker(object):
    def __init__(self):
        print('Initialize DeepSORT & YOLO-V5')
        # ***************** Initialize ******************************************************
        self.input_path = 'test.mp4'
        self.img_size = 640                        # image size in detector, default is 640

        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.vdo = cv2.VideoCapture()

        # ***************************** initialize DeepSORT **********************************
        # cfg = get_config()
        # cfg.merge_from_file("./configs/deep_sort.yaml")

        use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()
        self.deepsort = DeepSort("deep_sort/deep/checkpoint/model_trial.pth", 
                max_dist=0.2, min_confidence=0.3, 
                nms_max_overlap=0.5, max_iou_distance= 0.7, 
                max_age= 70, n_init=3, nn_budget=100, use_cuda=use_cuda)

        # ***************************** initialize YOLO-V5 **********************************
        self.detector = torch.load('yolov5/weights/yolov5s.pt', map_location=self.device)['model'].float()  # load to FP32
        self.detector.to(self.device).eval()
        if self.half:
            self.detector.half()  # to FP16

        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names

        print('Device: ', self.device)


    def __enter__(self):

        assert os.path.isfile(self.input_path), "Path error"
        self.vdo.open(self.input_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.vdo.isOpened()
        print('Done. Load video file ', self.input_path)

        # ************************* create output *************************
        os.makedirs('output/', exist_ok=True)
        # path of saved video and results
        self.save_video_path = os.path.join('output/', "results.mp4")

        # create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
                                        self.vdo.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))
        print('Done. Create output file ', self.save_video_path)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.vdo.release()
        self.writer.release()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        yolo_time, sort_time, avg_fps = [], [], []
        t_start = time.time()

        idx_frame = 0
        last_out = None
        while self.vdo.grab():
            # Inference *********************************************************************
            t0 = time.time()
            _, img0 = self.vdo.retrieve()

            if idx_frame % 2 == 0:
                outputs, yt, st = self.image_track(img0)        # (#ID, 5) x1,y1,x2,y2,id
                last_out = outputs
                yolo_time.append(yt)
                sort_time.append(st)
                print('Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, yt, st))
            else:
                outputs = last_out  # directly use prediction in last frames
            t1 = time.time()
            avg_fps.append(t1 - t0)

            # post-processing ***************************************************************
            # visualize bbox  ********************************
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                for i,box in enumerate(bbox_xyxy):
                    x1,y1,x2,y2 = [int(i) for i in box]
                    # box text and bar
                    id = int(identities[i]) if identities is not None else 0    
                    color = (255,0,0)
                    label = '{}{:d}'.format("", id)
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                    cv2.rectangle(img0,(x1, y1),(x2,y2),color,3)
                    cv2.rectangle(img0,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
                    cv2.putText(img0,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)

                # add FPS information on output video
                text_scale = max(1, img0.shape[1] // 1600)
                cv2.putText(img0, 'frame: %d fps: %.2f ' % (idx_frame, len(avg_fps) / sum(avg_fps)),
                        (20, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

            # save to video file *****************************
            self.writer.write(img0)

            idx_frame += 1

        print('Avg YOLO time (%.3fs), Sort time (%.3fs) per frame' % (sum(yolo_time) / len(yolo_time),
                                                            sum(sort_time)/len(sort_time)))
        t_end = time.time()
        print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))

    def image_track(self, im0):
        """
        :param im0: original image, BGR format
        :return:
        """
        # preprocess ************************************************************
        # Padded resize
        img = letterbox(im0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # numpy to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        s = '%gx%g ' % img.shape[2:]    # print string

        # Detection time *********************************************************
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.detector(img, augment=True)[0]  # list: bz * [ (#obj, 6)]

        # Apply NMS and filter object other than person (cls:0)
        pred = non_max_suppression(pred, 0.5, 0.5, classes=[2], agnostic=True)
        t2 = time_synchronized()

        # get all obj ************************************************************
        det = pred[0]  # for video, bz is 1
        if det is not None and len(det):  # det: (#obj, 6)  x1 y1 x2 y2 conf cls

            # Rescale boxes from img_size to original im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results. statistics of number of each obj
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

            bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
            confs = det[:, 4:5].cpu()

            # ****************************** deepsort ****************************
            outputs = self.deepsort.update(bbox_xywh, confs, im0)
            # (#ID, 5) x1,y1,x2,y2,track_ID
        else:
            outputs = torch.zeros((0, 5))

        t3 = time.time()
        return outputs, t2-t1, t3-t2


if __name__ == '__main__':
    with VideoTracker() as vdo_trk:
        vdo_trk.run()


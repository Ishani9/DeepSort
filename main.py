## INBUILT YOLO FUNCTIONS
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox
import torchvision.transforms as transforms

from deep_sort.deep_sort import DeepSort
from speed_estimation.orignal_model import NeuralFactory

import os
import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import sys
import matplotlib.pyplot as plt


__all__ = ['DeepSort']

url = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(url, 'yolov5')))
cudnn.benchmark = True

def optical_flow_calc(frame, old_frame):
    prev_image_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    curr_image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_image_gray, curr_image_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    hsv = np.zeros_like(old_frame)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_image_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_image_bgr

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((240,320)),
    transforms.Normalize((0.1,0.1,0.1),(0.5,0.5,0.5))
])

def plot_graphs(prinddata):
    yolo_time, sort_time, ids_no, speede, speedg = prinddata
    yolo_time = np.array(yolo_time)
    sort_time = np.array(sort_time)

    feat = np.unique(ids_no)[1:]
    ytime_data = []
    stime_data = []

    for sp in feat:
        idsall = np.where(ids_no == sp)[0]
        
        ytime_data.append(yolo_time[idsall.astype(int)])
        stime_data.append(sort_time[idsall.astype(int)])

    plt.boxplot(ytime_data)
    plt.xticks(list(range(1,len(feat)+1)), feat)
    plt.title("YOLO Time vs Objects in Frame")
    plt.ylabel("Time")
    plt.xlabel("Objects in Frame")
    plt.savefig("output/YOLO_Time_Analysis.jpg")
    plt.show()

    plt.boxplot(stime_data)
    plt.xticks(list(range(1,len(feat)+1)), feat)
    plt.title("SORT Time vs Objects in Frame")
    plt.ylabel("Time")
    plt.xlabel("Objects in Frame")
    plt.savefig("output/SORT_Time_Analysis.jpg")
    plt.show()

    plt.plot(range(4,len(speede)), speede[4:], label="Speed Estimate")
    plt.plot(range(4,len(speedg)), speedg[4:], label="Ground Truth")
    plt.title("Ground Truth vs Prediction")
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Speed")
    plt.savefig("output/Speed_Analysis.jpg")
    plt.show()

    print("Average Speed Error", np.mean(abs(np.array(speedg)-np.array(speede))))

class VideoTracker(object):
    def __init__(self):
        self.input_path = 'Video_Files/fullvid.mp4'
        deepsort_model_path = "deep_sort/deep/checkpoint/model_orginal_lr2030.pth"
        yolo_model_path = 'yolov5/weights/yolov5s.pt'
        speed_model_path = 'speed_estimation/checkpoints/model_weights.pt'
        self.img_size = 640  
        self.video = cv2.VideoCapture()
        speed_truth_path = 'speed_estimation/Dataset/data.txt'

        self.device = select_device('')
        self.half = self.device.type != 'cpu'
        use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()

        self.deepsort = DeepSort(deepsort_model_path, use_cuda=use_cuda)

        self.detector = torch.load(yolo_model_path, map_location=self.device)['model'].float()  
        self.detector.to(self.device).eval()
        if self.half:
            self.detector.half() 

        self.speed_model = NeuralFactory()
        self.speed_model.eval().load_state_dict(torch.load(speed_model_path, map_location='cpu'))
        self.speed_model = self.speed_model.to(self.device)
        self.speed_truth = open(speed_truth_path).readlines()
        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names

        self.printdata = None

        print('Device: ', self.device)


    def __enter__(self):
        self.video.open(self.input_path)
        assert self.video.isOpened()

        os.makedirs('output/', exist_ok=True)
        self.save_video_path = os.path.join('output/', "results.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.save_video_path, fourcc, self.video.get(cv2.CAP_PROP_FPS), (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        print('Saving output to ', self.save_video_path)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.video.release()
        self.writer.release()
        plot_graphs(self.printdata)
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        yolo_time, sort_time, ids_no, speede, speedg = [], [], [], [], []
        frame_no = 0
        old_frame = np.zeros((480, 640, 3))

        opticalflows = []
        while self.video.grab():
            self.printdata = [yolo_time, sort_time, ids_no, speede, speedg]
            _, frame = self.video.retrieve()                
            if frame_no<4:
                flow = optical_flow_calc(frame.astype(np.uint8), old_frame.astype(np.uint8))
                opticalflows.append(flow)
                speed_est = 0
            else:
                flow = optical_flow_calc(frame.astype(np.uint8), old_frame.astype(np.uint8))
                opticalflows.append(flow)
                opticalflows = opticalflows[-4:]
                speed_est = self.speed_modeler(opticalflows[-1],opticalflows[-2], opticalflows[-3],opticalflows[-4],frame)
            
            old_frame = frame
            outputs, yt, st = self.object_tracker(frame)        
            yolo_time.append(yt)
            sort_time.append(st)
            gtspeed =round(float(self.speed_truth[frame_no].split()[0]),3)
            speede.append(speed_est)
            speedg.append(float(self.speed_truth[frame_no].split()[0]))
            print('Frame %d. Time: YOLO: %.3fs SORT:%.3fs' % (frame_no, yt, st), "Objects: ", len(outputs), "Speed Est: ", round(speed_est,3), "GT: ",  gtspeed)
            ids_no.append(len(outputs))
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                for i,box in enumerate(bbox_xyxy):
                    x1,y1,x2,y2 = [int(i) for i in box]
                    id = int(identities[i]) if identities is not None else 0            
                    cv2.rectangle(frame,(x1, y1),(x2,y2),(255,0,0),3)
                    cv2.rectangle(frame,(x1, y1-15),(x1+25,y1), (255,0,0),-1)
                    cv2.putText(frame,str(id),(x1,y1), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)
            
            cv2.putText(frame, 'Speed Estimate: '+str(round(speed_est,3))+'m/s   Real Speed: '+str(gtspeed)+'m/s' , (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            self.writer.write(frame)
            frame_no = frame_no + 1

    def speed_modeler(self, flow1, flow2, flow3, flow4, frame):
        flow_image_bgr = (flow1 + flow2 + flow3 + flow4)/4
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
        combined = 0.1*frame + flow_image_bgr
        input =  tfms(combined)
        speed = self.speed_model(torch.unsqueeze(input,0).to(self.device).float())
        return speed.item()


    def object_tracker(self, frame):
        img = letterbox(frame, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float() 
        img = img/255.0  

        if img.ndimension() == 3:
            img = torch.unsqueeze(img,0)

        # ------------------ YOLO -----------------------
        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.detector(img, augment=True)[0]  
            pred = non_max_suppression(pred, 0.5, 0.5, classes=[2], agnostic=True)[0]
        t2 = time_synchronized()
        yolot=t2-t1

        # ------------------ SORT -----------------------
        if pred is not None and len(pred):  
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            bbox_xywh = xyxy2xywh(pred[:, :4]).cpu()
            confs = pred[:, 4:5].cpu()
            outputs = self.deepsort.update(bbox_xywh, confs, frame)
        else:
            outputs = torch.zeros((0, 5))
        t3 = time.time()
        sortt = t3-t2
        return outputs, yolot, sortt


if __name__ == '__main__':
    with VideoTracker() as runner:
        runner.run()


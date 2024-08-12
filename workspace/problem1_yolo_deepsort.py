import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pylab import mpl
import torch
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
from models.common import DetectMultiBackend
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

# DeepSORT -> Importing DeepSORT.
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet



# DeepSORT -> Initializing tracker.
max_cosine_distance = 0.4
nn_budget = None
model_filename = './model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)



now_name="undefined"

save_dir = "yolo_"


def read_video(name):
    cap = cv2.VideoCapture(name)
    num = 1
    frames=[]
    while True:
        success, data = cap.read()
        if not success:
            break
        # im = Image.fromarray(data)
        frames.append(data)
        print("正在读取第"+str(num)+"帧")
        num = num + 1
    cap.release()
    return frames



def preprocess_frame(frame, img_size=640):
    # Convert BGR to RGB and resize
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    # Convert to tensor
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img

def yolo(video_path):
    weights_path = 'pretrained/yolov5l.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights_path, device=device)
    names = model.names
    print(names)
    cap = cv2.VideoCapture(video_path)
    num = 0
    while cap.isOpened():
        num = num + 1
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame and perform detection
        img = preprocess_frame(frame).to(device)
        pred = model(img)
        det = non_max_suppression(pred, conf_thres=0.05, iou_thres=0.45)[0]

        # Rescale boxes from img size to original size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

        # Annotator for drawing boxes
        annotator = Annotator(frame, line_width=2, example=str('class_names'))
        save_path = os.path.join(save_dir, (f'frame_{num}') )
        txt_path = os.path.join(save_dir, (f'frame_{num}') )

        if len(det):
            # Process detections
            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            for *xyxy, conf, cls in det:
                c = int(cls)
                if c not in [2, 5, 7]:
                    continue
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

                # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf)
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')


                # save_one_box(xyxy, img, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


        # Show frame
        cv2.imshow('YOLO Detection', annotator.result())
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
            break
        # Save results (image with detections)
        cv2.imwrite(save_path+".png", annotator.result())

    cap.release()
    cv2.destroyAllWindows()

avi_path="show/video_process.avi"
tracker = cv2.TrackerCSRT_create()
now_name=input("请输入这一次任务的名字，不要有空格")
save_dir+=now_name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


yolo(avi_path)
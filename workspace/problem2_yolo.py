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
# 视频尺寸
video_width = 352
video_height = 288
video_len = 7644

# 跟踪设置
T = 12
D = 10
Area = 10


now_name="undefined"

save_dir = "yolo_track_"


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

# 存储每一帧中，所有车辆的信息
mess=[] # 第 i 帧 第 j 辆车的信息，信息格式为 x,y,a,b,ID=0,book 分别为中心点坐标,ID,是否已经有后继车辆
# ID = 0 # 唯一识别符
cross=set() # 已经跨线的车辆ID

# 反归一化函数
def de_normalize(x,y):
    x *= video_width
    y *= video_height
    return x,y

# 欧拉距离
def dis(x1,y1,x2,y2):
    # print("dis=",((x1-x2)**2+(y1-y2)**2)**0.5)
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def superposition (xc1,yc1,a1,b1,xc2,yc2,a2,b2):
    l1 = xc1 - a1 / 2
    r1 = xc1 + a1 / 2
    t1 = yc1 - b1 / 2
    d1 = yc1 + b1 / 2

    l2 = xc2 - a2 / 2
    r2 = xc2 + a2 / 2
    t2 = yc2 - b2 / 2
    d2 = yc2 + b2 / 2

    x1=max(l1,l2)
    x2=min(r1,r2)
    y1=max(t1,t2)
    y2=min(d1,d2)
    if x2>x1 and y2>y1:
        # print("same area=",(x2-x1)*(y2-y1))
        return (x2-x1)*(y2-y1)
    return 0

def calc_line(x,y):
    # print("x=",x,"y=",y,",在白线的")
    # if 7*x+y-1680>=0:
        # print("下")
    # else:
        # print("上")
    return 7*x+y-1680>=0

def yolo(video_path):
    ID=0
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
        det = non_max_suppression(pred, conf_thres=0.10, iou_thres=0.45)[0]

        # Rescale boxes from img size to original size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

        # Annotator for drawing boxes
        annotator = Annotator(frame, line_width=1, example=str('class_names'))

        y1, y2 = 0, 240
        x1 = int(-7 * y1 + 1680)
        x2 = int(-7 * y2 + 1680)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        save_path = os.path.join(save_dir, (f'frame_{num}') )
        txt_path = os.path.join(save_dir, (f'frame_{num}') )

        now_mess=[]
        if len(det):
            # Process detections
            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            for *xyxy, conf, cls in det:
                c = int(cls)
                if c not in [2, 5, 7]:
                    continue
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                if xywh[1] < 0.45:
                    continue

                # tracker
                y,x=de_normalize(xywh[0], xywh[1])
                b,a=de_normalize(xywh[2],xywh[3])
                now_car = [x, y, a, b, 0, 0]
                # print(f"search {max(0,num-1-T)}-{num-1}")
                # print("mess=")
                # print(mess)
                min_t,min_i,max_area=0,0,0

                # find who 4 != 0
                for t in range(max(0,num-1-T),num-1):
                    for i in range(len(mess[t])):
                        # print("t=",t,"i=",i)
                        old_car=mess[t][i]
                        if old_car[5] == True or old_car[4] == 0:
                            continue
                        if superposition(x,y,a,b,old_car[0],old_car[1],old_car[2],old_car[3]) >= max_area:
                            min_t,min_i,max_area = t,i,superposition(x,y,a,b,old_car[0],old_car[1],old_car[2],old_car[3])
                if max_area <= Area:
                    for t in range(max(0,num-1-T),num-1):
                        for i in range(len(mess[t])):
                            # print("t=",t,"i=",i)
                            old_car=mess[t][i]
                            if old_car[5] == True:
                                continue
                            if superposition(x,y,a,b,old_car[0],old_car[1],old_car[2],old_car[3]) >= max_area:
                                min_t,min_i,max_area = t,i,superposition(x,y,a,b,old_car[0],old_car[1],old_car[2],old_car[3])

                if max_area >= Area:
                    old_car = mess[min_t][min_i]
                    # print("choose ",old_car)
                    if old_car[4]==0:
                        ID+=1
                        mess[min_t][min_i][4]=now_car[4]=ID
                    else:
                        now_car[4]=old_car[4]
                    mess[min_t][min_i][5]=True
                    # cross white line
                    if (now_car[4] not in cross) and (calc_line(old_car[0],old_car[1]) != calc_line(x,y)):
                        cross.add(now_car[4])


                now_mess.append(now_car)

                def hex_to_bgr(hex_color):
                    """将十六进制颜色字符串转换为RGB元组"""
                    hex_color = hex_color.lstrip('#')  # 移除开头的'#'（如果有）
                    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))

                label = f'car {now_car[4]}'
                label_color = hex_to_bgr("FF0000")
                if calc_line(now_car[0],now_car[1]):
                    label_color = hex_to_bgr("00CCFF")
                if now_car[4] in cross:
                    label_color = hex_to_bgr("000000")
                # annotator.box_label(xyxy, label, color=colors(c, True))
                annotator.box_label(xyxy, label, color=label_color)

                # Write to file

                line = (cls, *xywh, conf)
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')


                # save_one_box(xyxy, img, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
        # print(now_mess)
        mess.append(now_mess)
        print("frame = ", num, "ID =", ID,"Cross = ", len(cross))
        # Show frame
        cv2.imshow('YOLO Detection', annotator.result())
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
            break
        # Save results (image and cross message)
        cv2.imwrite(save_path+".png", annotator.result())
        with open(os.path.join(save_dir,'cross.txt'), 'a') as fc:
            fc.write(str(num) + " " + str(len(cross)) + "\n")

    cap.release()
    cv2.destroyAllWindows()

avi_path="show/video_process.avi"
now_name=input("请输入这一次任务的名字，不要有空格")
save_dir+=now_name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


yolo(avi_path)
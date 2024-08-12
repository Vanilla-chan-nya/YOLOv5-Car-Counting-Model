import os

save_dir = "yolo_"

# 视频尺寸
video_width = 352
video_height = 288
video_len = 7644
# 读取txt文件中的坐标结果
def read_detection_results(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    detections = []
    for line in lines:
        data = line.strip().split()
        class_id, x_center, y_center, width, height, confidence = map(float, data)
        detections.append((x_center, y_center, width, height))

    return detections

# 反归一化坐标值
def de_normalize(detection):
    x_center, y_center, width, height = detection

    # 反归一化计算真实坐标值
    x_center *= video_width
    y_center *= video_height
    width *= video_width
    height *= video_height

    # 计算左上角和右下角坐标
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # 输出真实坐标信息
    print(f"Bounding Box: ({x1}, {y1}) - ({x2}, {y2})")
    exit()
    return (x1+x2)/2, y2


yolo_dir=save_dir+input("请输入已经完成的任务名，不要有空格")

# 计算该目录下以.txt结尾的文件的总数
# txt_files = [f for f in os.listdir(yolo_dir) if f.endswith(".txt")]
# txt_num = len(txt_files)

detections=[]
for i in range(video_len):
    txt_file = os.path.join(yolo_dir, "frame_"+str(i+1)+".txt")
    if(os.path.exists(txt_file) == False):
        detections.append([])
    else:
        detections.append(read_detection_results(txt_file))
    print(f"第{i+1}帧识别到{len(detections[i])}辆汽车")

down_detections=[]

for i in range(video_len):
    now=[]
    for x in detections[i]:
        now.append(de_normalize(x))
    down_detections.append(now)
print(down_detections[0])




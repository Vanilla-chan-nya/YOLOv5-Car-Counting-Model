import cv2

# 视频源
cap = cv2.VideoCapture('show/video_process.avi')

# 从视频中读取第一帧
ret, frame = cap.read()

# 选择ROI (Region of Interest)
bbox = cv2.selectROI(frame, False)

# 初始化CSRT跟踪器
tracker = cv2.legacy.TrackerCSRT_create()

tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 更新跟踪器
    success, bbox = tracker.update(frame)

    if success:
        # 绘制跟踪的对象
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

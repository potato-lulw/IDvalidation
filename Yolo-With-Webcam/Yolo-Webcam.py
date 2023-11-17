from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0) # webcam driver number
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../Videos/bikes.mp4")



classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

model = YOLO("../Yolo-Weights/yolov8l.pt")

while True:
    success, img = cap.read()
    result = model(img, stream=True)

    for r in result:
        boxes = r.boxes
        for box in boxes:

        # for the outer rectangle
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)


            # w, h = x2-x1, y2-y1
            # bbox = int(x1), int(y1), int(w), int(h)
            # cvzone.cornerRect(img, bbox)


        # for printing the confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # print(conf)


        # for printing the className
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f"{conf}, {classNames[cls]}", (max(0, x1), max(20, y1)), scale=1, thickness=1)
            # default thickness=3
    cv2.imshow("Image", img)
    cv2.waitKey(1)

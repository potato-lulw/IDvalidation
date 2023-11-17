import tkinter as tk
from tkinter import filedialog
from ttkthemes import ThemedStyle
from ultralytics import YOLO
import cv2
import cvzone
import math

# Create the main application window
app = tk.Tk()
app.title("Object Detection")
style = ThemedStyle(app)
style.set_theme("arc")

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

# Function to open a video file
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    if file_path:
        start_detection(file_path)


# Function to start object detection
def start_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    model = YOLO("../Yolo-Weights/yolov8m.pt")

    while True:
        success, img = cap.read()
        if not success:
            break

        result = model(img, stream=True)

        for r in result:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f"{conf}, {classNames[cls]}", (max(0, x1), max(20, y1)), scale=1, thickness=1)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create UI elements
label = tk.Label(app, text="Choose a video file or use webcam for object detection", font=("Helvetica", 14))
label.pack(pady=10)

video_button = tk.Button(app, text="Open Video File", command=open_file, font=("Helvetica", 12))
video_button.pack(pady=5)

webcam_button = tk.Button(app, text="Use Webcam", command=lambda: start_detection(0), font=("Helvetica", 12))
webcam_button.pack(pady=5)


app.mainloop()
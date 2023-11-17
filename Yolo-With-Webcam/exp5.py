import tkinter as tk
from tkinter import filedialog
from ttkthemes import ThemedStyle
import cv2
import math
from pyzbar.pyzbar import decode
from ultralytics import YOLO

app = tk.Tk()
app.title("ID Card Object Detection")
style = ThemedStyle(app)
style.set_theme("arc")

class_names = ["Stamp", "Sign", "Logo", "Profile", "ID"]
class_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]

def draw_result_text(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def start_detection(image):
    model = YOLO("../Yolo-Weights/ID.pt")

    result = model(image)

    original_height, original_width = image.shape[:2]

    detected_classes = {class_name: False for class_name in class_names}

    for r in result:
        boxes = r.boxes

        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            scale_x = original_width / image.shape[1]
            scale_y = original_height / image.shape[0]
            x1_resized = int(x1 * scale_x)
            y1_resized = int(y1 * scale_y)

            cls = int(box.cls[0])
            class_name = class_names[cls]
            class_color = class_colors[cls]

            text = f"{class_name}: {conf}"

            text_scale = 1.5
            text_thickness = 2

            cv2.rectangle(image, (x1_resized, y1_resized), (x2, y2), class_color, 3)
            cv2.putText(image, text, (max(0, x1_resized), max(20, y1_resized)), cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                        class_color, text_thickness)

            detected_classes[class_name] = True

    all_classes_detected = all(detected_classes.values())

    result_text = "Valid" if all_classes_detected else "Invalid"

    # Draw the top rectangle indicating Valid or Invalid
    draw_result_text(image, result_text)

    cv2.imshow("ID Card Detection", image)

def process_webcam_feed():
    cap = cv2.VideoCapture(1)  # 0 is the default camera (you can change it if you have multiple cameras)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        start_detection(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

label = tk.Label(app, text="Press 'q' to exit the webcam feed", font=("Helvetica", 14))
label.pack(pady=10)

webcam_button = tk.Button(app, text="Start Webcam", command=process_webcam_feed, font=("Helvetica", 12))
webcam_button.pack(pady=5)

app.mainloop()

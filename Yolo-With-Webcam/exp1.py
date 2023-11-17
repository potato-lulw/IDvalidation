import tkinter as tk
from tkinter import filedialog
from ttkthemes import ThemedStyle
import cv2
import cvzone
import math
from ultralytics import YOLO

app = tk.Tk()
app.title("ID Card Object Detection")
style = ThemedStyle(app)
style.set_theme("arc")

class_names = ["Stamp", "Sign", "Logo", "Profile", "ID"]
class_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]


def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        start_detection(file_path)


def start_detection(image_path):
    img = cv2.imread(image_path)
    model = YOLO("../Yolo-Weights/ID.pt")

    result = model(img)

    original_height, original_width = img.shape[:2]

    detected_classes = {class_name: False for class_name in class_names}

    for r in result:
        boxes = r.boxes

        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            scale_x = original_width / img.shape[1]
            scale_y = original_height / img.shape[0]
            x1_resized = int(x1 * scale_x)
            y1_resized = int(y1 * scale_y)

            cls = int(box.cls[0])
            class_name = class_names[cls]
            class_color = class_colors[cls]

            text = f"{class_name}: {conf}"

            text_scale = 1.5
            text_thickness = 2

            cv2.rectangle(img, (x1_resized, y1_resized), (x2, y2), class_color, 3)

            cv2.putText(img, text, (max(0, x1_resized), max(20, y1_resized)), cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                        class_color, text_thickness)

            detected_classes[class_name] = True

    all_classes_detected = all(detected_classes.values())

    result_text = "Valid" if all_classes_detected else "Invalid"
    cv2.rectangle(img, (0, 0), (original_width, 50), (0, 0, 0), -1)
    cv2.putText(img, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    new_width = 800
    scale_factor = new_width / img.shape[1]
    new_height = int(img.shape[0] * scale_factor)
    img = cv2.resize(img, (new_width, new_height))

    cv2.imshow("ID Card Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


label = tk.Label(app, text="Choose an image file for ID card detection", font=("Helvetica", 14))
label.pack(pady=10)

image_button = tk.Button(app, text="Open Image File", command=open_image, font=("Helvetica", 12))
image_button.pack(pady=5)

app.mainloop()

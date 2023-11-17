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


def start_detection():
    cap = cv2.VideoCapture(1)  # Open the default camera (you can specify a different camera index if needed)

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam

        if not ret:
            break

        # Perform object detection on the frame
        detected_classes = detect_objects(frame)

        # Check if all classes have been detected
        all_classes_detected = all(detected_classes.values())

        # Create a text box with "Valid" or "Invalid" at the top of the frame
        result_text = "Valid" if all_classes_detected else "Invalid"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)  # Black background for the text box
        cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame with detection results
        cv2.imshow("ID Card Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_objects(frame):
    model = YOLO("../Yolo-Weights/ID.pt")  # Replace with the actual path

    original_height, original_width = frame.shape[:2]

    detected_classes = {class_name: False for class_name in class_names}

    result = model(frame)

    for r in result:
        boxes = r.boxes

        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf < 0.5:
                continue  # Skip low-confidence detections
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate resized coordinates while keeping aspect ratio
            scale_x = original_width / frame.shape[1]
            scale_y = original_height / frame.shape[0]
            x1_resized = int(x1 * scale_x)
            y1_resized = int(y1 * scale_y)

            cls = int(box.cls[0])
            class_name = class_names[cls]
            class_color = class_colors[cls]

            # Mark the class as detected
            detected_classes[class_name] = True

            # Draw rectangle with class-specific color
            cv2.rectangle(frame, (x1_resized, y1_resized), (x2, y2), class_color, 3)

            # Display text with class-specific color
            text = f"{class_name}: {conf}"
            text_scale = 1.5
            text_thickness = 2
            cv2.putText(frame, text, (max(0, x1_resized), max(20, y1_resized)), cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                        class_color, text_thickness)

    return detected_classes


label = tk.Label(app, text="Press 'q' to exit the webcam view", font=("Helvetica", 12))
label.pack(pady=10)

start_button = tk.Button(app, text="Start Webcam Detection", command=start_detection, font=("Helvetica", 12))
start_button.pack(pady=5)

app.mainloop()

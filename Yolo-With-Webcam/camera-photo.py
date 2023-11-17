import tkinter as tk
from ttkthemes import ThemedStyle
import cv2
import cvzone
import math
from ultralytics import YOLO

app = tk.Tk()
app.title("Live ID Validation")
style = ThemedStyle(app)
style.set_theme("arc")

class_names = ["Stamp", "Sign", "Logo", "Profile", "ID"]
class_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]

# Initialize the camera
cap = cv2.VideoCapture(0)  # camera driver number


def start_detection():
    while True:
        ret, frame = cap.read()
        original_height, original_width = frame.shape[:2]
        model = YOLO("../Yolo-Weights/ID.pt")

        result = model(frame)
        detected_classes = {class_name: False for class_name in class_names}

        for r in result:
            boxes = r.boxes

            for box in boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                if conf < 0.5:
                    continue  # Skip detection below the conf value 0.5
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # scale_x = original_width / img.shape[1]
                # scale_y = original_height / img.shape[0]
                # x1_resized = int(x1 * scale_x)
                # y1_resized = int(y1 * scale_y)

                cls = int(box.cls[0])
                class_name = class_names[cls]
                class_color = class_colors[cls]
                text = f"{class_name}: {conf}"

                text_scale = 1.5
                text_thickness = 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), class_color, 3)

                # cvzone.putTextRect(frame, f"{class_name}", (max(0, x1), max(20, y1)), scale=1, thickness=1)
                cv2.putText(frame, text, (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                            class_color, text_thickness)

                detected_classes[class_name] = True

        all_classes_detected = all(detected_classes.values())
        result_text = "Valid" if all_classes_detected else "Invalid"
        cv2.rectangle(frame, (0, 0), (original_width, 50), (0, 0, 0), -1)
        cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Live ID Validation", frame)
        new_width = 800
        scale_factor = new_width / frame.shape[1]
        new_height = int(frame.shape[0] * scale_factor)
        img = cv2.resize(frame, (new_width, new_height))
        cv2.imshow("ID Card Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()


start_detection()

app.mainloop()

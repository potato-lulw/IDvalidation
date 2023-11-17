# Define currency class labels and their corresponding values
import math
import tkinter as tk
from tkinter import filedialog

from networkx.utils import open_file
from ttkthemes import ThemedStyle
import cv2
import cvzone
from ultralytics import YOLO



app = tk.Tk()
app.title("Object Detection")
style = ThemedStyle(app)
style.set_theme("arc")

currency_values = {
    "Rs10": 10,
    "Rs20": 20,
    "Rs50": 50,
    "Rs100": 100,
    "Rs200": 200,
    "Rs500": 500,
    "Rs2000": 2000
}

classNames = ["Rs10", "Rs20", "Rs50", "Rs100", "Rs200", "Rs500", "Rs2000"]

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    if file_path:
        start_detection(file_path)


def start_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO("../Yolo-Weights/best200on350.pt")  # Replace with the actual path

    total_sum = 0  # Initialize the total sum

    while True:
        # ... (other code remains the same)
        success, img = cap.read()
        if not success:
            break

        result = model(img, stream=True)
        for r in result:
            boxes = r.boxes
            for box in boxes:
                # ... (other code remains the same)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                if class_name in currency_values:
                    currency_value = currency_values[class_name]
                    total_sum += currency_value  # Add currency value to total sum

                cvzone.putTextRect(img, f"{conf}, {class_name}", (max(0, x1), max(20, y1)), scale=1, thickness=1)

        # Display total sum on the image
        cvzone.putTextRect(img, f"Total: Rs{total_sum}", (10, 50), scale=2, thickness=2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # ...
    cap.release()
    cv2.destroyAllWindows()

label = tk.Label(app, text="Choose a video file or use webcam for object detection", font=("Helvetica", 14))
label.pack(pady=10)

video_button = tk.Button(app, text="Open Video File", command=open_file, font=("Helvetica", 12))
video_button.pack(pady=5)

webcam_button = tk.Button(app, text="Use Webcam", command=lambda: start_detection(0), font=("Helvetica", 12))
webcam_button.pack(pady=5)


app.mainloop()
# ...

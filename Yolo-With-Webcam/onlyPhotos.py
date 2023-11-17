import tkinter as tk
from tkinter import filedialog
from ttkthemes import ThemedStyle
import cv2
import cvzone
import math
from ultralytics import YOLO

app = tk.Tk()
app.title("Image Object Detection")
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



def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        start_detection(file_path)

def start_detection(image_path):
    img = cv2.imread(image_path)
    model = YOLO("../Yolo-Weights/best200on350.pt")  # Replace with the actual path

    total_sum = 0  # Initialize the total sum

    result = model(img)

    for r in result:
        boxes = r.boxes

        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf < 0.5:
                continue  # Skip low-confidence detections
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name in currency_values:
                currency_value = currency_values[class_name]
                total_sum += currency_value  # Add currency value to total sum

            cvzone.putTextRect(img, f"{class_name}", (max(0, x1), max(20, y1)), scale=1, thickness=1)

    # Display total sum on the image
    cvzone.putTextRect(img, f"Total: Rs{total_sum}", (10, 50), scale=2, thickness=2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

label = tk.Label(app, text="Choose an image file for object detection", font=("Helvetica", 14))
label.pack(pady=10)

image_button = tk.Button(app, text="Open Image File", command=open_image, font=("Helvetica", 12))
image_button.pack(pady=5)

app.mainloop()

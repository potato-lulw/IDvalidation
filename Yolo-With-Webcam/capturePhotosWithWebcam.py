import tkinter as tk
from tkinter import messagebox
from ttkthemes import ThemedStyle
import cv2
import cvzone
import math
from ultralytics import YOLO

app = tk.Tk()
app.title("Capture and Detect Objects")
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

# Initialize the camera
cap = cv2.VideoCapture(0)

# Number of images to capture
num_images_to_capture = 5
current_image_index = 0

def capture_image():
    global current_image_index
    if current_image_index < num_images_to_capture:
        ret, frame = cap.read()
        if ret:
            image_filename = f"captured_image_{current_image_index + 1}.jpg"
            cv2.imwrite(image_filename, frame)
            current_image_index += 1
            if current_image_index == num_images_to_capture:
                messagebox.showinfo("Capture Complete", "Images captured successfully.")
                start_detection()
            else:
                messagebox.showinfo("Capture Image", f"Image {current_image_index} captured.")
    else:
        messagebox.showinfo("Capture Complete", "You have already captured the maximum number of images.")

def start_detection():
    for i in range(num_images_to_capture):
        image_filename = f"captured_image_{i + 1}.jpg"
        img = cv2.imread(image_filename)

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

label = tk.Label(app, text="Capture 5 images and detect objects", font=("Helvetica", 14))
label.pack(pady=10)

capture_button = tk.Button(app, text="Capture Image", command=capture_image, font=("Helvetica", 12))
capture_button.pack(pady=5)

app.mainloop()
    
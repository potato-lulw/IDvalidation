import os
import cv2

# Function to resize an image and update bounding boxes in .txt format
def resize_and_update_annotations(input_image_path, input_annotation_path, output_image_path, output_annotation_path, target_size):
    # Read the image
    image = cv2.imread(input_image_path)
    height, width, _ = image.shape

    # Calculate the scaling factors
    x_scale = target_size[0] / width
    y_scale = target_size[1] / height

    # Read and update bounding boxes in the .txt annotation file
    with open(input_annotation_path, 'r') as infile, open(output_annotation_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, box_width, box_height = map(float, parts)
                x_center_new = x_center * x_scale
                y_center_new = y_center * y_scale
                box_width_new = box_width * x_scale
                box_height_new = box_height * y_scale
                # Write the updated annotation to the output file
                outfile.write(f"{int(class_id)} {x_center_new} {y_center_new} {box_width_new} {box_height_new}\n")

    # Resize and save the image to the output file
    resized_image = cv2.resize(image, target_size)
    cv2.imwrite(output_image_path, resized_image)

# Define input and output directories
input_dir = "C:/Users/ompat/OneDrive/Desktop/data"
output_dir = "C:/Users/ompat/OneDrive/Desktop/newdata"

# Target size for resizing
target_size = (224, 224)

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through images and annotations in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        input_image_path = os.path.join(input_dir, filename)
        input_annotation_path = os.path.join(input_dir, filename.replace(".jpg", ".txt"))
        output_image_path = os.path.join(output_dir, filename)
        output_annotation_path = os.path.join(output_dir, filename.replace(".jpg", ".txt"))

        resize_and_update_annotations(input_image_path, input_annotation_path, output_image_path, output_annotation_path, target_size)




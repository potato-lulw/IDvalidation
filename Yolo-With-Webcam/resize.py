import os
from PIL import Image

# Input directory containing the original images
input_directory = 'C:/Users/ompat/OneDrive/Desktop/data'

# Output directory where resized images will be saved
output_directory = 'C:/Users/ompat/OneDrive/Desktop/newdata/Images2'

# Desired width and height for resizing
new_width = 800
new_height = 800

# Ensure the output directory exists, or create it if necessary
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through all files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):  # Add more image formats if needed
        # Open the image using Pillow
        img = Image.open(os.path.join(input_directory, filename))

        # Resize the image using the "NEAREST" resampling filter (no anti-aliasing)
        img = img.resize((new_width, new_height), Image.NEAREST)

        # Save the resized image to the output directory
        img.save(os.path.join(output_directory, filename))

print("Images resized without anti-aliasing and saved to", output_directory)

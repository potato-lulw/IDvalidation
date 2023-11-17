import os
import cv2
import numpy as np
import tensorflow as tf

import sys

sys.path.append('C:/Users/ompat/PycharmProjects/IDvalidation/models/research')
from object_detection.utils import dataset_util

# Define your TFRecord output directory
tfrecord_output_dir = "C:/Users/ompat/OneDrive/Desktop/newdata"

# Create output directory if it doesn't exist
if not os.path.exists(tfrecord_output_dir):
    os.makedirs(tfrecord_output_dir)


# Function to create a TFRecord example
def create_tf_example(image_path, annotation_path):
    # Read the image
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    # Read the annotation file and parse bounding box coordinates
    with open(annotation_path, 'r') as annotation_file:
        lines = annotation_file.readlines()

    # Extract bounding box information
    xmins, ymins, xmaxs, ymaxs, class_ids = [], [], [], [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id, x_center, y_center, box_width, box_height = map(float, parts)
            x_min = x_center - (box_width / 2.0)
            x_max = x_center + (box_width / 2.0)
            y_min = y_center - (box_height / 2.0)
            y_max = y_center + (box_height / 2.0)

            class_ids.append(int(class_id))
            xmins.append(x_min)
            ymins.append(y_min)
            xmaxs.append(x_max)
            ymaxs.append(y_max)

    # Create a TFExample
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/label': dataset_util.int64_list_feature(class_ids),
    }))
    return tf_example


# Define your input directory with resized images and annotations
input_dir = "C:/Users/ompat/OneDrive/Desktop/newdata"

# Loop through images and annotations
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        annotation_path = os.path.join(input_dir, filename.replace(".jpg", ".txt"))
        tfrecord_path = os.path.join(tfrecord_output_dir, filename.replace(".jpg", ".record"))

        tf_example = create_tf_example(image_path, annotation_path)

        # Write TFExample to TFRecord file
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            writer.write(tf_example.SerializeToString())

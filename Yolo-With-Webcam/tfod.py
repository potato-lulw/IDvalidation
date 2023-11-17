import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the directory containing the TFRecord files
tfrecord_dir = "C:/Users/ompat/OneDrive/Desktop/newdata"

# Define the image size and batch size
image_size = (224, 224)
batch_size = 32

# Function to parse TFRecord files and decode images
def _parse_tfrecord_fn(example):
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/object/class/label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(example["image/encoded"], channels=3)
    image = tf.image.resize(image, image_size) / 255.0
    label = example["image/object/class/label"]
    return image, label

# Create a dataset from TFRecord files
tfrecord_files = [os.path.join(tfrecord_dir, file) for file in os.listdir(tfrecord_dir) if file.endswith(".record")]
dataset = tf.data.TFRecordDataset(tfrecord_files)
dataset = dataset.map(_parse_tfrecord_fn)

# Split the dataset into training and validation sets
validation_split = 0.2
num_samples = sum(1 for _ in dataset)
num_validation_samples = int(validation_split * num_samples)
num_training_samples = num_samples - num_validation_samples

validation_dataset = dataset.take(num_validation_samples)
training_dataset = dataset.skip(num_validation_samples)

# Create data augmentation for the training dataset
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.1,
)

# Create and compile a CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Change the number of units based on your classification task
])

cnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Change the loss function based on your task
                  metrics=['accuracy'])

# Train the CNN model
history_cnn = cnn_model.fit(datagen.flow(training_dataset.batch(batch_size)),
                            steps_per_epoch=num_training_samples // batch_size,
                            epochs=10,  # Adjust the number of epochs
                            validation_data=validation_dataset.batch(batch_size),
                            validation_steps=num_validation_samples // batch_size)

# Create and compile an ANN model
ann_model = models.Sequential([
    layers.Flatten(input_shape=(224, 224, 3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Change the number of units based on your classification task
])

ann_model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Change the loss function based on your task
                  metrics=['accuracy'])

# Train the ANN model
history_ann = ann_model.fit(training_dataset.batch(batch_size),
                            epochs=10,  # Adjust the number of epochs
                            validation_data=validation_dataset.batch(batch_size))

# Evaluate the models or save them for future use

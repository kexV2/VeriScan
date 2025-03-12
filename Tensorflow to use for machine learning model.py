import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Define dataset path (script is inside the dataset folder)
DATASET_PATH = os.getcwd()  # Automatically uses the current directory

# Define image size and batch size for XceptionNet
IMG_SIZE = (299, 299)  
BATCH_SIZE = 32  

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Load validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Print class names to verify the dataset structure
print("Class Names:", train_ds.class_names)

# Display sample images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis("off")
    plt.show()


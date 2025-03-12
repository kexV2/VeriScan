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

# Get class names before any transformations
class_names = train_ds.class_names

# Print class names to verify the dataset structure
print("Class Names:", class_names)

# Normalize the image pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Cache and prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Load the Xception model pre-trained on ImageNet
base_model = tf.keras.applications.Xception(
    weights='imagenet',  # Load pre-trained weights from ImageNet
    input_shape=(299, 299, 3),  # Image size for Xception
    include_top=False  # Exclude the top classification layer
)

# Freeze the base model to prevent training it
base_model.trainable = False

# Build the full model by adding custom layers on top of Xception
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),  # Global pooling to reduce output size
    tf.keras.layers.Dense(1024, activation='relu'),  # Fully connected layer
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_ds,
    epochs=10,  # Change as needed
    validation_data=val_ds
)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


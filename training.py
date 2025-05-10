import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import shutil

# Define paths
CURRENT_MODEL_PATH = "deepfake_model_expanded.h5"  # Path to your current model
RAW_DATASET_PATH = "C:/Users/Dylan Keogh/Desktop/real_and_fake_face"
PROCESSED_DATASET_PATH = "processed_dataset"  # Temporary path for reorganized data
FINAL_MODEL_PATH = "deepfake_model_nearly.h5"  # Path to save the final model

# Define image size and batch size
IMG_SIZE = (299, 299)  # Keep the same image size for XceptionNet
BATCH_SIZE = 16

# Function to reorganize dataset if needed
def reorganize_dataset():
    """
    Reorganizes the dataset from training_fake/training_real structure
    to Train/Validation/Test with fake_images and real_images subfolders
    """
    # Create directory structure
    os.makedirs(PROCESSED_DATASET_PATH, exist_ok=True)
    
    # Create train/val/test splits with fake/real subfolders
    for split in ["Train", "Validation", "Test"]:
        os.makedirs(os.path.join(PROCESSED_DATASET_PATH, split, "fake_images"), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATASET_PATH, split, "real_images"), exist_ok=True)
    
    # Check if source folders exist
    fake_path = os.path.join(RAW_DATASET_PATH, "training_fake")
    real_path = os.path.join(RAW_DATASET_PATH, "training_real")
    
    if not os.path.exists(fake_path) or not os.path.exists(real_path):
        raise FileNotFoundError(f"Required folders not found. Please check paths: {fake_path}, {real_path}")
    
    # Count files
    fake_files = [f for f in os.listdir(fake_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    real_files = [f for f in os.listdir(real_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(fake_files)} fake images and {len(real_files)} real images")
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15  # remainder
    
    # Function to copy files based on split
    def copy_files(source_dir, files, category):
        # Shuffle files
        np.random.shuffle(files)
        
        # Determine split indices
        train_idx = int(len(files) * train_ratio)
        val_idx = train_idx + int(len(files) * val_ratio)
        
        # Split files
        train_files = files[:train_idx]
        val_files = files[train_idx:val_idx]
        test_files = files[val_idx:]
        
        # Copy files to appropriate directories
        for f in train_files:
            shutil.copy2(
                os.path.join(source_dir, f),
                os.path.join(PROCESSED_DATASET_PATH, "Train", f"{category}_images", f)
            )
        
        for f in val_files:
            shutil.copy2(
                os.path.join(source_dir, f),
                os.path.join(PROCESSED_DATASET_PATH, "Validation", f"{category}_images", f)
            )
        
        for f in test_files:
            shutil.copy2(
                os.path.join(source_dir, f),
                os.path.join(PROCESSED_DATASET_PATH, "Test", f"{category}_images", f)
            )
        
        print(f"Copied {len(train_files)} {category} images to Train")
        print(f"Copied {len(val_files)} {category} images to Validation")
        print(f"Copied {len(test_files)} {category} images to Test")
    
    # Copy files
    copy_files(fake_path, fake_files, "fake")
    copy_files(real_path, real_files, "real")
    
    return PROCESSED_DATASET_PATH

# Reorganize the dataset
print("Reorganizing dataset to standard structure...")
FINAL_DATASET_PATH = reorganize_dataset()
print(f"Dataset reorganized at: {FINAL_DATASET_PATH}")

# Load the current model
try:
    print(f"Loading model from {CURRENT_MODEL_PATH}...")
    current_model = tf.keras.models.load_model(CURRENT_MODEL_PATH)
    print("Successfully loaded current model.")
    current_model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise Exception("Failed to load the current model. Please check the path and file integrity.")

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s) for training")
    except Exception as e:
        print(f"Error configuring GPU: {str(e)}")

# Define consistent class names
CLASS_NAMES = ['fake_images', 'real_images']  # Match the folder names exactly

# Load training dataset
print(f"Loading training data from {FINAL_DATASET_PATH}...")
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(FINAL_DATASET_PATH, "Train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=True,
    class_names=CLASS_NAMES
)

# Load validation dataset
raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(FINAL_DATASET_PATH, "Validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=CLASS_NAMES
)

# Load test dataset
raw_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(FINAL_DATASET_PATH, "Test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=CLASS_NAMES
)

# Verify class names
print("Train Class Names:", raw_train_ds.class_names)
print("Validation Class Names:", raw_val_ds.class_names)
print("Test Class Names:", raw_test_ds.class_names)

# Check sample images to ensure they're properly loaded
print("\nChecking sample images:")
for images, labels in raw_train_ds.take(1):
    for i in range(min(3, len(images))):
        img = images[i].numpy()
        print(f"Image {i} shape: {img.shape}")
        print(f"Min value: {img.min()}, Max value: {img.max()}")
        print(f"Label: {raw_train_ds.class_names[labels[i]]}")

# Normalize the image pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply preprocessing to all datasets
final_train_ds = raw_train_ds.map(lambda x, y: (normalization_layer(x), y))
final_val_ds = raw_val_ds.map(lambda x, y: (normalization_layer(x), y))
final_test_ds = raw_test_ds.map(lambda x, y: (normalization_layer(x), y))

# Cache and prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
final_train_ds = final_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
final_val_ds = final_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
final_test_ds = final_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define data augmentation for training
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# Apply data augmentation to the training dataset
augmented_train_ds = final_train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

# Define callbacks for training
callbacks = [
    # Save the best model based on validation accuracy
    tf.keras.callbacks.ModelCheckpoint(
        filepath=FINAL_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Early stopping to prevent overfitting
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when performance plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    # TensorBoard logging
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs_final',
        histogram_freq=1
    )
]

# Compile the model with a lower learning rate for fine-tuning
current_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Lower learning rate for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Show example images from final dataset
plt.figure(figsize=(10, 10))
for images, labels in raw_train_ds.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(raw_train_ds.class_names[labels[i]])
        plt.axis("off")
    plt.savefig('final_dataset_samples.png')
    plt.show()

# Continue training the model on new data
print("Starting final fine-tuning on new dataset...")
history = current_model.fit(
    augmented_train_ds,  # Use augmented dataset for training
    epochs=15,  # Adjust as needed
    validation_data=final_val_ds,
    callbacks=callbacks
)

# Save the final model (in case callbacks didn't save it)
current_model.save(FINAL_MODEL_PATH)
print(f"Final model saved to {FINAL_MODEL_PATH}")

# Evaluate the model on the test set
test_loss, test_acc = current_model.evaluate(final_test_ds)
print(f"Test Accuracy on final data: {test_acc * 100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('final_training_history.png')
plt.show()

# Create a confusion matrix for deeper insight
# Collect predictions
y_true = []
y_pred = []

# Use batches from the test dataset
for x, y in final_test_ds:
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(current_model.predict(x), axis=1))

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, 
            yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Final Model Confusion Matrix')
plt.savefig('final_confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

print("Training complete!")
print(f"Final model saved to: {FINAL_MODEL_PATH}")

# Check if the model file was created successfully
if os.path.exists(FINAL_MODEL_PATH):
    print(f"Verified: Model file created successfully at {FINAL_MODEL_PATH}")
    print(f"File size: {os.path.getsize(FINAL_MODEL_PATH) / (1024*1024):.2f} MB")
else:
    print("Warning: Model file was not created. Check for errors in the training process.")

# Update the model path in veriscan.py if needed
try:
    with open("veriscan.py", "r") as f:
        content = f.read()
    
    # Check if we need to update the model path
    if 'MODEL_PATH = "deepfake_model_expanded.h5"' in content:
        updated_content = content.replace(
            'MODEL_PATH = "deepfake_model_expanded.h5"', 
            'MODEL_PATH = "deepfake_model_final.h5"'
        )
        
        with open("veriscan.py", "w") as f:
            f.write(updated_content)
        
        print("Updated model path in veriscan.py to use the final model.")
except Exception as e:
    print(f"Note: Could not update veriscan.py automatically: {str(e)}")
    print("You may need to manually update the MODEL_PATH in veriscan.py to use your new model.")

# Clean up temporary processed dataset directory if desired
# Uncomment the following lines if you want to remove the temporary files
# print("Cleaning up temporary files...")
# shutil.rmtree(PROCESSED_DATASET_PATH)
# print("Temporary files removed.")
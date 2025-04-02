import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

# Define paths
ORIGINAL_MODEL_PATH = "deepfake_model.h5"  # Path to your existing model
NEW_DATASET_PATH = r"C:\Users\Dylan Keogh\Desktop\more training"  # Path to your new dataset
EXPANDED_MODEL_PATH = "deepfake_model_expanded.h5"  # Path to save the expanded model

# Define image size and batch size for XceptionNet (same as original)
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

# Load the previously trained model
try:
    previous_model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)
    print("Successfully loaded previous model.")
    previous_model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise Exception("Failed to load the original model. Please check the path and file integrity.")

# Load new training dataset (adjusting for your specific folder names)
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(NEW_DATASET_PATH, "Train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=True,
    class_names=['fake_images', 'real_images']  # Specify your exact folder names
)

# Store class names before applying transformations
train_class_names = raw_train_ds.class_names
print("Train Class Names:", train_class_names)

# Load validation dataset with corrected folder name
raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(NEW_DATASET_PATH, "Validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=['Fake', 'Real']
)

# Store class names for validation
val_class_names = raw_val_ds.class_names
print("Validation Class Names:", val_class_names)

# Load test dataset
raw_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(NEW_DATASET_PATH, "Test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=['Fake', 'Real']
)

# Store class names for test
test_class_names = raw_test_ds.class_names
print("Test Class Names:", test_class_names)

# Normalize the image pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply preprocessing to all datasets
new_train_ds = raw_train_ds.map(lambda x, y: (normalization_layer(x), y))
new_val_ds = raw_val_ds.map(lambda x, y: (normalization_layer(x), y))
new_test_ds = raw_test_ds.map(lambda x, y: (normalization_layer(x), y))

# Cache and prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
new_train_ds = new_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
new_val_ds = new_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
new_test_ds = new_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define callbacks for training
callbacks = [
    # Save the best model based on validation accuracy
    tf.keras.callbacks.ModelCheckpoint(
        filepath=EXPANDED_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Early stopping to prevent overfitting
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when performance plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    # TensorBoard logging
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# Compile the model with a lower learning rate for fine-tuning
previous_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display a summary of the model architecture
print("Model Summary:")
previous_model.summary()

# Show example images from new dataset
plt.figure(figsize=(10, 10))
for images, labels in raw_train_ds.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # No need to scale for display, using original images
        plt.title(train_class_names[labels[i]])
        plt.axis("off")
    plt.savefig('new_dataset_samples.png')
    plt.show()

# Continue training the model on new data
print("Starting fine-tuning on new dataset...")
history = previous_model.fit(
    new_train_ds,
    epochs=10,  # Adjust as needed
    validation_data=new_val_ds,
    callbacks=callbacks
)

# Save the final model (in case callbacks didn't save it)
previous_model.save(EXPANDED_MODEL_PATH)
print(f"Expanded model saved to {EXPANDED_MODEL_PATH}")

# Evaluate the model on the test set
test_loss, test_acc = previous_model.evaluate(new_test_ds)
print(f"Test Accuracy on new data: {test_acc * 100:.2f}%")

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
plt.savefig('training_history.png')
plt.show()

# Create a confusion matrix for deeper insight
# Collect predictions
y_true = []
y_pred = []

# Use batches from the test dataset
for x, y in new_test_ds:
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(previous_model.predict(x), axis=1))

# Create confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=test_class_names, 
            yticklabels=test_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_class_names))

print("Training complete!")
print(f"Expanded model saved to: {EXPANDED_MODEL_PATH}")

# Check if the model file was created successfully
if os.path.exists(EXPANDED_MODEL_PATH):
    print(f"Verified: Model file created successfully at {EXPANDED_MODEL_PATH}")
    print(f"File size: {os.path.getsize(EXPANDED_MODEL_PATH) / (1024*1024):.2f} MB")
else:
    print("Warning: Model file was not created. Check for errors in the training process.")
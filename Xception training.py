import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define paths
CURRENT_MODEL_PATH = "deepfake_model_expanded.h5"  # Path to your current model
FINAL_DATASET_PATH = r"C:\Users\Dylan Keogh\Desktop\training_final"  # Path to your final dataset
FINAL_MODEL_PATH = "deepfake_model_final.h5"  # Path to save the final model

# Define image size and batch size (same as previous models)
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

# Load the current model
try:
    print(f"Loading model from {CURRENT_MODEL_PATH}...")
    current_model = tf.keras.models.load_model(CURRENT_MODEL_PATH)
    print("Successfully loaded current model.")
    current_model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise Exception("Failed to load the current model. Please check the path and file integrity.")

# Load final training dataset
print(f"Loading training data from {FINAL_DATASET_PATH}...")
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(FINAL_DATASET_PATH, "Train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=True,
    # Make sure these match your folder names exactly
    class_names=['fake_images', 'real_images']  
)

# Store class names
train_class_names = raw_train_ds.class_names
print("Train Class Names:", train_class_names)

# Load validation dataset
raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(FINAL_DATASET_PATH, "Validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=['Fake', 'Real']  # Adjust these if your folder names are different
)

val_class_names = raw_val_ds.class_names
print("Validation Class Names:", val_class_names)

# Load test dataset
raw_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(FINAL_DATASET_PATH, "Test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=['Fake', 'Real']  # Adjust these if your folder names are different
)

test_class_names = raw_test_ds.class_names
print("Test Class Names:", test_class_names)

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
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Lower learning rate for final tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Show example images from final dataset
plt.figure(figsize=(10, 10))
for images, labels in raw_train_ds.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_class_names[labels[i]])
        plt.axis("off")
    plt.savefig('final_dataset_samples.png')
    plt.show()

# Continue training the model on final data
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
            xticklabels=test_class_names, 
            yticklabels=test_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Final Model Confusion Matrix')
plt.savefig('final_confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_class_names))

# Generate some predictions with Grad-CAM visualization
from grad_cam import get_grad_cam, save_grad_cam_visualization, analyze_heatmap

# Find a suitable layer for visualization
def find_best_layer(model):
    """Find the best layer for visualization in the model"""
    # Try to find a convolutional layer first
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
            
    # Fallback options
    for layer in model.layers:
        if any(name in layer.name.lower() for name in ["conv", "pool", "dense", "global"]):
            return layer.name
            
    # Last resort
    return model.layers[-2].name

# Select a few test images for visualization
visualization_dir = "final_model_visualizations"
os.makedirs(visualization_dir, exist_ok=True)

target_layer = find_best_layer(current_model)
print(f"Using layer {target_layer} for Grad-CAM visualization")

# Create a test folder to store sample test images if it doesn't exist
test_samples_dir = os.path.join(visualization_dir, "test_samples")
os.makedirs(test_samples_dir, exist_ok=True)

# Sample a few test images
count = 0
for images, labels in raw_test_ds.take(2):  # Take 2 batches
    for i in range(min(5, len(images))):  # Take up to 5 images from each batch
        img = images[i].numpy().astype("uint8")
        label = labels[i].numpy()
        
        # Save the test image
        img_path = os.path.join(test_samples_dir, f"test_sample_{count}_{test_class_names[label]}.png")
        plt.imsave(img_path, img)
        
        # Normalize the image
        processed_img = np.expand_dims(img / 255.0, axis=0)
        
        # Get prediction
        pred = current_model.predict(processed_img)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)
        
        # Generate Grad-CAM
        heatmap = get_grad_cam(current_model, processed_img, target_layer, pred_class)
        
        if heatmap is not None:
            # Save visualization
            output_path = os.path.join(visualization_dir, f"gradcam_{count}_{test_class_names[label]}_pred_{test_class_names[pred_class]}_{confidence:.2f}.png")
            save_grad_cam_visualization(img_path, heatmap, output_path)
            
            # Analyze heatmap
            analysis = analyze_heatmap(heatmap)
            print(f"\nImage {count} Analysis:")
            print(f"True label: {test_class_names[label]}")
            print(f"Predicted: {test_class_names[pred_class]} with {confidence:.2f} confidence")
            print(f"Focus regions: {analysis.get('focus_regions', [])}")
            print(f"Scores: {analysis.get('scores', {})}")
        
        count += 1

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
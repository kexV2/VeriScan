import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define paths
CURRENT_MODEL_PATH = "deepfake_model_expanded.h5"  # Path to your current model
FINAL_DATASET_PATH = r"C:\Users\Dylan Keogh\Desktop\LastTime"  # Path to your final dataset
FINAL_MODEL_PATH = "deepfake_last.h5"  # Path to save the final model

# Define image size and batch size (reduced for memory constraints)
IMG_SIZE = (299, 299)
BATCH_SIZE = 16  # Reduced from 32 to avoid memory issues

# Print debug info about system and environment
print("Tensorflow version:", tf.__version__)
print("Python working directory:", os.getcwd())
print("Starting script execution...")

# Configure GPU memory growth early
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s) for training with memory growth enabled")
    except Exception as e:
        print(f"Error configuring GPU: {str(e)}")
else:
    print("No GPUs detected, using CPU")

# Check if dataset directories exist
print(f"Checking dataset directories...")
if not os.path.exists(FINAL_DATASET_PATH):
    print(f"ERROR: Dataset path {FINAL_DATASET_PATH} does not exist!")
    exit(1)
else:
    # Check expected subdirectories
    expected_dirs = ["Train", "Validation", "Test"]
    for dir_name in expected_dirs:
        full_path = os.path.join(FINAL_DATASET_PATH, dir_name)
        if not os.path.exists(full_path):
            print(f"ERROR: Expected directory {full_path} not found!")
            exit(1)
        else:
            print(f"Directory {full_path} exists. Contents: {os.listdir(full_path)}")

# Load the current model
try:
    print(f"Loading model from {CURRENT_MODEL_PATH}...")
    if not os.path.exists(CURRENT_MODEL_PATH):
        print(f"ERROR: Model file {CURRENT_MODEL_PATH} does not exist!")
        exit(1)
    current_model = tf.keras.models.load_model(CURRENT_MODEL_PATH)
    print("Successfully loaded current model.")
    current_model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise Exception("Failed to load the current model. Please check the path and file integrity.")

# Check class names consistency
print("Checking class folder names...")
print(f"Train folders: {os.listdir(os.path.join(FINAL_DATASET_PATH, 'Train'))}")
print(f"Validation folders: {os.listdir(os.path.join(FINAL_DATASET_PATH, 'Validation'))}")
print(f"Test folders: {os.listdir(os.path.join(FINAL_DATASET_PATH, 'Test'))}")

print("Starting to load datasets...")

# Function to safely load a dataset with error handling
def safe_load_dataset(dataset_path, image_size, batch_size, class_names=None):
    try:
        print(f"Loading dataset from {dataset_path}...")
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset path {dataset_path} does not exist!")
            return None
            
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='int',
            shuffle=(dataset_path.endswith('Train')),  # Only shuffle training data
            class_names=class_names
        )
        print(f"Successfully loaded dataset from {dataset_path}")
        print(f"Class names: {ds.class_names}")
        print(f"Dataset shape: {tf.data.experimental.cardinality(ds)} batches of size {batch_size}")
        return ds
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {str(e)}")
        return None

# Load training dataset
raw_train_ds = safe_load_dataset(
    os.path.join(FINAL_DATASET_PATH, "Train"),
    IMG_SIZE,
    BATCH_SIZE
)
if raw_train_ds is None:
    print("Failed to load training dataset. Exiting.")
    exit(1)

# Store class names
train_class_names = raw_train_ds.class_names
print("Train Class Names:", train_class_names)

# Load validation dataset - use the same class names as training if possible
raw_val_ds = safe_load_dataset(
    os.path.join(FINAL_DATASET_PATH, "Validation"),
    IMG_SIZE,
    BATCH_SIZE
)
if raw_val_ds is None:
    print("Failed to load validation dataset. Exiting.")
    exit(1)

val_class_names = raw_val_ds.class_names
print("Validation Class Names:", val_class_names)

# Load test dataset - use the same class names as training if possible
raw_test_ds = safe_load_dataset(
    os.path.join(FINAL_DATASET_PATH, "Test"),
    IMG_SIZE,
    BATCH_SIZE
)
if raw_test_ds is None:
    print("Failed to load test dataset. Exiting.")
    exit(1)

test_class_names = raw_test_ds.class_names
print("Test Class Names:", test_class_names)
print("Test dataset loaded successfully")

# Normalize the image pixel values to [0, 1]
print("Starting normalization...")
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply preprocessing to all datasets - without cache to avoid memory issues
print("Applying normalization to datasets...")
final_train_ds = raw_train_ds.map(lambda x, y: (normalization_layer(x), y))
print("Train dataset normalized")
final_val_ds = raw_val_ds.map(lambda x, y: (normalization_layer(x), y))
print("Validation dataset normalized")
final_test_ds = raw_test_ds.map(lambda x, y: (normalization_layer(x), y))
print("Test dataset normalized")

# Use prefetch only (no cache) to avoid memory issues
AUTOTUNE = tf.data.AUTOTUNE
final_train_ds = final_train_ds.prefetch(buffer_size=AUTOTUNE)
final_val_ds = final_val_ds.prefetch(buffer_size=AUTOTUNE)
final_test_ds = final_test_ds.prefetch(buffer_size=AUTOTUNE)
print("Dataset prefetch configured")

# Define data augmentation for training
print("Setting up data augmentation...")
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# Apply data augmentation to the training dataset
print("Applying data augmentation to training dataset...")
augmented_train_ds = final_train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)
print("Data augmentation applied")

# Define callbacks for training
print("Setting up training callbacks...")
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
print("Compiling model...")
current_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Lower learning rate for final tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Model compiled successfully")

# Show example images from final dataset
print("Generating example images visualization...")
try:
    plt.figure(figsize=(10, 10))
    images_plotted = False
    
    for images, labels in raw_train_ds.take(1):
        for i in range(min(9, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(train_class_names[labels[i]])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig('final_dataset_samples.png')
        images_plotted = True
        plt.close()  # Close to free memory
    
    if images_plotted:
        print("Example images saved to final_dataset_samples.png")
    else:
        print("No images available to plot")
except Exception as e:
    print(f"Error generating example images: {str(e)}")

# Continue training the model on final data
print("Starting final fine-tuning on new dataset...")
try:
    history = current_model.fit(
        augmented_train_ds,  # Use augmented dataset for training
        epochs=15,  # Adjust as needed
        validation_data=final_val_ds,
        callbacks=callbacks,
        verbose=1
    )
    print("Training completed successfully")
except Exception as e:
    print(f"Error during training: {str(e)}")
    import traceback
    traceback.print_exc()
    print("Attempting to save the model anyway...")
    
    # Try to save the model even if training failed
    try:
        current_model.save(FINAL_MODEL_PATH)
        print(f"Model saved to {FINAL_MODEL_PATH} despite training errors")
    except:
        print("Failed to save the model")
    exit(1)

# Save the final model (in case callbacks didn't save it)
print("Saving final model...")
current_model.save(FINAL_MODEL_PATH)
print(f"Final model saved to {FINAL_MODEL_PATH}")

# Evaluate the model on the test set
print("Evaluating model on test set...")
try:
    test_loss, test_acc = current_model.evaluate(final_test_ds)
    print(f"Test Accuracy on final data: {test_acc * 100:.2f}%")
except Exception as e:
    print(f"Error evaluating model: {str(e)}")

# Plot training history
try:
    print("Generating training history plots...")
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
    plt.tight_layout()
    plt.savefig('final_training_history.png')
    plt.close()  # Close to free memory
    print("Training history plots saved to final_training_history.png")
except Exception as e:
    print(f"Error generating training history plots: {str(e)}")

# Create a confusion matrix for deeper insight
print("Generating confusion matrix...")
try:
    # Collect predictions
    y_true = []
    y_pred = []
    
    # Use batches from the test dataset
    print("Making predictions on test data...")
    batch_count = 0
    for x, y in final_test_ds:
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"Processing batch {batch_count}...")
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(current_model.predict(x, verbose=0), axis=1))
    
    # Create confusion matrix
    print(f"Creating confusion matrix with {len(y_true)} samples...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_class_names, 
                yticklabels=test_class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Final Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png')
    plt.close()  # Close to free memory
    print("Confusion matrix saved to final_confusion_matrix.png")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=test_class_names))
except Exception as e:
    print(f"Error generating confusion matrix: {str(e)}")

# Generate some predictions with Grad-CAM visualization
print("Starting Grad-CAM visualization generation...")
try:
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
    print("Processing test images for visualization...")
    for images, labels in raw_test_ds.take(2):  # Take 2 batches
        for i in range(min(5, len(images))):  # Take up to 5 images from each batch
            print(f"Processing visualization for image {count+1}...")
            img = images[i].numpy().astype("uint8")
            label = labels[i].numpy()
            
            # Save the test image
            img_path = os.path.join(test_samples_dir, f"test_sample_{count}_{test_class_names[label]}.png")
            plt.imsave(img_path, img)
            
            # Normalize the image
            processed_img = np.expand_dims(img / 255.0, axis=0)
            
            # Get prediction
            pred = current_model.predict(processed_img, verbose=0)
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
    print(f"Processed {count} images for visualization")
except Exception as e:
    print(f"Error during Grad-CAM visualization: {str(e)}")
    import traceback
    traceback.print_exc()

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
    if os.path.exists("veriscan.py"):
        print("Checking veriscan.py for model path updates...")
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
        else:
            print("No need to update model path in veriscan.py")
    else:
        print("veriscan.py not found in current directory")
except Exception as e:
    print(f"Note: Could not update veriscan.py automatically: {str(e)}")
    print("You may need to manually update the MODEL_PATH in veriscan.py to use your new model.")

print("Script execution completed successfully")
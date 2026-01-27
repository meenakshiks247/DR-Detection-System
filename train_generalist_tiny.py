import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ============================================================
# OPTIMIZATION: Prevent CPU freezing by limiting threads
# ============================================================
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_DIR = r'E:\Major project\DR_Detection_System\dataset_generalist'
BATCH_SIZE = 16
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 10
MODEL_NAME = 'generalist_model.h5'

print("=" * 70)
print("TINY CNN MODEL TRAINING (Custom Lightweight Architecture)")
print("=" * 70)

# ============================================================
# DATA AUGMENTATION & LOADING (Streaming with tf.keras)
# ============================================================
# Use tf.keras.utils.image_dataset_from_directory for memory-efficient loading
print("\nLoading dataset with streaming (memory-efficient)...")
train_generator = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='categorical'
)

# Get class names from the directory structure
class_names = sorted(os.listdir(DATASET_DIR))
class_indices = {name: idx for idx, name in enumerate(class_names)}

# Apply data augmentation after loading
augmentation_pipeline = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomWidth(0.1),
    tf.keras.layers.RandomHeight(0.1),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.15),
    Rescaling(1./255)
])

# Apply augmentation to the dataset
train_generator = train_generator.map(
    lambda x, y: (augmentation_pipeline(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Prefetch for performance
train_generator = train_generator.prefetch(tf.data.AUTOTUNE)

# Count total images
num_samples = len([f for subdir in class_names for f in os.listdir(os.path.join(DATASET_DIR, subdir)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
print(f"Found {num_samples} images belonging to {len(class_names)} classes.")

# ============================================================
# PRINT CLASS MAPPING
# ============================================================
print("\nClass Indices Mapping:")
print("-" * 70)
for class_name, class_idx in sorted(class_indices.items(), key=lambda x: x[1]):
    print(f"  {class_idx}: {class_name}")
print("-" * 70)

# ============================================================
# BUILD TINY CNN MODEL
# ============================================================
print("\nBuilding Tiny CNN architecture...")
model = Sequential([
    # Block 1: Conv + Pool
    Conv2D(32, kernel_size=(3, 3), activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 2: Conv + Pool
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 3: Conv + Pool
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Global pooling and Dense layers (memory-stable)
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes: Normal, Cataract, Glaucoma, DR
])

# ============================================================
# COMPILE MODEL
# ============================================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# PRINT MODEL SUMMARY
# ============================================================
print("\nModel Summary:")
print("-" * 70)
model.summary()
print("-" * 70)

# ============================================================
# TRAIN MODEL
# ============================================================
print(f"\nTraining for {EPOCHS} epochs with batch_size={BATCH_SIZE}...")
print("=" * 70)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    verbose=1,
    steps_per_epoch=None  # Use all batches
)

# ============================================================
# SAVE MODEL
# ============================================================
print("\n" + "=" * 70)
print(f"Saving model as {MODEL_NAME}...")
model.save(MODEL_NAME)
print(f"✅ Model saved successfully!")
print(f"   Location: {os.path.abspath(MODEL_NAME)}")
print("=" * 70)

# ============================================================
# TRAINING SUMMARY
# ============================================================
print("\nTraining Summary:")
print("-" * 70)
print(f"Final Training Loss:     {history.history['loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print("-" * 70)

print("\n✅ Training complete!")

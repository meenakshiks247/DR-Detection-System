import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import os

# Dataset configuration
DATASET_DIR = r'E:\Major project\DR_Detection_System\dataset_generalist'
BATCH_SIZE = 2
IMG_HEIGHT = 96
IMG_WIDTH = 96
EPOCHS = 3
MODEL_NAME = 'generalist_model.h5'

print("=" * 60)
print("GENERALIST MODEL TRAINING (TF Data Pipeline)")
print("=" * 60)

# Get class names and create label mapping
class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
class_indices = {name: idx for idx, name in enumerate(class_names)}

print("\nClass Indices Mapping:")
print("-" * 60)
for class_name, class_idx in sorted(class_indices.items(), key=lambda x: x[1]):
    print(f"  {class_idx}: {class_name}")
print("-" * 60)

# Build dataset using tf.data API (more memory efficient)
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    # One-hot encode the label
    label = tf.one_hot(label, depth=4)
    return image, label

# Create dataset
image_paths = []
labels = []

for class_name, class_idx in class_indices.items():
    class_dir = os.path.join(DATASET_DIR, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(class_dir, img_file))
            labels.append(class_idx)

print(f"\nFound {len(image_paths)} images")

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.shuffle(buffer_size=len(image_paths))
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Load pretrained MobileNetV2
print("\nLoading pretrained MobileNetV2...")
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model weights
base_model.trainable = False
print(f"Base model loaded. Trainable layers: {len([l for l in base_model.layers if l.trainable])}")

# Build the model
print("\nBuilding model architecture...")
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
print("-" * 60)
model.summary()
print("-" * 60)

# Train
print(f"\nTraining for {EPOCHS} epochs...")
print("=" * 60)
history = model.fit(
    dataset,
    epochs=EPOCHS,
    verbose=1
)

# Save the model
print("\n" + "=" * 60)
print(f"Saving model as {MODEL_NAME}...")
model.save(MODEL_NAME)
print(f"✅ Model saved successfully: {os.path.abspath(MODEL_NAME)}")
print("=" * 60)

# Print training summary
print("\nTraining Summary:")
print("-" * 60)
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print("-" * 60)

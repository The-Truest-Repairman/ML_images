#%% Step 1: Load and Filter Data
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
images = np.load("images/images.npy")        # shape: (N, 1080, 1080, 6)
targets = np.load("images/targets.npy")      # shape: (N,)

# Remove classes with <2 samples
label_counts = Counter(targets)
valid_classes = {label for label, count in label_counts.items() if count >= 2}
valid_indices = np.array([i for i, label in enumerate(targets) if label in valid_classes])

images_filtered = images[valid_indices]
targets_filtered = targets[valid_indices]

# Encode labels to 0-based continuous range
label_encoder = LabelEncoder()
targets_encoded = label_encoder.fit_transform(targets_filtered)

# Calculate safe test size
num_classes = len(np.unique(targets_encoded))
num_samples = len(targets_encoded)
test_size = max(num_classes, int(0.2 * num_samples))

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    images_filtered,
    targets_encoded,
    test_size=test_size,
    stratify=targets_encoded,
    random_state=42
)

print(f"✅ Data split successful.")
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Classes in test set: {len(np.unique(y_test))}")

#%% Step 2: Define and Train CNN Model
import tensorflow as tf
from keras import layers, models

IMG_SIZE = 1080  # use full resolution

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 6)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu', name="last_conv"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # Use encoded label count
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=4)

#%% Step 3: Grad-CAM + Export 7-Channel TIFFs
import os
from scipy.ndimage import zoom
import tifffile

def get_gradcam_heatmap(model, image, class_index, last_conv_layer_name="last_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(image, axis=0))
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

def resize_heatmap(heatmap, target_shape):
    h_factor = target_shape[0] / heatmap.shape[0]
    w_factor = target_shape[1] / heatmap.shape[1]
    return zoom(heatmap, (h_factor, w_factor), order=1)

# Output directory
output_dir = "outputs/gradcam_overlays"
os.makedirs(output_dir, exist_ok=True)

for idx, img in enumerate(X_test):
    pred = model.predict(tf.expand_dims(img, axis=0), verbose=0)
    class_idx = tf.argmax(pred[0]).numpy()

    heatmap = get_gradcam_heatmap(model, img, class_idx)
    heatmap_resized = resize_heatmap(heatmap, img.shape[:2])  # should match 1080x1080

    # Combine image (6 channels) + Grad-CAM (1 channel) → (H, W, 7)
    heatmap_resized = np.expand_dims(heatmap_resized.astype(np.float32), axis=-1)
    combined = np.concatenate([img, heatmap_resized], axis=-1)

    # Transpose to (channels, height, width) for TIFF export
    combined_stack = np.transpose(combined, (2, 0, 1))  # shape: (7, 1080, 1080)

    # Decode original label for filename
    original_label = label_encoder.inverse_transform([class_idx])[0]
    filename = os.path.join(output_dir, f"image_{idx:03}_class{original_label}.tiff")

    tifffile.imwrite(filename, combined_stack, photometric='minisblack')

    if idx % 10 == 0 or idx == len(X_test) - 1:
        print(f"✅ Saved {idx + 1}/{len(X_test)}: {filename}")

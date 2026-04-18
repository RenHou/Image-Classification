# ============================================================
# Train CNN Models on Clean, Noisy, and Mixed Image Sets
# ============================================================

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# =========================
# 1. CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "clustered_fruits")
CLEAN_DIR = os.path.join(DATASET_DIR, "clean")
NOISY_DIR = os.path.join(DATASET_DIR, "noisy")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 2. LOAD DATASETS
# =========================
def load_dataset(path, subset=None):
    return keras.preprocessing.image_dataset_from_directory( #Automatically: Reads images from folders, Assigns labels based on folder names 
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2 if subset else None,
        subset=subset
    )

print("Loading datasets...")

# Clean dataset (train + validation)
clean_train_ds = load_dataset(CLEAN_DIR, subset="training")
clean_val_ds   = load_dataset(CLEAN_DIR, subset="validation")

# Noisy dataset (training only)
noisy_train_ds = load_dataset(NOISY_DIR)

# Mixed dataset = clean (train) + noisy
mixed_train_ds = clean_train_ds.concatenate(noisy_train_ds)

class_names = clean_train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# =========================
# 3. NORMALIZATION & PREFETCH
# =========================
normalization_layer = layers.Rescaling(1.0 / 255) #Changes pixel values from 0–255 → 0–1

def normalize(ds):
    return ds.map(lambda x, y: (normalization_layer(x), y))

# Applies normalization to every image
clean_train_ds = normalize(clean_train_ds)
clean_val_ds   = normalize(clean_val_ds)
noisy_train_ds = normalize(noisy_train_ds)
mixed_train_ds = normalize(mixed_train_ds)

# Makes training faster by loading data early
# AUTOTUNE automatically chooses the best data loading speed to improve training performance.
AUTOTUNE = tf.data.AUTOTUNE
clean_train_ds = clean_train_ds.prefetch(AUTOTUNE)
clean_val_ds   = clean_val_ds.prefetch(AUTOTUNE)
noisy_train_ds = noisy_train_ds.prefetch(AUTOTUNE)
mixed_train_ds = mixed_train_ds.prefetch(AUTOTUNE)

# =========================
# 4. CNN MODEL DEFINITION
# =========================
def build_model(num_classes):
    base_model = keras.applications.ResNet50(
        weights="imagenet", # use the trained model by ResNet50
        include_top=False, # do not include the classification layer
        input_shape=(224, 224, 3) # specify the input shape, the size of image and the color
    )
    base_model.trainable = False  # Fair comparison

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # Converts image features into numbers
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"), # A thinking layer with 256 neurons， 
        # ReLU is an activation function that replaces negative values with zero to help the model learn faster.
        layers.Dropout(0.5), # Prevents overfitting，Turns off 50% of neurons randomly
        layers.Dense(num_classes, activation="softmax")
        # Softmax converts the model’s output into probabilities so that one class can be selected.
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4), # Works very well for images
        loss="sparse_categorical_crossentropy", # suitable for multiple class classification, and label in number (e.g. 0,1,2)
        metrics=["accuracy"]
    )
    return model

# =========================
# 5. TRAIN CLEAN MODEL
# =========================

print("\n" + "="*60)
print("TRAINING CNN ON CLEAN DATA")
print("="*60)

clean_model = build_model(num_classes)

history_clean = clean_model.fit(
    clean_train_ds,
    validation_data=clean_val_ds,
    epochs=EPOCHS
    
)


clean_model_path = os.path.join(MODEL_DIR, "cnn_clean_model.h5")
clean_model.save(clean_model_path)

# =========================
# 6. TRAIN NOISY MODEL
# =========================
print("\n" + "="*60)
print("TRAINING CNN ON NOISY DATA")
print("="*60)

noisy_model = build_model(num_classes)

history_noisy = noisy_model.fit(
    noisy_train_ds,
    epochs=EPOCHS
)

noisy_model_path = os.path.join(MODEL_DIR, "cnn_noisy_model.h5")
noisy_model.save(noisy_model_path)

# =========================
# 7. TRAIN MIXED MODEL
# =========================
print("\n" + "="*60)
print("TRAINING CNN ON MIXED (CLEAN + NOISY) DATA")
print("="*60)

mixed_model = build_model(num_classes)

history_mixed = mixed_model.fit(
    mixed_train_ds,
    validation_data=clean_val_ds,
    epochs=EPOCHS
)

mixed_model_path = os.path.join(MODEL_DIR, "cnn_mixed_model.h5")
mixed_model.save(mixed_model_path)

# =========================
# 8. EVALUATION (SAME CLEAN SET)
# =========================
print("\n" + "="*60)
print("EVALUATION ON CLEAN VALIDATION SET")
print("="*60)

clean_eval = clean_model.evaluate(clean_val_ds, verbose=0)
noisy_eval = noisy_model.evaluate(clean_val_ds, verbose=0)
mixed_eval = mixed_model.evaluate(clean_val_ds, verbose=0)

print(f"CNN-Clean Accuracy : {clean_eval[1]:.4f}")
print(f"CNN-Noisy Accuracy : {noisy_eval[1]:.4f}")
print(f"CNN-Mixed Accuracy : {mixed_eval[1]:.4f}")

# =========================
# 9. TRAINING CURVES
# =========================
def plot_history(history, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

plot_history(
    history_clean,
    "CNN Training on Clean Data",
    os.path.join(MODEL_DIR, "clean_training_curve.png")
)

plot_history(
    history_noisy,
    "CNN Training on Noisy Data",
    os.path.join(MODEL_DIR, "noisy_training_curve.png")
)

plot_history(
    history_mixed,
    "CNN Training on Mixed Data",
    os.path.join(MODEL_DIR, "mixed_training_curve.png")
)

# =========================
# 10. FINAL SUMMARY
# =========================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"Number of classes: {num_classes}")
print(f"Clean model saved  : {clean_model_path}")
print(f"Noisy model saved  : {noisy_model_path}")
print(f"Mixed model saved  : {mixed_model_path}")

print("\nPerformance comparison:")
print(f"  Clean  → {clean_eval[1]:.4f}")
print(f"  Noisy  → {noisy_eval[1]:.4f}")
print(f"  Mixed  → {mixed_eval[1]:.4f}")

print("\nConclusion:")
if clean_eval[1] >= mixed_eval[1] >= noisy_eval[1]:
    print("✓ Clean data yields best performance; noise degrades learning.")
else:
    print("✓ Mixed training partially mitigates noise impact.")

print("\nAll models and plots saved to: trained_models/")

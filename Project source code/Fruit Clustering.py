import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import json
import shutil
from pathlib import Path
import cv2

# =========================
# 1. CONFIGURATION
# =========================
# Resolve dataset/output paths relative to this script so the script
# works regardless of the current working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_by_fruit")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
OUTPUT_DIR = os.path.join(BASE_DIR, "clustered_fruits")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "clean"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "noisy"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "outliers"), exist_ok=True)

# =========================
# 2. LOAD DATASET WITH FILE PATHS
# =========================
print("Loading dataset...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,  # Keep order for tracking files
    seed=SEED
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Get file paths
file_paths = []
file_labels = []
for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(DATASET_DIR, class_name)
    for img_file in sorted(os.listdir(class_dir)):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_paths.append(os.path.join(class_dir, img_file))
            file_labels.append(class_idx)

print(f"Total images: {len(file_paths)}")

# =========================
# 3. FEATURE EXTRACTION
# =========================
print("\nExtracting features using ResNet50...")
feature_model = keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features_from_path(img_path):
    """Extract features from a single image file"""
    img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed = keras.applications.resnet50.preprocess_input(img_array)
    features = feature_model.predict(processed, verbose=0)
    return features[0]

def calculate_image_quality_metrics(img_path):
    """Calculate quality metrics for noise detection"""
    img = cv2.imread(img_path)
    if img is None:
        return {'blur': 0, 'brightness': 0, 'contrast': 0}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur detection (Laplacian variance)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Brightness
    brightness = np.mean(gray)
    
    # Contrast (standard deviation)
    contrast = np.std(gray)
    
    return {
        'blur': blur_score,
        'brightness': brightness,
        'contrast': contrast
    }

# Extract features and quality metrics
features_list = []
quality_metrics_list = []

for i, img_path in enumerate(file_paths):
    if i % 50 == 0:
        print(f"Processing image {i+1}/{len(file_paths)}...")
    
    features = extract_features_from_path(img_path)
    features_list.append(features)
    
    quality = calculate_image_quality_metrics(img_path)
    quality_metrics_list.append([quality['blur'], quality['brightness'], quality['contrast']])

features = np.array(features_list)
quality_metrics = np.array(quality_metrics_list)
true_labels = np.array(file_labels)

print(f"Features shape: {features.shape}")
print(f"Quality metrics shape: {quality_metrics.shape}")

# =========================
# 4. NOISE DETECTION USING ISOLATION FOREST
# =========================
print("\n" + "="*50)
print("DETECTING NOISY IMAGES")
print("="*50)

# Detect outliers/noisy images using Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=SEED)
noise_predictions = iso_forest.fit_predict(features)  # -1 for outliers, 1 for inliers

# Also detect based on quality metrics
quality_outliers = IsolationForest(contamination=0.15, random_state=SEED)
quality_predictions = quality_outliers.fit_predict(quality_metrics)

# Combine both methods (if either method flags as outlier)
is_noisy = (noise_predictions == -1) | (quality_predictions == -1)
is_clean = ~is_noisy

print(f"Clean images: {np.sum(is_clean)} ({np.sum(is_clean)/len(is_clean)*100:.1f}%)")
print(f"Noisy images: {np.sum(is_noisy)} ({np.sum(is_noisy)/len(is_noisy)*100:.1f}%)")

# =========================
# 5. CLUSTERING WITHIN EACH CATEGORY
# =========================
print("\n" + "="*50)
print("CLUSTERING BY FRUIT TYPE (CLEAN vs NOISY)")
print("="*50)

results = {
    'clean': {},
    'noisy': {},
    'statistics': {}
}

for category, mask in [('clean', is_clean), ('noisy', is_noisy)]:
    print(f"\n--- {category.upper()} IMAGES ---")
    
    if np.sum(mask) == 0:
        print(f"No {category} images found.")
        continue
    
    category_features = features[mask]
    category_labels = true_labels[mask]
    category_paths = [file_paths[i] for i in range(len(file_paths)) if mask[i]]
    
    # Cluster within this category
    n_clusters = num_classes
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(category_features)
    
    # Map clusters to classes
    cluster_to_class = {}
    for c in range(n_clusters):
        inds = np.where(cluster_labels == c)[0]
        if len(inds) == 0:
            cluster_to_class[c] = None
        else:
            votes = np.bincount(category_labels[inds], minlength=num_classes)
            majority = int(votes.argmax())
            cluster_to_class[c] = majority
    
    # Calculate accuracy
    mapped_labels = np.array([cluster_to_class[c] if cluster_to_class[c] is not None else -1 
                              for c in cluster_labels])
    valid = mapped_labels >= 0
    accuracy = (mapped_labels[valid] == category_labels[valid]).mean()
    
    print(f"\nCluster mapping for {category} images:")
    for c in range(n_clusters):
        inds = np.where(cluster_labels == c)[0]
        count = len(inds)
        if count == 0:
            continue
        maj = cluster_to_class[c]
        maj_name = class_names[maj] if maj is not None else "None"
        pct = (np.sum(category_labels[inds] == maj) / count) * 100
        print(f"  Cluster {c}: {maj_name} — size {count}, purity {pct:.1f}%")
    
    print(f"Cluster accuracy: {accuracy:.4f}")
    
    try:
        sil = silhouette_score(category_features, cluster_labels)
        print(f"Silhouette score: {sil:.4f}")
    except:
        sil = None
    
    # Save results
    results[category] = {
        'cluster_to_class': cluster_to_class,
        'accuracy': float(accuracy),
        'silhouette': float(sil) if sil is not None else None,
        'num_images': int(np.sum(mask))
    }
    
    # Organize files by fruit and category
    for i, (cluster_id, true_label, path) in enumerate(zip(cluster_labels, category_labels, category_paths)):
        fruit_name = class_names[true_label]
        dest_dir = os.path.join(OUTPUT_DIR, category, fruit_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy file with cluster info
        filename = os.path.basename(path)
        dest_path = os.path.join(dest_dir, f"cluster{cluster_id}_{filename}")
        shutil.copy2(path, dest_path)

# =========================
# 6. ANALYZE NOISE BY FRUIT TYPE
# =========================
print("\n" + "="*50)
print("NOISE DISTRIBUTION BY FRUIT TYPE")
print("="*50)

for class_idx, class_name in enumerate(class_names):
    class_mask = true_labels == class_idx
    total = np.sum(class_mask)
    clean_count = np.sum(class_mask & is_clean)
    noisy_count = np.sum(class_mask & is_noisy)
    
    print(f"{class_name}:")
    print(f"  Total: {total}")
    print(f"  Clean: {clean_count} ({clean_count/total*100:.1f}%)")
    print(f"  Noisy: {noisy_count} ({noisy_count/total*100:.1f}%)")
    
    results['statistics'][class_name] = {
        'total': int(total),
        'clean': int(clean_count),
        'noisy': int(noisy_count)
    }

# =========================
# 7. VISUALIZATIONS
# =========================
print("\nGenerating visualizations...")

# PCA visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: True labels colored, with noise/clean markers
ax1 = axes[0, 0]
ax1.scatter(reduced_features[is_clean, 0], reduced_features[is_clean, 1], 
           c=true_labels[is_clean], cmap='tab10', alpha=0.6, s=30, 
           marker='o', label='Clean', edgecolors='black', linewidth=0.5)
ax1.scatter(reduced_features[is_noisy, 0], reduced_features[is_noisy, 1], 
           c=true_labels[is_noisy], cmap='tab10', alpha=0.6, s=80, 
           marker='x', label='Noisy', linewidth=2)
ax1.set_title('PCA: True Labels (Clean=circles, Noisy=X)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Clean images only
ax2 = axes[0, 1]
if np.sum(is_clean) > 0:
    clean_features = reduced_features[is_clean]
    clean_labels = true_labels[is_clean]
    scatter = ax2.scatter(clean_features[:, 0], clean_features[:, 1], 
                         c=clean_labels, cmap='tab10', alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
    ax2.set_title('Clean Images Only', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

# Plot 3: Noisy images only
ax3 = axes[1, 0]
if np.sum(is_noisy) > 0:
    noisy_features = reduced_features[is_noisy]
    noisy_labels = true_labels[is_noisy]
    scatter = ax3.scatter(noisy_features[:, 0], noisy_features[:, 1], 
                         c=noisy_labels, cmap='tab10', alpha=0.7, s=80, 
                         marker='x', linewidth=2)
    ax3.set_title('Noisy/Outlier Images Only', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

# Plot 4: Quality metrics scatter
ax4 = axes[1, 1]
ax4.scatter(quality_metrics[is_clean, 0], quality_metrics[is_clean, 2], 
           alpha=0.6, s=30, label='Clean', color='green')
ax4.scatter(quality_metrics[is_noisy, 0], quality_metrics[is_noisy, 2], 
           alpha=0.6, s=30, label='Noisy', color='red')
ax4.set_xlabel('Blur Score (Laplacian variance)')
ax4.set_ylabel('Contrast (std dev)')
ax4.set_title('Image Quality: Blur vs Contrast', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'noise_analysis.png'), dpi=150, bbox_inches='tight')
print(f"Visualization saved to {OUTPUT_DIR}/noise_analysis.png")
plt.show()

# Bar chart of noise distribution
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(class_names))
width = 0.35

clean_counts = [results['statistics'][cn]['clean'] for cn in class_names]
noisy_counts = [results['statistics'][cn]['noisy'] for cn in class_names]

ax.bar(x - width/2, clean_counts, width, label='Clean', color='green', alpha=0.7)
ax.bar(x + width/2, noisy_counts, width, label='Noisy', color='red', alpha=0.7)

ax.set_xlabel('Fruit Type', fontweight='bold')
ax.set_ylabel('Number of Images', fontweight='bold')
ax.set_title('Clean vs Noisy Images by Fruit Type', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'noise_distribution.png'), dpi=150, bbox_inches='tight')
print(f"Distribution chart saved to {OUTPUT_DIR}/noise_distribution.png")
plt.show()

# =========================
# 8. SAVE RESULTS
# =========================
# Save JSON report
with open(os.path.join(OUTPUT_DIR, 'clustering_report.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save noise list
noise_list = {
    'noisy_files': [file_paths[i] for i in range(len(file_paths)) if is_noisy[i]],
    'clean_files': [file_paths[i] for i in range(len(file_paths)) if is_clean[i]]
}

with open(os.path.join(OUTPUT_DIR, 'noise_file_list.json'), 'w') as f:
    json.dump(noise_list, f, indent=2)

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Total images processed: {len(file_paths)}")
print(f"Clean images: {np.sum(is_clean)} ({np.sum(is_clean)/len(is_clean)*100:.1f}%)")
print(f"Noisy images: {np.sum(is_noisy)} ({np.sum(is_noisy)/len(is_noisy)*100:.1f}%)")
print(f"\nClean images clustering accuracy: {results['clean'].get('accuracy', 0):.4f}")
print(f"Noisy images clustering accuracy: {results['noisy'].get('accuracy', 0):.4f}")
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("  - clustering_report.json: Detailed statistics")
print("  - noise_file_list.json: Lists of clean/noisy files")
print("  - noise_analysis.png: PCA and quality visualizations")
print("  - noise_distribution.png: Bar chart by fruit type")
print(f"  - clean/: Clean images organized by fruit")
print(f"  - noisy/: Noisy images organized by fruit")

# =========================
# 9. RECOMMENDATIONS
# =========================
print("\n" + "="*50)
print("RECOMMENDATIONS")
print("="*50)

if np.sum(is_noisy) / len(is_noisy) > 0.2:
    print("  HIGH NOISE DETECTED (>20% of dataset)")
    print("   - Review noisy images in 'clustered_fruits/noisy/'")
    print("   - Consider removing or re-labeling problematic images")
    print("   - Check data collection process for quality issues")
else:
    print("✓ Noise level is acceptable (<20%)")

print("\nNext steps:")
print("1. Review images in 'noisy/' folders - some might be mislabeled")
print("2. Train model on 'clean/' images only for comparison")
print("3. Use noise detection as data augmentation strategy")
print("4. Investigate fruit types with high noise ratios")
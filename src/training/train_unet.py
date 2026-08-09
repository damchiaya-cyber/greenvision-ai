import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# Parameters
IMAGE_PATH = r'c:\stage PFE\sentinel_patch.tiff'
PATCH_SIZE = 128
POPULATION = 100000  # Optional for GSI per capita
PIXEL_SIZE = 10  # meters per pixel

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    exg = 2 * image_rgb[:, :, 1] - image_rgb[:, :, 0] - image_rgb[:, :, 2]
    exg = np.clip(exg, 0, 255).astype(np.uint8)
    return image_rgb, exg

def kmeans_thresholding(exg, k=2):
    flat = exg.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, n_init=10).fit(flat)
    clustered = kmeans.labels_.reshape(exg.shape)
    green_cluster = np.argmax([np.mean(exg[clustered == i]) for i in range(k)])
    binary_mask = (clustered == green_cluster).astype(np.uint8)
    return binary_mask

def extract_patches(image, mask, patch_size):
    patches_img, patches_mask = [], []
    h, w = image.shape[:2]
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            patches_img.append(img_patch)
            patches_mask.append(mask_patch[..., np.newaxis])
    return np.array(patches_img), np.array(patches_mask)

def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        return x

    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D()(c3)

    bn = conv_block(p3, 128)

    u1 = layers.UpSampling2D()(bn)
    u1 = layers.concatenate([u1, c3])
    c4 = conv_block(u1, 64)
    u2 = layers.UpSampling2D()(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = conv_block(u2, 32)
    u3 = layers.UpSampling2D()(c5)
    u3 = layers.concatenate([u3, c1])
    c6 = conv_block(u3, 16)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c6)
    return models.Model(inputs, outputs)

def calculate_gsi_from_mask(mask, pixel_size, population=None):
    green_pixels = np.sum(mask > 0)
    area_m2 = green_pixels * (pixel_size ** 2)
    area_km2 = area_m2 / 1e6
    gsi = area_km2 if population is None else area_km2 / population
    return green_pixels, area_km2, gsi

def predict_on_full_image(image_rgb, model, patch_size):
    h, w, _ = image_rgb.shape
    predicted_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = image_rgb[y:y+patch_size, x:x+patch_size]
            input_patch = patch[np.newaxis, ...] / 255.0
            pred = model.predict(input_patch)[0, ..., 0]
            pred_bin = (pred > 0.5).astype(np.uint8) * 255
            predicted_mask[y:y+patch_size, x:x+patch_size] = pred_bin

    return predicted_mask

# --- Load and preprocess image ---
image_rgb, exg = load_and_preprocess_image(IMAGE_PATH)
binary_mask = kmeans_thresholding(exg)

# --- Extract patches for training ---
X, y = extract_patches(image_rgb, binary_mask, PATCH_SIZE)
X = X / 255.0  # Normalize
y = y.astype(np.float32)

# --- Split data ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# --- Build and train U-Net ---
model = build_unet((PATCH_SIZE, PATCH_SIZE, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)
model.save('unet_model.h5')
print("\n✅ Training complete! Model saved.")

# --- Predict full image with patch-wise method ---
predicted_mask_bin = predict_on_full_image(image_rgb, model, PATCH_SIZE)
cv2.imwrite('unet_predicted_mask.png', predicted_mask_bin)
print("✅ Prediction complete. Saved as 'unet_predicted_mask.png'")

# --- Calculate GSI ---
green_pixels, area_km2, gsi = calculate_gsi_from_mask(predicted_mask_bin, PIXEL_SIZE, POPULATION)
print(f"\n🌱 Green Pixels: {green_pixels}")
print(f"📐 Area: {area_km2:.4f} km²")
print(f"📊 GSI (km² per capita): {gsi:.8f}")

# --- Visualization ---
original = image_rgb
predicted_bin = predicted_mask_bin
overlay = original.copy()
overlay[predicted_bin == 255] = [0, 255, 0]

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(original)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Predicted Mask')
plt.imshow(predicted_bin, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Overlay')
plt.imshow(overlay)
plt.axis('off')
plt.tight_layout()
plt.show()

# --- Optional evaluation ---
if os.path.exists('ground_truth_mask.png'):
    ground_truth = cv2.imread('ground_truth_mask.png', cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.resize(ground_truth, (predicted_bin.shape[1], predicted_bin.shape[0]))
    _, ground_truth_bin = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
    y_true = ground_truth_bin.flatten() // 255
    y_pred = predicted_bin.flatten() // 255

    print("\n📊 Evaluation Metrics:")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"IoU:       {jaccard_score(y_true, y_pred):.4f}")
else:
    print("⚠️ No ground truth mask found. Skipping metrics.")


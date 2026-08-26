from pathlib import Path
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = Path("data/processed")
DATASET_DIR = Path("data/dataset")

PATCH_SIZE = 128
STRIDE = 128

# NDVI threshold used to create vegetation masks.
# Values above this are considered vegetation.
NDVI_THRESHOLD = 0.30

TRAIN_RATIO = 0.80
RANDOM_STATE = 42


# ============================================================
# DIRECTORIES
# ============================================================

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_MASKS = DATASET_DIR / "train" / "masks"

VAL_IMAGES = DATASET_DIR / "val" / "images"
VAL_MASKS = DATASET_DIR / "val" / "masks"

for directory in [
    TRAIN_IMAGES,
    TRAIN_MASKS,
    VAL_IMAGES,
    VAL_MASKS,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND NDVI FILES
# ============================================================

def find_ndvi_files():

    files = list(PROCESSED_DIR.rglob("*_NDVI.tif"))

    print()
    print("=" * 60)
    print("Searching for NDVI files")
    print("=" * 60)

    for file in files:
        print(file)

    print()
    print(f"Found {len(files)} NDVI files.")

    return files


# ============================================================
# CREATE PATCHES
# ============================================================

def create_patches(ndvi_path):

    print()
    print(f"Processing: {ndvi_path}")

    with rasterio.open(ndvi_path) as src:

        ndvi = src.read(1).astype(np.float32)

        profile = src.profile

    # Remove invalid values
    ndvi = np.nan_to_num(
        ndvi,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    height, width = ndvi.shape

    print(f"Image size: {width} x {height}")

    image_patches = []
    mask_patches = []

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):

        for x in range(0, width - PATCH_SIZE + 1, STRIDE):

            patch = ndvi[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # Skip completely empty patches
            if np.all(patch == 0):
                continue

            # Normalize NDVI from [-1, 1] approximately to [0, 1]
            image = (patch + 1.0) / 2.0
            image = np.clip(image, 0.0, 1.0)

            # Vegetation mask
            mask = (patch >= NDVI_THRESHOLD).astype(np.uint8)

            # Skip patches containing almost no useful information
            vegetation_ratio = mask.mean()

            if vegetation_ratio < 0.01:
                continue

            image_patches.append(image)
            mask_patches.append(mask)

    print(f"Created {len(image_patches)} patches.")

    return image_patches, mask_patches


# ============================================================
# SAVE PATCH
# ============================================================

def save_patch(image, mask, directory_images, directory_masks, index):

    image_path = directory_images / f"patch_{index:05d}.npy"
    mask_path = directory_masks / f"patch_{index:05d}.npy"

    np.save(image_path, image.astype(np.float32))
    np.save(mask_path, mask.astype(np.uint8))


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("GREENVISION AI - DATASET CREATION")
    print("=" * 60)

    ndvi_files = find_ndvi_files()

    if not ndvi_files:

        print()
        print("❌ No NDVI files found.")
        print("Run compute_indices first.")
        return

    all_images = []
    all_masks = []

    # --------------------------------------------------------
    # Process every NDVI image
    # --------------------------------------------------------

    for ndvi_file in ndvi_files:

        images, masks = create_patches(ndvi_file)

        all_images.extend(images)
        all_masks.extend(masks)

    print()
    print("=" * 60)
    print(f"TOTAL PATCHES: {len(all_images)}")
    print("=" * 60)

    if not all_images:

        print("❌ No patches were created.")
        return

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    indices = np.arange(len(all_images))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=1 - TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print()
    print(f"Training patches:   {len(train_indices)}")
    print(f"Validation patches: {len(val_indices)}")

    # --------------------------------------------------------
    # Save training patches
    # --------------------------------------------------------

    print()
    print("Saving training dataset...")

    for i, idx in enumerate(tqdm(train_indices)):

        save_patch(
            all_images[idx],
            all_masks[idx],
            TRAIN_IMAGES,
            TRAIN_MASKS,
            i
        )

    # --------------------------------------------------------
    # Save validation patches
    # --------------------------------------------------------

    print()
    print("Saving validation dataset...")

    for i, idx in enumerate(tqdm(val_indices)):

        save_patch(
            all_images[idx],
            all_masks[idx],
            VAL_IMAGES,
            VAL_MASKS,
            i
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)

    print()
    print("Dataset:")
    print(DATASET_DIR)

    print()
    print("Training images:")
    print(len(list(TRAIN_IMAGES.glob("*.npy"))))

    print("Training masks:")
    print(len(list(TRAIN_MASKS.glob("*.npy"))))

    print()
    print("Validation images:")
    print(len(list(VAL_IMAGES.glob("*.npy"))))

    print("Validation masks:")
    print(len(list(VAL_MASKS.glob("*.npy"))))

    print()
    print("Next step:")
    print("    py -m scripts.train")


if __name__ == "__main__":
    main()
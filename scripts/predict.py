from pathlib import Path

import numpy as np
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/greenvision_unet.keras")

OUTPUT_DIR = Path("outputs/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 128
THRESHOLD = 0.5


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading GreenVision U-Net...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("✓ Model loaded")

    return model


# ============================================================
# LOAD NDVI
# ============================================================

def load_ndvi(path):

    with rasterio.open(path) as src:

        ndvi = src.read(1).astype(np.float32)

        profile = src.profile.copy()

    ndvi = np.nan_to_num(ndvi)

    minimum = np.min(ndvi)
    maximum = np.max(ndvi)

    if maximum > minimum:

        normalized = (
            (ndvi - minimum)
            /
            (maximum - minimum)
        )

    else:

        normalized = np.zeros_like(ndvi)

    return ndvi, normalized, profile


# ============================================================
# PREDICT FULL IMAGE
# ============================================================

def predict_image(model, normalized):

    height, width = normalized.shape

    prediction = np.zeros(
        (height, width),
        dtype=np.float32
    )

    count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    patches = []
    positions = []

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        PATCH_SIZE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            PATCH_SIZE
        ):

            patch = normalized[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            patches.append(
                patch[..., np.newaxis]
            )

            positions.append((y, x))

    patches = np.asarray(
        patches,
        dtype=np.float32
    )

    print(
        f"Predicting {len(patches)} patches..."
    )

    predictions = model.predict(
        patches,
        batch_size=16,
        verbose=1
    )

    for pred, (y, x) in zip(
        predictions,
        positions
    ):

        prediction[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ] += pred[..., 0]

        count[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ] += 1

    valid = count > 0

    prediction[valid] /= count[valid]

    return prediction


# ============================================================
# SAVE RESULTS
# ============================================================

def save_raster(
    path,
    data,
    profile
):

    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="deflate"
    )

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            data.astype(np.float32),
            1
        )


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    ndvi,
    prediction,
    name
):

    mask = (
        prediction >= THRESHOLD
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    axes[0].imshow(
        ndvi,
        cmap="RdYlGn"
    )

    axes[0].set_title(
        "NDVI"
    )

    axes[0].axis("off")

    axes[1].imshow(
        prediction,
        cmap="viridis"
    )

    axes[1].set_title(
        "U-Net prediction"
    )

    axes[1].axis("off")

    axes[2].imshow(
        mask,
        cmap="Greens"
    )

    axes[2].set_title(
        "Green-space mask"
    )

    axes[2].axis("off")

    plt.tight_layout()

    output = (
        OUTPUT_DIR /
        f"{name}_prediction.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Visualization saved: {output}"
    )


# ============================================================
# GREEN SPACE METRICS
# ============================================================

def calculate_metrics(
    prediction,
    pixel_size=10
):

    mask = (
        prediction >= THRESHOLD
    )

    total_pixels = mask.size

    green_pixels = np.sum(mask)

    coverage = (
        green_pixels
        /
        total_pixels
        *
        100
    )

    pixel_area_m2 = (
        pixel_size ** 2
    )

    green_area_m2 = (
        green_pixels
        *
        pixel_area_m2
    )

    green_area_km2 = (
        green_area_m2
        /
        1_000_000
    )

    return (
        coverage,
        green_area_km2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GREENVISION AI - SATELLITE PREDICTION")
    print("=" * 60)

    model = load_model()

    ndvi_files = list(
        Path("data/processed").rglob(
            "*_NDVI.tif"
        )
    )

    if not ndvi_files:

        print(
            "\n❌ No NDVI files found."
        )

        return

    print(
        f"\nFound {len(ndvi_files)} NDVI images."
    )

    for ndvi_path in ndvi_files:

        print("\n" + "-" * 60)

        print(
            f"Processing:\n{ndvi_path}"
        )

        name = ndvi_path.stem

        ndvi, normalized, profile = (
            load_ndvi(ndvi_path)
        )

        prediction = predict_image(
            model,
            normalized
        )

        # Save probability map

        probability_path = (
            OUTPUT_DIR /
            f"{name}_probability.tif"
        )

        save_raster(
            probability_path,
            prediction,
            profile
        )

        print(
            f"✓ Probability map saved:\n"
            f"  {probability_path}"
        )

        # Binary mask

        mask = (
            prediction >= THRESHOLD
        ).astype(np.float32)

        mask_path = (
            OUTPUT_DIR /
            f"{name}_mask.tif"
        )

        save_raster(
            mask_path,
            mask,
            profile
        )

        print(
            f"✓ Green-space mask saved:\n"
            f"  {mask_path}"
        )

        # Metrics

        coverage, area = calculate_metrics(
            prediction
        )

        print(
            f"\nGreen-space coverage: "
            f"{coverage:.2f}%"
        )

        print(
            f"Estimated green-space area: "
            f"{area:.2f} km²"
        )

        # Visualization

        create_visualization(
            ndvi,
            prediction,
            name
        )

    print("\n" + "=" * 60)

    print(
        "✓ PREDICTION PIPELINE COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
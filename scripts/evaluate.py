from pathlib import Path

import numpy as np
import tensorflow as tf


MODEL_PATH = Path("models/greenvision_unet.keras")
DATASET_DIR = Path("data/dataset")

BATCH_SIZE = 16
THRESHOLD = 0.5

def dice_score(y_true, y_pred):
    """
    Calculate Dice coefficient.
    """

    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    intersection = np.sum(y_true * y_pred)

    return (
        (2.0 * intersection + 1e-7)
        /
        (
            np.sum(y_true)
            +
            np.sum(y_pred)
            +
            1e-7
        )
    )


def iou_score(y_true, y_pred):
    """
    Calculate Intersection over Union.
    """

    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    intersection = np.sum(y_true * y_pred)

    union = (
        np.sum(y_true)
        +
        np.sum(y_pred)
        -
        intersection
    )

    return (intersection + 1e-7) / (union + 1e-7)


def precision_score(y_true, y_pred):
    """
    Calculate precision.
    """

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    true_positive = np.logical_and(
        y_true,
        y_pred
    ).sum()

    false_positive = np.logical_and(
        ~y_true,
        y_pred
    ).sum()

    return (
        true_positive
        /
        (true_positive + false_positive + 1e-7)
    )


def recall_score(y_true, y_pred):
    """
    Calculate recall.
    """

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    true_positive = np.logical_and(
        y_true,
        y_pred
    ).sum()

    false_negative = np.logical_and(
        y_true,
        ~y_pred
    ).sum()

    return (
        true_positive
        /
        (true_positive + false_negative + 1e-7)
    )


def load_dataset(split):
    """
    Load NumPy image and mask patches created by create_dataset.py.
    """

    image_dir = DATASET_DIR / split / "images"
    mask_dir = DATASET_DIR / split / "masks"

    image_files = sorted(image_dir.glob("*.npy"))

    print(
        f"Found {len(image_files)} image patches "
        f"for {split}."
    )

    images = []
    masks = []

    for image_path in image_files:

        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            print(
                f"Warning: missing mask for "
                f"{image_path.name}"
            )
            continue

        image = np.load(image_path)
        mask = np.load(mask_path)

        images.append(image)
        masks.append(mask)

    if not images:
        return (
            np.empty(
                (0, 128, 128, 1),
                dtype=np.float32
            ),
            np.empty(
                (0, 128, 128, 1),
                dtype=np.float32
            )
        )

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(masks, dtype=np.float32)

    if X.ndim == 3:
        X = X[..., np.newaxis]

    if y.ndim == 3:
        y = y[..., np.newaxis]

    return X, y


def evaluate_model():

    print("=" * 60)
    print("GREENVISION AI - MODEL EVALUATION")
    print("=" * 60)


    if not MODEL_PATH.exists():

        print(
            f"\n❌ Model not found:\n"
            f"{MODEL_PATH}"
        )

        return

    print(
        f"\nLoading model:\n"
        f"{MODEL_PATH}"
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("✓ Model loaded")


    print("\nLoading validation dataset...")

    X_val, y_val = load_dataset("val")

    print(
        f"Validation images: {X_val.shape}"
    )

    print(
        f"Validation masks:  {y_val.shape}"
    )

    if len(X_val) == 0:

        print(
            "\n❌ No validation patches found."
        )

        return


    print("\nGenerating predictions...")

    predictions = model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=1
    )


    predictions_binary = (
        predictions >= THRESHOLD
    ).astype(np.float32)

    y_val_binary = (
        y_val >= THRESHOLD
    ).astype(np.float32)

    dice = dice_score(
        y_val_binary,
        predictions_binary
    )

    iou = iou_score(
        y_val_binary,
        predictions_binary
    )

    precision = precision_score(
        y_val_binary,
        predictions_binary
    )

    recall = recall_score(
        y_val_binary,
        predictions_binary
    )

    accuracy = np.mean(
        y_val_binary == predictions_binary
    )


    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"Dice      : {dice:.4f}"
    )

    print(
        f"IoU       : {iou:.4f}"
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
    
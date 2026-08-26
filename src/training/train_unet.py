from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# GREENVISION AI - U-NET TRAINING
# ============================================================

PATCH_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 20

DATASET_DIR = Path("data/dataset")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(split):
    """
    Load pre-generated NumPy patches.

    Expected structure:

    data/dataset/
        train/
            images/
                patch_00000.npy
                patch_00001.npy
                ...
            masks/
                patch_00000.npy
                patch_00001.npy
                ...

        val/
            images/
            masks/
    """

    image_dir = DATASET_DIR / split / "images"
    mask_dir = DATASET_DIR / split / "masks"

    print(f"\nLoading {split} data...")
    print(f"Image directory: {image_dir}")
    print(f"Mask directory:  {mask_dir}")

    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return (
            np.empty((0, PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32),
            np.empty((0, PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32),
        )

    if not mask_dir.exists():
        print(f"❌ Mask directory not found: {mask_dir}")
        return (
            np.empty((0, PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32),
            np.empty((0, PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32),
        )

    image_files = sorted(image_dir.glob("*.npy"))

    print(f"Found {len(image_files)} image patches.")

    images = []
    masks = []

    for image_path in image_files:

        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            print(f"⚠ Missing mask: {mask_path.name}")
            continue

        try:
            image = np.load(image_path)
            mask = np.load(mask_path)

            image = np.nan_to_num(
                image,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            mask = np.nan_to_num(
                mask,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            image = image.astype(np.float32)
            mask = mask.astype(np.float32)

            # Make sure image has shape (128, 128, 1)
            if image.ndim == 2:
                image = image[..., np.newaxis]

            # Make sure mask has shape (128, 128, 1)
            if mask.ndim == 2:
                mask = mask[..., np.newaxis]

            # Skip malformed patches
            if image.shape != (PATCH_SIZE, PATCH_SIZE, 1):
                print(
                    f"⚠ Skipping {image_path.name}: "
                    f"image shape = {image.shape}"
                )
                continue

            if mask.shape != (PATCH_SIZE, PATCH_SIZE, 1):
                print(
                    f"⚠ Skipping {image_path.name}: "
                    f"mask shape = {mask.shape}"
                )
                continue

            images.append(image)
            masks.append(mask)

        except Exception as e:
            print(f"⚠ Error loading {image_path.name}: {e}")

    if not images:
        return (
            np.empty((0, PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32),
            np.empty((0, PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32),
        )

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(masks, dtype=np.float32)

    # Ensure masks are binary
    y = (y > 0.5).astype(np.float32)

    print(f"Loaded images: {X.shape}")
    print(f"Loaded masks:  {y.shape}")

    return X, y


# ============================================================
# U-NET MODEL
# ============================================================

def build_unet():

    inputs = tf.keras.Input(
        shape=(PATCH_SIZE, PATCH_SIZE, 1)
    )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    c1 = tf.keras.layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    )(inputs)

    c1 = tf.keras.layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    )(c1)

    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same"
    )(p1)

    c2 = tf.keras.layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same"
    )(c2)

    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = tf.keras.layers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same"
    )(p2)

    c3 = tf.keras.layers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same"
    )(c3)

    p3 = tf.keras.layers.MaxPooling2D()(c3)

    # --------------------------------------------------------
    # Bottleneck
    # --------------------------------------------------------

    b = tf.keras.layers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same"
    )(p3)

    b = tf.keras.layers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same"
    )(b)

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    u1 = tf.keras.layers.UpSampling2D()(b)

    u1 = tf.keras.layers.concatenate([
        u1,
        c3
    ])

    c4 = tf.keras.layers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same"
    )(u1)

    c4 = tf.keras.layers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same"
    )(c4)

    u2 = tf.keras.layers.UpSampling2D()(c4)

    u2 = tf.keras.layers.concatenate([
        u2,
        c2
    ])

    c5 = tf.keras.layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same"
    )(u2)

    c5 = tf.keras.layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same"
    )(c5)

    u3 = tf.keras.layers.UpSampling2D()(c5)

    u3 = tf.keras.layers.concatenate([
        u3,
        c1
    ])

    c6 = tf.keras.layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    )(u3)

    c6 = tf.keras.layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same"
    )(c6)

    outputs = tf.keras.layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c6)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )


# ============================================================
# DICE COEFFICIENT
# ============================================================

def dice_coefficient(y_true, y_pred):

    smooth = 1e-6

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred,
        tf.float32
    )

    y_pred = tf.round(y_pred)

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        (2.0 * intersection + smooth)
        /
        (
            tf.reduce_sum(y_true)
            +
            tf.reduce_sum(y_pred)
            +
            smooth
        )
    )


# ============================================================
# TRAINING
# ============================================================

def train_model():

    print("=" * 60)
    print("GREENVISION AI - U-NET TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X_train, y_train = load_dataset("train")

    print("\nTraining images:", X_train.shape)
    print("Training masks:", y_train.shape)

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    X_val, y_val = load_dataset("val")

    print("\nValidation images:", X_val.shape)
    print("Validation masks:", y_val.shape)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(X_train) == 0:

        print("\n❌ No training patches found.")

        return

    if len(X_val) == 0:

        print("\n❌ No validation patches found.")

        return

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding U-Net...")

    model = build_unet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            dice_coefficient
        ]
    )

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR / "greenvision_unet.keras",
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        MODEL_DIR /
        "greenvision_unet_final.keras"
    )

    model.save(final_model_path)

    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\n✓ Best model saved to:\n"
        f"  {MODEL_DIR / 'greenvision_unet.keras'}"
    )

    print(
        f"\n✓ Final model saved to:\n"
        f"  {final_model_path}"
    )

    return history


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_model()
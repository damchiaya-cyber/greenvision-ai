from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

RAW_FOLDER = Path("data/raw/sentinel")
OUTPUT_FOLDER = Path("data/processed")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def find_band(safe_folder, band):
    """
    Find a Sentinel-2 band inside a .SAFE folder.
    Example:
    B02 -> ..._B02_10m.jp2
    """

    files = list(safe_folder.rglob(f"*_{band}_10m.jp2"))

    if len(files) == 0:
        return None

    return files[0]


def normalize(image):
    image = image.astype(np.float32)

    image -= image.min()

    if image.max() != 0:
        image /= image.max()

    return image


for city in RAW_FOLDER.iterdir():

    if not city.is_dir():
        continue

    print(f"\n==============================")
    print(city.name)
    print("==============================")

    city_output = OUTPUT_FOLDER / city.name
    city_output.mkdir(exist_ok=True)

    for safe in city.glob("*.SAFE"):

        print(f"\nProcessing:")
        print(safe.name)

        blue = find_band(safe, "B02")
        green = find_band(safe, "B03")
        red = find_band(safe, "B04")
        nir = find_band(safe, "B08")

        if None in [blue, green, red, nir]:
            print("❌ Missing bands -> skipped")
            continue

        with rasterio.open(blue) as src:
            B = src.read(1).astype(np.float32)
            profile = src.profile

        with rasterio.open(green) as src:
            G = src.read(1).astype(np.float32)

        with rasterio.open(red) as src:
            R = src.read(1).astype(np.float32)

        with rasterio.open(nir) as src:
            NIR = src.read(1).astype(np.float32)

        print("✓ Bands loaded")

        # ----------------------------
        # NDVI
        # ----------------------------

        ndvi = (NIR - R) / (NIR + R + 1e-6)

        # ----------------------------
        # ExG
        # ----------------------------

        exg = 2 * G - R - B

        ndvi_norm = normalize(ndvi)
        exg_norm = normalize(exg)

        profile.update(
             driver="GTiff",
            dtype=rasterio.float32,
            count=1,
            compress="lzw"
)

        ndvi_file = city_output / f"{safe.stem}_NDVI.tif"
        exg_file = city_output / f"{safe.stem}_EXG.tif"

        with rasterio.open(ndvi_file, "w", **profile) as dst:
            dst.write(ndvi_norm.astype(np.float32), 1)

        with rasterio.open(exg_file, "w", **profile) as dst:
            dst.write(exg_norm.astype(np.float32), 1)

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.imshow(ndvi_norm, cmap="RdYlGn")
        plt.title("NDVI")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(exg_norm, cmap="Greens")
        plt.title("ExG")
        plt.axis("off")

        plt.tight_layout()

        preview = city_output / f"{safe.stem}_preview.png"

        plt.savefig(preview, dpi=150)
        plt.close()

        print("✓ NDVI saved")
        print("✓ ExG saved")
        print("✓ Preview saved")

print("\n===================================")
print("Finished!")
print("Results saved in data/processed/")
print("===================================")
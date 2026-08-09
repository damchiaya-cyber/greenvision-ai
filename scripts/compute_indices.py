from pathlib import Path
import numpy as np
import rasterio

from config.settings import RAW_DATA, PROCESSED_DATA


def find_band(safe_path, band):
    """
    Find a Sentinel-2 10m band inside a .SAFE directory.
    """
    matches = list(safe_path.rglob(f"*_{band}_10m.jp2"))

    if not matches:
        raise FileNotFoundError(
            f"Band {band} not found in {safe_path.name}"
        )

    return matches[0]


def read_band(path):
    """
    Read a raster band as float32.
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    return data, profile


def calculate_ndvi(red, nir):
    """
    NDVI = (NIR - Red) / (NIR + Red)
    """
    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(nir, dtype=np.float32),
        where=denominator != 0
    )

    return ndvi


def calculate_exg(red, green, blue):
    """
    Excess Green Index:
    ExG = 2G - R - B
    """
    return (2 * green - red - blue).astype(np.float32)


def save_raster(path, data, profile):
    """
    Save the calculated index as a GeoTIFF.
    Float32 is supported by GeoTIFF.
    """
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="deflate"
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def process_safe(safe_path, output_folder):
    print(f"\nProcessing:")
    print(safe_path.name)

    blue_path = find_band(safe_path, "B02")
    green_path = find_band(safe_path, "B03")
    red_path = find_band(safe_path, "B04")
    nir_path = find_band(safe_path, "B08")

    blue, profile = read_band(blue_path)
    green, _ = read_band(green_path)
    red, _ = read_band(red_path)
    nir, _ = read_band(nir_path)

    print("✓ Bands loaded")

    ndvi = calculate_ndvi(red, nir)
    exg = calculate_exg(red, green, blue)

    output_folder.mkdir(parents=True, exist_ok=True)

    name = safe_path.name.replace(".SAFE", "")

    ndvi_file = output_folder / f"{name}_NDVI.tif"
    exg_file = output_folder / f"{name}_ExG.tif"

    save_raster(ndvi_file, ndvi, profile)
    save_raster(exg_file, exg, profile)

    print(f"✓ NDVI saved: {ndvi_file.name}")
    print(f"✓ ExG saved:  {exg_file.name}")


def main():
    for city_folder in RAW_DATA.iterdir():

        if not city_folder.is_dir():
            continue

        print("\n" + "=" * 30)
        print(city_folder.name)
        print("=" * 30)

        safe_folders = list(city_folder.glob("*.SAFE"))

        if not safe_folders:
            print("No SAFE folders found.")
            continue

        output_folder = (
            PROCESSED_DATA / city_folder.name
        )

        for safe_path in safe_folders:

            try:
                process_safe(
                    safe_path,
                    output_folder
                )

            except Exception as e:
                print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
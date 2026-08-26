from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as raster_mask


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PREDICTION_DIR = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
)

BOUNDARY_DIR = (
    PROJECT_DIR
    / "data"
    / "boundaries"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "indicators"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CITY CONFIGURATION
# ============================================================

CITIES = {
    "Barcelona": {
        "prediction_pattern": "*T31TDF*_NDVI_mask.tif",
        "boundary": (
            BOUNDARY_DIR
            / "barcelona.geojson"
        ),
    },

    "Oujda": {
        "prediction_pattern": "*T30SWD*_NDVI_mask.tif",
        "boundary": (
            BOUNDARY_DIR
            / "oujda.geojson"
        ),
    },
}


# ============================================================
# FIND PREDICTION
# ============================================================

def find_prediction(pattern):

    matches = list(
        PREDICTION_DIR.glob(pattern)
    )

    if not matches:

        raise FileNotFoundError(
            f"No prediction found for pattern: {pattern}"
        )

    return matches[0]


# ============================================================
# LOAD CITY BOUNDARY
# ============================================================

def load_boundary(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Boundary file not found:\n{path}"
        )

    boundary = gpd.read_file(path)

    if boundary.empty:

        raise ValueError(
            f"Boundary file contains no geometries:\n{path}"
        )

    # Remove empty geometries
    boundary = boundary[
        boundary.geometry.notna()
        & ~boundary.geometry.is_empty
    ]

    # Combine all geometries into one city geometry
    geometry = boundary.geometry.union_all()

    return geometry


# ============================================================
# ANALYZE CITY
# ============================================================

def analyze_city(city_name, config):

    print("\n" + "=" * 70)
    print(f"ANALYZING: {city_name}")
    print("=" * 70)

    prediction_path = find_prediction(
        config["prediction_pattern"]
    )

    boundary_path = config["boundary"]

    print(
        f"\nPrediction:\n{prediction_path.name}"
    )

    print(
        f"\nBoundary:\n{boundary_path}"
    )

    # --------------------------------------------------------
    # Load raster
    # --------------------------------------------------------

    with rasterio.open(prediction_path) as src:

        raster_crs = src.crs

        print(
            f"\nRaster CRS: {raster_crs}"
        )

        # ----------------------------------------------------
        # Load boundary
        # ----------------------------------------------------

        geometry = load_boundary(
            boundary_path
        )

        # ----------------------------------------------------
        # Reproject boundary to raster CRS
        # ----------------------------------------------------

        boundary_gdf = gpd.GeoDataFrame(
            geometry=[geometry],
            crs="EPSG:4326"
        )

        if boundary_gdf.crs != raster_crs:

            boundary_gdf = boundary_gdf.to_crs(
                raster_crs
            )

        geometry = boundary_gdf.geometry.iloc[0]

        # ----------------------------------------------------
        # Clip prediction to city
        # ----------------------------------------------------

        clipped, clipped_transform = raster_mask(
            src,
            [geometry],
            crop=True,
            filled=True,
            nodata=0
        )

        clipped_mask = clipped[0]

        # ----------------------------------------------------
        # Determine valid pixels
        # ----------------------------------------------------

        valid_pixels = (
            clipped_mask >= 0
        )

        total_pixels = np.sum(
            valid_pixels
        )

        green_pixels = np.sum(
            clipped_mask > 0
        )

        if total_pixels == 0:

            raise ValueError(
                f"No valid pixels found inside "
                f"{city_name} boundary."
            )

        # ----------------------------------------------------
        # Pixel dimensions
        # ----------------------------------------------------

        pixel_width = abs(
            clipped_transform.a
        )

        pixel_height = abs(
            clipped_transform.e
        )

        pixel_area_m2 = (
            pixel_width *
            pixel_height
        )

        # ----------------------------------------------------
        # Areas
        # ----------------------------------------------------

        total_area_m2 = (
            total_pixels *
            pixel_area_m2
        )

        green_area_m2 = (
            green_pixels *
            pixel_area_m2
        )

        total_area_km2 = (
            total_area_m2 /
            1_000_000
        )

        green_area_km2 = (
            green_area_m2 /
            1_000_000
        )

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        green_coverage = (
            green_pixels /
            total_pixels
        ) * 100

        # ----------------------------------------------------
        # Save clipped mask
        # ----------------------------------------------------

        clipped_path = (
            OUTPUT_DIR /
            f"{city_name.lower()}_green_mask.tif"
        )

        profile = src.profile.copy()

        profile.update(
            height=clipped_mask.shape[0],
            width=clipped_mask.shape[1],
            transform=clipped_transform,
            dtype=rasterio.uint8,
            count=1,
            nodata=0
        )

        with rasterio.open(
            clipped_path,
            "w",
            **profile
        ) as dst:

            dst.write(
                clipped_mask.astype(
                    np.uint8
                ),
                1
            )

    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "city": city_name,

        "prediction_file": (
            prediction_path.name
        ),

        "total_area_km2": round(
            total_area_km2,
            4
        ),

        "green_area_km2": round(
            green_area_km2,
            4
        ),

        "green_coverage_percent": round(
            green_coverage,
            4
        ),

        "total_pixels": int(
            total_pixels
        ),

        "green_pixels": int(
            green_pixels
        ),

        "pixel_size_m": [
            float(pixel_width),
            float(pixel_height)
        ],

        "crs": str(
            raster_crs
        )
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = (
        OUTPUT_DIR /
        f"{city_name.lower()}.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "-" * 70)

    print(
        f"City:              {city_name}"
    )

    print(
        f"Total area:        "
        f"{total_area_km2:.2f} km²"
    )

    print(
        f"Green area:        "
        f"{green_area_km2:.2f} km²"
    )

    print(
        f"Green coverage:    "
        f"{green_coverage:.2f}%"
    )

    print(
        f"Total pixels:      "
        f"{total_pixels:,}"
    )

    print(
        f"Green pixels:      "
        f"{green_pixels:,}"
    )

    print(
        f"\n✓ Clipped mask:"
        f"\n  {clipped_path}"
    )

    print(
        f"\n✓ JSON:"
        f"\n  {json_path}"
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GREENVISION AI - CITY ENVIRONMENTAL INDICATORS")
    print("=" * 70)

    all_results = []

    for city_name, config in CITIES.items():

        try:

            result = analyze_city(
                city_name,
                config
            )

            all_results.append(
                result
            )

        except Exception as error:

            print(
                f"\n❌ ERROR processing "
                f"{city_name}:"
            )

            print(error)

    # ========================================================
    # CITY COMPARISON
    # ========================================================

    if all_results:

        comparison = pd.DataFrame(
            all_results
        )

        comparison = comparison[
            [
                "city",
                "total_area_km2",
                "green_area_km2",
                "green_coverage_percent"
            ]
        ]

        csv_path = (
            OUTPUT_DIR /
            "city_comparison.csv"
        )

        comparison.to_csv(
            csv_path,
            index=False
        )

        print("\n" + "=" * 70)
        print("CITY COMPARISON")
        print("=" * 70)

        print(
            comparison.to_string(
                index=False
            )
        )

        print(
            f"\n✓ Comparison saved to:"
            f"\n  {csv_path}"
        )

    print("\n" + "=" * 70)
    print("CITY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
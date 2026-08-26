from pathlib import Path
import json

import numpy as np
import rasterio
import streamlit as st
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent

PREDICTION_DIR = PROJECT_DIR / "outputs" / "predictions"
INDICATOR_DIR = PROJECT_DIR / "outputs" / "indicators"

st.set_page_config(
    page_title="GreenVision AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">🌱 GreenVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered urban green-space analysis from satellite imagery'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    GreenVision AI combines **Sentinel-2 satellite imagery**, 
    vegetation indices and a **U-Net semantic segmentation model**
    to detect vegetation and calculate urban green-space indicators.
    """
)

st.divider()



comparison_file = INDICATOR_DIR / "city_comparison.csv"

if not comparison_file.exists():

    st.error(
        "City indicator results were not found. "
        "Please run the city analysis pipeline first."
    )

    st.code(
        "py -m scripts.calculate_city_indicators",
        language="powershell"
    )

    st.stop()


@st.cache_data
def load_city_data():

    import pandas as pd

    df = pd.read_csv(comparison_file)

    return df


city_data = load_city_data()


st.sidebar.header("🌍 Study Area")

available_cities = city_data["city"].tolist()

selected_city = st.sidebar.selectbox(
    "Select city",
    available_cities
)


city_row = city_data[
    city_data["city"] == selected_city
].iloc[0]


total_area = float(
    city_row["total_area_km2"]
)

green_area = float(
    city_row["green_area_km2"]
)

green_coverage = float(
    city_row["green_coverage_percent"]
)


st.markdown(
    f'<div class="section-title">📍 {selected_city}</div>',
    unsafe_allow_html=True
)

st.write(
    "Municipality-level green-space analysis based on "
    "U-Net vegetation segmentation."
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🌳 Green area",
        f"{green_area:.2f} km²"
    )


with col2:

    st.metric(
        "📊 Green coverage",
        f"{green_coverage:.2f}%"
    )


with col3:

    st.metric(
        "🌍 Total area",
        f"{total_area:.2f} km²"
    )


st.divider()


st.subheader(
    f"🗺️ Green-space segmentation — {selected_city}"
)


clipped_mask = (
    INDICATOR_DIR /
    f"{selected_city.lower()}_green_mask.tif"
)


if clipped_mask.exists():

    @st.cache_data
    def load_clipped_mask(path):

        with rasterio.open(path) as src:

            mask = src.read(1)

            crs = src.crs

            width = src.width
            height = src.height

            transform = src.transform

        return (
            mask,
            crs,
            width,
            height,
            transform
        )


    (
        mask,
        crs,
        width,
        height,
        transform
    ) = load_clipped_mask(clipped_mask)


    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.imshow(
        mask,
        cmap="Greens"
    )

    ax.set_title(
        f"{selected_city} — Detected Green Spaces"
    )

    ax.axis("off")

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

else:

    st.warning(
        f"Clipped mask not found for {selected_city}."
    )



st.subheader(
    "📊 Green-space composition"
)


non_green_area = (
    total_area -
    green_area
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Green area",
        f"{green_area:.2f} km²"
    )

    st.metric(
        "Non-green area",
        f"{non_green_area:.2f} km²"
    )


with col2:

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.pie(
        [
            green_area,
            non_green_area
        ],
        labels=[
            "Green space",
            "Other area"
        ],
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        f"{selected_city} land composition"
    )

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

st.divider()

st.subheader(
    "🏙️ City comparison"
)


if len(city_data) >= 2:

    st.dataframe(
        city_data[
            [
                "city",
                "total_area_km2",
                "green_area_km2",
                "green_coverage_percent"
            ]
        ].rename(
            columns={
                "city": "City",
                "total_area_km2": "Total area (km²)",
                "green_area_km2": "Green area (km²)",
                "green_coverage_percent":
                    "Green coverage (%)"
            }
        ),
        hide_index=True,
        width="stretch"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        city_data["city"],
        city_data["green_coverage_percent"]
    )

    ax.set_ylabel(
        "Green coverage (%)"
    )

    ax.set_title(
        "Green-space coverage by city"
    )

    ax.set_ylim(
        0,
        max(
            city_data["green_coverage_percent"]
        ) * 1.25
    )

    for i, value in enumerate(
        city_data["green_coverage_percent"]
    ):

        ax.text(
            i,
            value + 0.2,
            f"{value:.2f}%",
            ha="center"
        )

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


st.divider()

st.subheader(
    "🤖 U-Net model performance"
)

metric1, metric2, metric3, metric4, metric5 = (
    st.columns(5)
)


with metric1:

    st.metric(
        "Accuracy",
        "98.29%"
    )


with metric2:

    st.metric(
        "Precision",
        "98.94%"
    )


with metric3:

    st.metric(
        "Recall",
        "97.95%"
    )


with metric4:

    st.metric(
        "Dice",
        "98.44%"
    )


with metric5:

    st.metric(
        "IoU",
        "96.92%"
    )


st.caption(
    "Evaluation results obtained from the validation dataset."
)


st.divider()

with st.expander("🔬 Technical information"):

    st.markdown(
        """
        **Data source**

        Sentinel-2 multispectral satellite imagery.

        **Vegetation index**

        NDVI (Normalized Difference Vegetation Index) is used
        as the primary model input.

        **Segmentation model**

        U-Net convolutional neural network for semantic
        segmentation.

        **Patch size**

        128 × 128 pixels.

        **Output**

        A binary vegetation mask identifying predicted
        green-space pixels.

        **Spatial analysis**

        Predictions are clipped to the administrative
        municipality boundary before calculating the
        city-level indicators.

        **Indicators**

        - Total municipality area
        - Detected green-space area
        - Green-space coverage percentage
        """
    )


with st.expander("📁 Generated results"):

    st.write(
        "**City indicators:**"
    )

    st.code(
        str(comparison_file),
        language="text"
    )

    st.write(
        "**Clipped green-space mask:**"
    )

    st.code(
        str(clipped_mask),
        language="text"
    )


st.divider()

st.caption(
    "GreenVision AI · Sentinel-2 · U-Net · "
    "Remote Sensing · Computer Vision"
)
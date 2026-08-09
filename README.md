# Green Space Segmentation AI

An AI and remote-sensing project for estimating and segmenting urban green spaces using Sentinel-2 satellite imagery.

## Project Goal

The goal is to develop a pipeline that can analyze satellite imagery and identify vegetation/green spaces in urban areas.

The project is being developed using Sentinel-2 imagery from the Copernicus Data Space Ecosystem (CDSE).

## Current Pipeline

The project currently includes:

1. Copernicus Data Space authentication
2. Sentinel-2 product search
3. Satellite data download
4. `.SAFE` archive extraction
5. Sentinel-2 band detection
6. NDVI computation
7. ExG (Excess Green Index) computation

## Cities

The current dataset includes:

- Barcelona, Spain
- Oujda, Morocco

## Technologies

- Python
- NumPy
- Pandas
- Rasterio
- Requests
- PyYAML
- python-dotenv
- Sentinel-2
- Copernicus Data Space Ecosystem

## Project Structure

```text
green-space-segmentation-ai/
│
├── config/
│   ├── cities.yaml
│   └── settings.py
│
├── scripts/
│   ├── download_data.py
│   ├── extract_safe.py
│   ├── check_safe.py
│   ├── compute_indices.py
│   ├── preprocess.py
│   ├── create_dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── src/
│   ├── cdse.py
│   ├── downloader.py
│   ├── preprocessing.py
│   ├── training/
│   └── inference/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
└── README.md
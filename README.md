# 🌱 GreenVision AI

### Intelligent Urban Green-Space Analysis from Satellite Imagery

> **Turning satellite imagery into actionable insights for greener, more sustainable cities.**

GreenVision AI is an **AI and remote-sensing project** that uses satellite imagery to automatically detect urban vegetation and turn it into environmental indicators. It combines **computer vision, machine learning, geospatial analysis, and environmental indices** to move beyond "where is the vegetation?" toward questions that support real decisions: *how green is this city, where are its green spaces, and how could that be tracked over time?*

An interactive Streamlit dashboard sits on top of the pipeline, letting anyone browse the results city by city.

---

## 🏢 Project Background

GreenVision AI started as an internship project at the **Centre Régional d'Investissement de l'Oriental (CRI Oriental)**, exploring how satellite imagery and AI could quantify green spaces in urban areas. I've since continued developing it independently, broadening the scope toward computer vision, geospatial analysis, and data-driven urban planning.

> **From an internship problem → to an evolving AI portfolio project.**

## 🌍 Why This Project?

Manually assessing vegetation across a whole city is slow and hard to keep current. Satellite imagery offers a scalable alternative. GreenVision AI explores:

* 🌱 How much of an urban area is covered by vegetation?
* 🗺️ Where are green spaces distributed?
* 📊 How can vegetation coverage be quantified and compared across cities?
* 📈 How could these indicators be tracked over time?

---

## 🎯 Project Objectives

1. **🛰️ Acquire satellite data** — retrieve Sentinel-2 imagery from the Copernicus Data Space Ecosystem.
2. **🌿 Analyze vegetation** — compute NDVI (Normalized Difference Vegetation Index) and ExG (Excess Green Index).
3. **🤖 Detect green spaces with AI** — a U-Net semantic segmentation model, trained on NDVI-derived patches, identifies vegetation pixel by pixel.
4. **📊 Turn detection into insights** — clip predictions to municipality boundaries and calculate green-space coverage, then compare across cities in an interactive dashboard.

---

## 🧠 Pipeline

```text
Copernicus Data Space
        ↓
Sentinel-2 Product Search
        ↓
Satellite Data Download
        ↓
.SAFE Archive Extraction
        ↓
Band Detection & Selection
        ↓
Image Preprocessing
        ↓
NDVI / ExG Calculation
        ↓
Dataset Creation (128×128 patches)
        ↓
U-Net Training
        ↓
Model Evaluation
        ↓
Prediction + Municipality Clipping
        ↓
City Indicators
        ↓
Streamlit Dashboard
```

### Status

* ✅ Copernicus Data Space authentication
* ✅ Sentinel-2 product search
* ✅ Satellite data download
* ✅ `.SAFE` archive extraction
* ✅ Sentinel-2 band detection
* ✅ Image preprocessing
* ✅ NDVI computation
* ✅ ExG computation
* ✅ Dataset creation and segmentation pipeline
* ✅ U-Net training and quantitative evaluation
* ✅ City-level indicators and comparison dashboard
* 🔮 Historical / time-series monitoring
* 🔮 Green-space-per-capita and additional Moroccan cities

> **Note:** 🔮 marks planned, not-yet-started work.

---

## 🤖 Model Performance

Evaluated on the held-out validation split (`scripts/evaluate.py`):

| Metric | Score |
|---|---|
| Accuracy | 98.29% |
| Precision | 98.94% |
| Recall | 97.95% |
| Dice | 98.44% |
| IoU | 96.92% |

---

## 🗺️ Study Areas

* 🇪🇸 **Barcelona, Spain** — a large, diverse urban environment.
* 🇲🇦 **Oujda, Morocco** — a locally relevant Moroccan case study.

Additional Moroccan cities are a planned extension.

---

## 🛠️ Technologies

**Programming & data** — Python, NumPy, Pandas, Rasterio, GeoPandas
**Remote sensing & geospatial** — Sentinel-2, Copernicus Data Space Ecosystem, NDVI, ExG, raster processing
**Machine learning** — TensorFlow/Keras, U-Net, semantic segmentation, computer vision
**App** — Streamlit, Matplotlib
**Tooling** — Requests, PyYAML, python-dotenv, tqdm

---

## 📁 Project Structure

```text
greenvision-ai/
│
├── app.py                       # Streamlit dashboard
├── requirements.txt
│
├── config/
│   ├── cities.yaml               # Study-area definitions
│   └── settings.py
│
├── scripts/
│   ├── download_data.py          # Sentinel-2 search & download
│   ├── check_safe.py
│   ├── extract_safe.py           # .SAFE archive extraction
│   ├── compute_indices.py        # NDVI / ExG
│   ├── create_dataset.py         # Patch dataset for training
│   ├── train.py                  # Entry point → src/training/train_unet.py
│   ├── evaluate.py                # Accuracy / precision / recall / Dice / IoU
│   ├── predict.py
│   └── calculate_city_indicators.py
│
├── src/
│   ├── cdse.py                   # Copernicus Data Space client
│   ├── downloader.py
│   ├── preprocessing.py
│   ├── utils.py
│   ├── training/
│   │   └── train_unet.py
│   └── inference/
│       └── predict_image.py
│
└── data/
    ├── boundaries/                # Municipality boundary files (shapefiles, GeoJSON)
    └── metadata/
```

*(`data/raw/`, `data/processed/`, `data/dataset/`, `models/`, and `outputs/` are generated locally and git-ignored — see `.gitignore`.)*

---

## 🚀 Getting Started

```bash
git clone https://github.com/damchiaya-cyber/greenvision-ai.git
cd greenvision-ai
pip install -r requirements.txt
```

Set your Copernicus Data Space credentials in a local `.env` file (never commit this):

```text
CDSE_USERNAME=your_username
CDSE_PASSWORD=your_password
```

Run the pipeline end to end, then launch the dashboard:

```bash
python -m scripts.download_data
python -m scripts.extract_safe
python -m scripts.compute_indices
python -m scripts.create_dataset
python -m scripts.train
python -m scripts.evaluate
python -m scripts.predict
python -m scripts.calculate_city_indicators

streamlit run app.py
```

---

## 🚀 Future Development

* [ ] Green-space-per-capita calculation
* [ ] Historical / time-series comparisons
* [ ] City-level environmental reports
* [ ] Additional Moroccan cities
* [ ] Improved model generalization
* [ ] Automated environmental reports

---

## 🌱 Personal Motivation

This project reflects my broader interest in using AI and data analysis on real-world problems. Rather than optimizing for a benchmark score, I'm interested in what comes *after* the prediction: what the data shows, how to visualize it, and how it can support better decisions. GreenVision AI is my exploration of that through AI, remote sensing, environmental analysis, and smart-city applications.

---

## 📌 Project Status

**Status:** 🚧 Active development — core pipeline (data acquisition → segmentation → indicators → dashboard) is complete; extensions (time-series tracking, more cities) are in progress.

---

## 👩‍💻 Author

**Aya Addamchi**
Artificial Intelligence & Data

Interested in: 🏥 Healthcare AI · ⚽ Sports Analytics · 🌍 Tourism Intelligence · 🌱 Sustainable Cities · 📊 Business Intelligence

---

⭐ If you find this project interesting, feel free to explore the repository and follow its development.

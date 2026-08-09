# 🌱 GreenVision AI

### Intelligent Urban Green Space Analysis from Satellite Imagery

> **Turning satellite imagery into actionable insights for greener and more sustainable cities.**

GreenVision AI is an **AI and remote-sensing project** that explores how satellite imagery can be used to automatically identify urban green spaces and transform them into meaningful environmental indicators.

The project combines **Computer Vision, Machine Learning, geospatial data analysis, and environmental indices** to move beyond simply detecting vegetation — the long-term goal is to help understand **how green a city is, where green spaces are located, and how they can be monitored over time.**

---

## 🌍 Why This Project?

Urban green spaces play an important role in the environmental quality and livability of cities.

However, assessing vegetation across an entire urban area manually can be time-consuming and difficult to update regularly.

Satellite imagery provides another approach.

Instead of asking:

> *"Where is the vegetation?"*

GreenVision AI aims to explore questions such as:

* 🌱 How much of an urban area is covered by vegetation?
* 🗺️ Where are green spaces distributed?
* 📊 How can vegetation coverage be quantified?
* 📈 How could green-space indicators be monitored over time?
* 🏙️ How could this information support smarter urban planning?

The project therefore treats **AI as a tool for analysis and decision-making**, rather than as an end in itself.

---
## 🏢 Project Background

GreenVision AI originated during my internship at the **Centre Régional d'Investissement de l'Oriental (CRI Oriental)**, where the project idea was proposed as part of an exploration of urban green-space analysis.

The objective was to investigate how **satellite imagery, environmental indices, and Artificial Intelligence** could be used to identify and quantify green spaces within urban areas.

The initial work developed during the internship provided the foundation for this project. I am continuing to develop and improve it as an independent portfolio project, with a broader focus on **Computer Vision, geospatial data analysis, environmental intelligence, and data-driven urban planning**.

> **From an internship problem → to an evolving AI project.**


## 🎯 Project Objectives

The project is being developed around four main objectives:

### 1. 🛰️ Acquire satellite data

Retrieve and process **Sentinel-2 imagery** from the Copernicus Data Space Ecosystem.

### 2. 🌿 Analyze vegetation

Calculate environmental indices such as:

* **NDVI** — Normalized Difference Vegetation Index
* **ExG** — Excess Green Index

These indices provide different ways of identifying and analyzing vegetation.

### 3. 🤖 Detect green spaces with AI

Develop a **semantic segmentation pipeline** capable of identifying vegetation directly from satellite imagery.

The project explores deep-learning approaches such as **U-Net** for pixel-level segmentation.

### 4. 📊 Transform detection into insights

The long-term objective is to transform segmentation results into understandable environmental indicators, such as:

* Green-space coverage
* Green-space distribution
* Green-space-per-capita indicators
* Geographic visualizations
* Comparisons between areas or cities

---

# 🧠 Current Pipeline

The current project pipeline includes:

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
Dataset Creation
        ↓
Green Space Segmentation
        ↓
Model Evaluation
        ↓
Environmental Indicators
```

### Currently implemented

* ✅ Copernicus Data Space authentication
* ✅ Sentinel-2 product search
* ✅ Satellite data download
* ✅ `.SAFE` archive extraction
* ✅ Sentinel-2 band detection
* ✅ Image preprocessing
* ✅ NDVI computation
* ✅ ExG computation
* 🚧 Dataset creation and segmentation pipeline
* 🚧 U-Net training and quantitative evaluation
* 🔮 Interactive environmental analysis platform

> **Note:** Features marked as 🚧 or 🔮 represent ongoing or planned development and are not presented as completed functionality.

---

# 🗺️ Study Areas

The current dataset includes:

### 🇪🇸 Barcelona, Spain

A large and diverse urban environment used as one of the study areas.

### 🇲🇦 Oujda, Morocco

A Moroccan urban study area that provides a locally relevant case for exploring vegetation distribution and urban sustainability.

Future versions of the project may expand the analysis to additional Moroccan cities and other urban environments.

---

# 🛠️ Technologies

### Programming & Data

* Python
* NumPy
* Pandas
* Rasterio

### Remote Sensing & Geospatial Analysis

* Sentinel-2
* Copernicus Data Space Ecosystem
* NDVI
* ExG
* Raster data processing

### Machine Learning

* Deep Learning
* Semantic Segmentation
* U-Net
* Computer Vision

### Development

* Requests
* PyYAML
* python-dotenv

---

# 📁 Project Structure

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
```

---

# 📊 From Detection to Decision Support

One of the main directions of GreenVision AI is to go beyond producing segmentation masks.

The broader vision is to create a system where satellite imagery can be transformed into information that is easier for people to understand and use.

For example:

```text
Satellite Image
      ↓
Vegetation Detection
      ↓
Spatial Analysis
      ↓
Green-Space Indicators
      ↓
Visualization
      ↓
Urban Environmental Insights
```

This could eventually support applications such as:

* Urban sustainability analysis
* Environmental monitoring
* Green-space planning
* City comparisons
* Long-term vegetation monitoring

---

# 🚀 Future Development

The project is designed to evolve from an experimental segmentation pipeline into a more complete **urban environmental intelligence platform**.

Planned improvements include:

* [ ] Complete U-Net training pipeline
* [ ] Quantitative model evaluation
* [ ] IoU and Dice Score evaluation
* [ ] Interactive satellite/segmentation maps
* [ ] Green-space coverage dashboard
* [ ] Green-space-per-capita calculation
* [ ] Historical comparisons
* [ ] City-level environmental reports
* [ ] Additional Moroccan cities
* [ ] Improved model generalization
* [ ] Web-based visualization
* [ ] Automated environmental reports

---

# 🌱 Personal Motivation

This project is part of my broader interest in using **Artificial Intelligence and Data Analysis to understand real-world problems**.

Rather than building an AI model simply to achieve a high accuracy score, I am interested in what happens **after the prediction**:

> **What can we learn from the data?**

> **How can we visualize it?**

> **And how can the result support better decisions?**

GreenVision AI is my exploration of that idea through **AI, remote sensing, environmental analysis, and smart-city applications.**

---

# 📌 Project Status

**Status:** 🚧 Active Development

This project is continuously being improved as I strengthen my skills in:

**Machine Learning · Computer Vision · Data Analysis · Remote Sensing · Geospatial Data · Software Development**

---

## 👩‍💻 Author

**Aya Addamchi**

Artificial Intelligence & Data

Interested in:

🏥 Healthcare AI · ⚽ Sports Analytics · 🌍 Tourism Intelligence · 🌱 Sustainable Cities · 📊 Business Intelligence

---

⭐ If you find this project interesting, feel free to explore the repository and follow its development.

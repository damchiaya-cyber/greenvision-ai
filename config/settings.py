from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

RAW_DATA = DATA_DIR / "raw" / "sentinel"
PROCESSED_DATA = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Create folders automatically
for folder in [
    RAW_DATA,
    PROCESSED_DATA,
    METADATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# =====================================================
# COPERNICUS SETTINGS
# =====================================================

CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/"
    "realms/CDSE/protocol/openid-connect/token"
)

COLLECTION = "SENTINEL-2"

MAX_CLOUD_COVER = 10

MAX_RESULTS = 3

START_DATE = "2024-01-01T00:00:00.000Z"

END_DATE = "2030-01-01T00:00:00.000Z"

SEARCH_RADIUS = 0.10  # ~10 km


# =====================================================
# DATASET SETTINGS
# =====================================================

PATCH_SIZE = 256

OVERLAP = 32

RANDOM_SEED = 42


# =====================================================
# TRAINING SETTINGS
# =====================================================

IMAGE_SIZE = 256

BATCH_SIZE = 8

EPOCHS = 50

LEARNING_RATE = 1e-4
from pathlib import Path
import zipfile

RAW_FOLDER = Path("data/raw/sentinel")

count = 0

for city in RAW_FOLDER.iterdir():

    if not city.is_dir():
        continue

    print(f"\n📁 {city.name}")

    for zip_path in city.glob("*.zip"):

        safe_folder = city / zip_path.stem

        if safe_folder.exists():
            print(f"✓ Already extracted: {zip_path.name}")
            continue

        print(f"Extracting {zip_path.name}...")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(city)

        print("✓ Done")
        count += 1

print(f"\nFinished. {count} archive(s) extracted.")
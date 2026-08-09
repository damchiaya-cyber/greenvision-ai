from pathlib import Path

ROOT = Path("data/raw/sentinel")

for city in ROOT.iterdir():
    if not city.is_dir():
        continue

    print(f"\n===== {city.name} =====")

    for safe in city.glob("*.SAFE"):
        print(f"\n{safe.name}")

        jp2_files = list(safe.rglob("*.jp2"))

        print(f"Found {len(jp2_files)} JP2 files")

        for f in jp2_files[:15]:
            print(f.relative_to(safe))
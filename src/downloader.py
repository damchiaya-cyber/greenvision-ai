from pathlib import Path
import csv
import yaml

from config.settings import RAW_DATA, METADATA_DIR
from src.cdse import CDSEClient


class DatasetDownloader:

    def __init__(self):

        self.client = CDSEClient()

        self.metadata_file = METADATA_DIR / "metadata.csv"

        self.create_metadata_file()


    def create_metadata_file(self):

        if self.metadata_file.exists():
            return

        with open(self.metadata_file, "w", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)

            writer.writerow([
                "city",
                "product_name",
                "date",
                "product_id",
                "path"
            ])


    def load_cities(self):

        with open("config/cities.yaml", "r", encoding="utf-8") as file:

            return yaml.safe_load(file)


    def save_metadata(self,
                      city,
                      product,
                      path):

        with open(self.metadata_file,
                  "a",
                  newline="",
                  encoding="utf-8") as file:

            writer = csv.writer(file)

            writer.writerow([

                city,

                product["Name"],

                product["ContentDate"]["Start"],

                product["Id"],

                path

            ])


    def download_city(self,
                      city_name,
                      latitude,
                      longitude):

        print("\n" + "=" * 60)
        print(city_name)
        print("=" * 60)

        products = self.client.search_products(
            latitude,
            longitude
        )

        if not products:

            print("No products found.")

            return

        self.client.print_products(products)

        city_folder = RAW_DATA / city_name

        city_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for product in products:

            path = self.client.download_product(
                product,
                city_folder
            )

            if path is not None:

                self.save_metadata(
                    city_name,
                    product,
                    path
                )


    def run(self):

        self.client.login()

        cities = self.load_cities()

        for city in cities:

            self.download_city(

                city["name"],

                city["latitude"],

                city["longitude"]

            )

        print("\n")
        print("=" * 60)
        print("Dataset collection finished.")
        print("=" * 60)
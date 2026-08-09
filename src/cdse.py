import os
import requests
from dotenv import load_dotenv

from config.settings import (
    TOKEN_URL,
    CATALOG_URL,
    COLLECTION,
    MAX_CLOUD_COVER,
    MAX_RESULTS,
    SEARCH_RADIUS,
)


class CDSEClient:
    """
    Copernicus Data Space Ecosystem client.

    Handles:
    - Login
    - Search
    - Download
    """

    def __init__(self):
        load_dotenv()

        self.username = os.getenv("CDSE_USERNAME")
        self.password = os.getenv("CDSE_PASSWORD")

        self.token = None
        self.headers = None

    # =====================================================
    # LOGIN
    # =====================================================

    def login(self):

        print("Connecting to Copernicus...")

        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "password",
                "client_id": "cdse-public",
                "username": self.username,
                "password": self.password,
            },
        )

        if response.status_code != 200:
            raise Exception(
                f"Login failed\n\n{response.status_code}\n{response.text}"
            )

        self.token = response.json()["access_token"]

        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }

        print("✓ Login successful")

    # =====================================================
    # POLYGON
    # =====================================================

    def create_polygon(self, latitude, longitude):

        r = SEARCH_RADIUS

        return (
            f"POLYGON(("
            f"{longitude-r} {latitude-r},"
            f"{longitude+r} {latitude-r},"
            f"{longitude+r} {latitude+r},"
            f"{longitude-r} {latitude+r},"
            f"{longitude-r} {latitude-r}"
            f"))"
        )

    # =====================================================
    # SEARCH
    # =====================================================

    def search_products(self, latitude, longitude):

        self.login()
        polygon = self.create_polygon(latitude, longitude)

        filter_query = (
            f"Collection/Name eq '{COLLECTION}' "
            f"and contains(Name,'MSIL2A') "
            f"and Attributes/OData.CSC.DoubleAttribute/any(a:"
            f"a/Name eq 'cloudCover' "
            f"and a/OData.CSC.DoubleAttribute/Value lt {MAX_CLOUD_COVER}) "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')"
        )

        params = {
            "$filter": filter_query,
            "$orderby": "ContentDate/Start desc",
            "$top": MAX_RESULTS,
        }

        response = requests.get(
            CATALOG_URL,
            headers=self.headers,
            params=params,
        )

        if response.status_code != 200:

            raise Exception(
                f"\nSearch failed\n\n"
                f"{response.status_code}\n"
                f"{response.text}"
            )

        data = response.json()

        return data.get("value", [])

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    def print_products(self, products):

        print()

        print(f"{len(products)} products found\n")

        for i, product in enumerate(products, start=1):

            print(f"[{i}]")

            print(product["Name"])

            print(product["ContentDate"]["Start"])

            print()

    # =====================================================
    # DOWNLOAD URL
    # =====================================================

    def get_download_url(self, product_id):

        return (
            "https://zipper.dataspace.copernicus.eu/odata/v1/"
            f"Products({product_id})/$value"
        )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    def download_product(
        self,
        product,
        destination_folder,
    ):

        self.login()
        os.makedirs(destination_folder, exist_ok=True)

        filename = product["Name"] + ".zip"

        destination = os.path.join(
            destination_folder,
            filename,
        )

        if os.path.exists(destination):

            print(f"✓ Already downloaded : {filename}")

            return destination

        url = self.get_download_url(product["Id"])

        print(f"Downloading {filename}")

        response = requests.get(
            url,
            headers=self.headers,
            stream=True,
        )

        if response.status_code != 200:

            print("Download failed")

            print(response.text)

            return None

        total = int(
            response.headers.get(
                "content-length",
                0,
            )
        )

        downloaded = 0

        with open(destination, "wb") as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(chunk)

                    downloaded += len(chunk)

                    if total:

                        percent = downloaded / total * 100

                        print(
                            f"\r{percent:5.1f} %",
                            end="",
                            flush=True,
                        )

        print()

        print("✓ Download completed")

        return destination
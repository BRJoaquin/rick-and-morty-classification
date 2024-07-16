import torch
from torch.utils.data import Dataset
import requests
from urllib.parse import urlunparse, urlencode
import pandas as pd
from PIL import Image
from io import BytesIO
import os


class RichAndMortyDataSet(Dataset):
    """
    A PyTorch Dataset class for loading and transforming data related to characters from the Rick and Morty series.

    Attributes:
        base_url (str): Base URL for the Rick and Morty API.
        img_folder (str): Directory path where images are stored.
        data_df (DataFrame): Pandas DataFrame containing character data.

    Methods:
        __len__: Returns the number of items in the dataset.
        __getitem__: Retrieves and returns data at the specified index.
        exists_data: Checks if the dataset file exists.
        get_character_url: Constructs the URL for API requests.
        parse_results: Parses API response data into a structured format.
        retrive_tabular_data: Fetches data from the API.
        prepare_data: Encodes categorical data fields.
    """

    def __init__(self, path="data", transform=None, base_url="rickandmortyapi.com"):
        # Create the folder if it does not exist
        os.makedirs(path, exist_ok=True)
        self.transform = transform
        self.base_url = base_url
        self.img_folder = os.path.join(path, "images")
        os.makedirs(self.img_folder, exist_ok=True)

        data_path = os.path.join(path, "data.csv")
        if self._exists_data(path):
            self.data_df = pd.read_csv(data_path)
        else:
            self.data_df = self._retrive_tabular_data()
            self._prepare_data()
            self.data_df.to_csv(data_path, index=False)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # Get row
        row = self.data_df.iloc[idx]
        # Get species and image
        species_t = torch.tensor(row["species"])  # Convert to tensor
        image_url = row["image_url"]  # Remote URL of the image
        image_name = row["image_name"]  # Image name
        image_local_path = self.img_folder + "/" + image_name  # Local path of the image

        # Load image
        if os.path.exists(image_local_path):
            # If the image is already downloaded, load it
            image = Image.open(image_local_path)
        else:
            # If the image is not downloaded, download it and save it
            image_content = requests.get(image_url).content
            image = Image.open(BytesIO(image_content))
            if image.mode != "RGB":
                image = image.convert("RGB")
            # Resize image
            image = image.resize((250, 250))
            image.save(image_local_path)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return image, species_t

    def _exists_data(self, path):
        return os.path.exists(path + "/data.csv")

    def _get_character_url(self, page=1):
        protocol = "https"
        path = "api/character"
        params = ""
        query = {"page": page}

        query_string = urlencode(query)

        return urlunparse((protocol, self.base_url, path, params, query_string, ""))

    def _parse_results(self, results):
        characters = []
        for character in results:
            characters.append(
                {
                    "species": character["species"],
                    "image_url": character["image"],
                    "image_name": os.path.basename(character["image"]),
                }
            )
        return characters

    def _retrive_tabular_data(self):
        page = 1
        characters_list = []

        while True:
            response = requests.get(self._get_character_url(page))

            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break

            data = response.json()

            characters_list.extend(self._parse_results(data["results"]))

            if data["info"]["next"] is None:
                break

            page += 1

        return pd.DataFrame(characters_list)

    def _prepare_data(self):
        self.data_df["species_c"] = pd.Categorical(self.data_df["species"])
        self.data_df["species"] = self.data_df["species_c"].cat.codes

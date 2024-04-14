try:
    from openai import OpenAI
except ImportError:
    print(
        "openai not installed, please install it using `pip install openai` to use this plugin"
    )
import base64
import io
import os

import requests
from PIL import Image


def resize_image(image: Image, max_side: int = 512):
    """
    Resize the image to max_side if any of the sides is greater than max_side
    """
    # get the image shape
    width, height = image.size
    if max(width, height) > max_side:
        # resize the image to max_side
        # keeping the aspect ratio
        if width > height:
            new_width = max_side
            new_height = int(height * max_side / width)
        else:
            new_height = max_side
            new_width = int(width * max_side / height)
        return image.resize((new_width, new_height))
    return image


def encode_image(image: Image):
    """
    Convert the image to base64
    """
    # create a buffer to store the image
    buffer = io.BytesIO()
    # save the image to the buffer
    image.save(buffer, format="JPEG")
    # convert the image to base64
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class PromptClient:
    def __init__(self, url: str) -> None:
        self.client = OpenAI(
            base_url=url,
            api_key=os.getenv("OPENAI_API_KEY", "sk-no-key-required"),
        )
        self.base_settings = {"cache_prompt": True, "temperature": 0.01}
        self.headers = {
            "accept-language": "en-US,en",
            "content-type": "application/json",
        }

    def get_prompt(self):
        raise NotImplementedError


class ImageDescription:
    """A client to interact with the LLaMA image description API.
    The API is used to generate short phrases that describe an image.

    Methods:
        summarise_image(image: Image) -> str:
            Summarise the image using the LLaMA API.
    """

    def __init__(self, url: str = "http://localhost:8080"):
        """Initialise the client with the base URL of the LLaMA API.
        Args:
            url (str): The base URL of the LLaMA API.
        """
        """TODO: migrate to OpenAI API when available"""
        if url is None:
            url = "http://localhost:8080/"
        self.url = url
        self.headers = {
            "accept-language": "en-GB,en",
            "content-type": "application/json",
        }
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.session = requests.Session()

    def get_prompt(self):
        return """You're an AI assistant that describes images using short phrases.
        The image is shown below.
        \nIMAGE:[img-10]
        \nASSISTANT:"""

    def summarise_image(self, image: Image):
        """Summarise the image using the LLaMA API.
        Args:
            image (Image): The image to summarise.
        Returns:
            str: The description of the image.
        """
        b64image = encode_image(resize_image(image))

        json_body = {
            "stream": False,
            "n_predict": 300,
            "temperature": 0.1,
            "repeat_last_n": 78,
            "image_data": [{"data": b64image, "id": 10}],
            "cache_prompt": True,
            "top_k": 40,
            "top_p": 1,
            "min_p": 0.05,
            "tfs_z": 1,
            "typical_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "mirostat": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
            "grammar": "",
            "n_probs": 0,
            "min_keep": 0,
            "api_key": "",
            "slot_id": 0,
            "stop": ["</s>", "Llama:", "User:"],
            "prompt": self.get_prompt(),
        }

        response = self.session.post(
            f"{self.url}/completion",
            json=json_body,
            headers=self.headers,
            stream=False,
        )
        if response.status_code != 200:
            print(f"Failed to summarise image: {response.content}")
            return None
        return response.json()["content"].strip()


class VideoSummary(PromptClient):
    """A client to interact with the LLaMA video summarisation API.
    The API is used to generate a summary of a video based on image descriptions.

    Methods:
        summarise_video(image_descriptions: list[str]) -> str:
            Summarise the video using the LLaMA API.
    """

    def __init__(self, url: str = "http://localhost:8080/v1"):
        """Initialise the client with the base URL of the LLaMA API.
        Args:
            url (str): The base URL of the LLaMA API."""
        if url is None:
            url = "http://localhost:8080/v1"
        super().__init__(url)

    def get_prompt(self):
        return """You're an AI assistant that summarises videos based on image descriptions.
        Combine image descriptions into a coherent summary of the video."""

    def summarise_video(self, image_descriptions: list[str]):
        """Summarise the video using the LLaMA API.
        Args:
            image_descriptions (list[str]): The descriptions of the images in the video.
        Returns:
            str: The summary of the video.
        """
        return self.client.chat.completions.create(
            model="LLaMA_CPP",
            messages=[
                {
                    "role": "system",
                    "content": self.get_prompt(),
                },
                {"role": "user", "content": "\n".join(image_descriptions)},
            ],
            max_tokens=300,
        )

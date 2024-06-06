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
    Resize the image to max_side if any of the sides is greater than max_side.
    If max_side is None, the image is returned as is.
    """
    # get the image shape
    if max_side is None:
        return image
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


class ImageDescription:
    """A client to interact with the image description API.
    The API is used to generate short phrases that describe an image.

    Methods:
        summarise_image(image: Image) -> str:
            Summarise the image using the LLaMA API.
    """

    def __init__(self, url: str) -> None:
        if url is None:
            url = "http://localhost:8080/"
        self.url = url

    def summarise_image(self, image: Image) -> str:
        """Summarise the image
        Args:
            image (Image): The image to summarise.
        Returns:
            str: The description of the image.
        """
        ...


class ImageDescriptionDefault(ImageDescription):
    """A client to interact with the LLaMA image description API.
    The API is used to generate short phrases that describe an image.

    Methods:
        summarise_image(image: Image) -> str:
            Summarise the image using the LLaMA API.
    """

    def __init__(self, url: str = "http://localhost:8080/completion"):
        """Initialise the client with the base URL of the LLaMA API.
        Args:
            url (str): The base URL of the LLaMA API.
        """
        """TODO: migrate to OpenAI API when available"""
        super().__init__(url)
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

    def summarise_image(self, image: Image) -> str:
        """Summarise the image using the LLaMA API.
        Args:
            image (Image): The image to summarise.
        Returns:
            str: The description of the image.
        """
        b64image = encode_image(resize_image(image))

        json_body = {
            "model": os.getenv("OPENAI_MODEL", "LLaVA_CPP"),
            "stream": False,
            "n_predict": 1000,
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
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "slot_id": 0,
            "stop": ["</s>", "Llama:", "User:"],
            "prompt": self.get_prompt(),
        }
        response = self.session.post(
            f"{self.url}",
            json=json_body,
            headers=self.headers,
            stream=False,
        )
        if response.status_code != 200:
            print(f"Failed to summarise image: {response}")
            return None
        res = response.json()
        if "choices" in res:
            return res["choices"][0]["text"].strip()
        elif "content" in res:
            return res["content"].strip()
        raise ValueError(f"Failed to summarise image: unknown response format: {res}")


class ImageDescriptionOpenAI(ImageDescription):
    def __init__(self, url: str = "http://localhost:8080"):
        super().__init__(url)
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.client = OpenAI(base_url=self.url, api_key=api_key)

    def summarise_image(self, image: Image) -> str:
        b64image = encode_image(resize_image(image))
        completion = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "LLaVA_CPP"),
            messages=[
                {
                    "role": "system",
                    "content": "This is a chat between a user and an assistant. "
                    "The assistant is helping the user to describe an image.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64image}"},
                        },
                    ],
                },
            ],
            max_tokens=300,
            stream=False,
        )
        return completion["choices"][0]["message"]["content"]


class VideoSummary:
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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import base64
import io
import os
from abc import ABC, abstractmethod

from PIL import Image


def resize_image(image: Image, max_side: int | None = 512) -> Image:
    if max_side is None:
        return image
    width, height = image.size
    if max(width, height) > max_side:
        if width > height:
            new_width = max_side
            new_height = int(height * max_side / width)
        else:
            new_height = max_side
            new_width = int(width * max_side / height)
        return image.resize((new_width, new_height))
    return image


def encode_image(image: Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_image(self, image: Image.Image | str) -> str:
        raise NotImplementedError


class OpenAIMultimodalEmbedder(BaseEmbedder):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        if OpenAI is None:
            raise ImportError("pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY")

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.client = OpenAI(**kwargs)

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding

    def embed_image(self, image: Image.Image | str) -> str:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        b64_image = encode_image(resize_image(image, 512))

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=500,
        )

        return response.choices[0].message.content

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from video_sampler.integrations.llava_chat import (
    ImageDescriptionDefault,
    ImageDescriptionOpenAI,
    VideoSummary,
    encode_image,
    resize_image,
)


@pytest.fixture
def sample_image():
    # Create a simple 100x100 red image
    return Image.new("RGB", (100, 100), color="red")


def test_resize_image(sample_image):
    resized = resize_image(sample_image, max_side=50)
    assert resized.size == (50, 50)

    # Test when image is smaller than max_side
    small_image = resize_image(sample_image, max_side=200)
    assert small_image.size == (100, 100)


def test_encode_image(sample_image):
    encoded = encode_image(sample_image)
    assert isinstance(encoded, str)
    assert encoded.startswith("/9j/")  # JPEG header in base64


@patch("requests.Session.post")
def test_image_description_default(mock_post, sample_image):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"text": "A red square"}]}
    mock_post.return_value = mock_response

    describer = ImageDescriptionDefault()
    description = describer.summarise_image(sample_image)

    assert description == "A red square"
    mock_post.assert_called_once()


@patch("video_sampler.integrations.llava_chat.OpenAI")
def test_image_description_openai(mock_openai_class, sample_image):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="An image of a red square"))]
    )
    mock_openai_class.return_value = mock_client

    describer = ImageDescriptionOpenAI()
    description = describer.summarise_image(sample_image)
    assert description == "An image of a red square"
    mock_client.chat.completions.create.assert_called_once()


@patch("video_sampler.integrations.llava_chat.OpenAI")
def test_video_summary(mock_openai_class):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(message=MagicMock(content="A video showing various red objects"))
        ]
    )
    mock_openai_class.return_value = mock_client

    summarizer = VideoSummary()
    summary = summarizer.summarise_video(
        ["A red square", "A red circle", "A red triangle"]
    )
    summary = summary.choices[0].message.content
    assert summary == "A video showing various red objects"
    mock_client.chat.completions.create.assert_called_once()


def test_image_description_default_error():
    with patch("requests.Session.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        describer = ImageDescriptionDefault()
        description = describer.summarise_image(Image.new("RGB", (10, 10)))

        assert description is None


def test_image_description_default_unknown_response():
    with patch("requests.Session.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unknown_key": "value"}
        mock_post.return_value = mock_response

        describer = ImageDescriptionDefault()
        with pytest.raises(ValueError):
            describer.summarise_image(Image.new("RGB", (10, 10)))

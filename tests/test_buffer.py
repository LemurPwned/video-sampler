import random

import pytest
from PIL import Image, ImageDraw

from video_sampler.buffer import GridBuffer, HashBuffer, create_buffer

random.seed(42)


def generate_random_image():
    width = 300
    height = 300
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    image = Image.new("RGB", (width, height), color)

    # Add some random shapes to increase variance
    draw = ImageDraw.Draw(image)
    for _ in range(random.randint(2, 10)):
        shape = random.choice(
            ["rectangle", "ellipse", "line", "point", "arc", "chord", "pie"]
        )
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        x0 = random.randint(0, width)
        y0 = random.randint(0, height)
        x1 = random.randint(x0, width)
        y1 = random.randint(0, height)
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        coords = [x0, y0, x1, y1]
        if shape == "rectangle":
            draw.rectangle(coords, fill=color)
        elif shape == "ellipse":
            draw.ellipse(coords, fill=color)
        else:
            draw.line(coords, fill=color, width=random.randint(1, 10))

    return image


@pytest.fixture
def random_image():
    return generate_random_image()


@pytest.fixture
def fixed_img_pair():
    return (
        Image.open("./tests/assets/images/imgA.jpg"),
        Image.open("./tests/assets/images/imgB.jpg"),
    )


@pytest.mark.parametrize(
    "buffer_type, buffer_size",
    [
        (GridBuffer, 10),
        (GridBuffer, 20),
        (HashBuffer, 10),
        (HashBuffer, 20),
    ],
)
def test_buffer_add_single_image(random_image, buffer_type, buffer_size):
    buffer = buffer_type(size=buffer_size)
    metadata = {"test_key": "test_value"}
    buffer.add(random_image, metadata)
    assert len(buffer) == 1
    _, (stored_image, stored_metadata) = buffer.popitem()
    assert stored_image == random_image
    assert stored_metadata == metadata


@pytest.mark.parametrize(
    "buffer_size, num_images",
    [
        (10, 5),
        (20, 15),
        (15, 25),
        (5, 100),
    ],
)
def test_hash_buffer_add_multiple_images(buffer_size, num_images):
    buffer = HashBuffer(size=buffer_size, debug_flag=True, hash_size=8)
    for _ in range(num_images):
        buffer.add(generate_random_image(), {})
    assert len(buffer) == min(num_images, buffer_size)


@pytest.mark.parametrize(
    "buffer_size, num_images",
    [
        (10, 5),
        (20, 15),
        (15, 25),
        (5, 100),
    ],
)
def test_grid_buffer_add_multiple_images(buffer_size, num_images):
    buffer = GridBuffer(size=buffer_size, debug_flag=True, hash_size=8, max_hits=30)
    for _ in range(num_images):
        buffer.add(generate_random_image(), {})
    assert len(buffer) == min(num_images, buffer_size)


@pytest.mark.parametrize(
    "buffer_type, buffer_size",
    [
        (GridBuffer, 10),
        (GridBuffer, 20),
        (HashBuffer, 10),
        (HashBuffer, 20),
    ],
)
def test_buffer_clear(buffer_type, buffer_size):
    buffer = buffer_type(size=buffer_size)
    buffer.add(generate_random_image(), {})
    buffer.clear()
    assert len(buffer) == 0


@pytest.mark.parametrize(
    "buffer_type, buffer_size",
    [
        (GridBuffer, 5),
        (GridBuffer, 10),
        (HashBuffer, 5),
        (HashBuffer, 10),
    ],
)
def test_grid_buffer_same_image(fixed_img_pair, buffer_type, buffer_size):
    buffer = buffer_type(size=buffer_size)
    imgA, _ = fixed_img_pair
    metadata1 = {"timestamp": 1}
    metadata2 = {"timestamp": 2}
    buffer.add(imgA, metadata1)
    buffer.add(imgA, metadata2)
    assert len(buffer) == 1
    _, (stored_image, stored_metadata) = buffer.popitem()
    assert stored_image == imgA
    assert stored_metadata == metadata1  # first is kept


@pytest.mark.parametrize(
    "buffer_type, buffer_size",
    [
        (GridBuffer, 10),
        (GridBuffer, 20),
        (HashBuffer, 10),
        (HashBuffer, 20),
    ],
)
def test_grid_buffer_same_image_different_metadata(
    fixed_img_pair, buffer_type, buffer_size
):
    buffer = buffer_type(size=buffer_size)
    imgA, _ = fixed_img_pair
    buffer.add(imgA, {"data": "item1"})
    buffer.add(imgA, {"data": "item2"})
    assert len(buffer) == 1


@pytest.mark.parametrize(
    "buffer_type, buffer_size",
    [
        (GridBuffer, 10),
        (GridBuffer, 20),
        (HashBuffer, 10),
        (HashBuffer, 20),
    ],
)
def test_grid_buffer_same_image_same_metadata(fixed_img_pair, buffer_type, buffer_size):
    buffer = buffer_type(size=buffer_size)
    imgA, _ = fixed_img_pair
    buffer.add(imgA, {"data": "item1"})
    buffer.add(imgA, {"data": "item1"})
    assert len(buffer) == 1


@pytest.mark.parametrize(
    "buffer_type, buffer_size",
    [
        (GridBuffer, 10),
        (GridBuffer, 20),
        (HashBuffer, 10),
        (HashBuffer, 20),
    ],
)
def test_grid_buffer_same_metadata_different_image(
    fixed_img_pair, buffer_type, buffer_size
):
    buffer = buffer_type(size=buffer_size)
    imgA, imgB = fixed_img_pair
    buffer.add(imgA, {"data": "item1"})
    buffer.add(imgB, {"data": "item1"})
    assert len(buffer) == 2


@pytest.mark.parametrize(
    "buffer_type, buffer_size",
    [
        (GridBuffer, 10),
        (GridBuffer, 20),
        (HashBuffer, 10),
        (HashBuffer, 20),
    ],
)
def test_grid_buffer_different_image(fixed_img_pair, buffer_type, buffer_size):
    buffer = buffer_type(size=buffer_size, debug_flag=True)
    imgA, imgB = fixed_img_pair
    buffer.add(imgA, {"data": "item1"})
    buffer.add(imgB, {"data": "item2"})
    assert len(buffer) == 2


def test_create_buffer_function():
    def assert_buffer_properties(buffer, expected_type, size, hash_size, max_hits=None):
        assert isinstance(buffer, expected_type)
        assert buffer.max_size == size
        assert buffer.hash_size == hash_size
        if max_hits is not None:
            assert buffer.max_hits == max_hits

    buffer = create_buffer(
        {
            "type": "grid",
            "size": 10,
            "hash_size": 8,
            "max_hits": 30,
            "grid_x": 3,
            "grid_y": 3,
        }
    )
    assert_buffer_properties(buffer, GridBuffer, 10, 8, 30)

    buffer = create_buffer({"type": "hash", "size": 10, "hash_size": 8})
    assert_buffer_properties(buffer, HashBuffer, 10, 8)


def test_create_invalid_buffer():
    with pytest.raises(AssertionError):
        create_buffer({"type": "invalid"})
    with pytest.raises(AssertionError):
        create_buffer({"type": "hash", "hash_size": 3, "size": -1})
    with pytest.raises(AssertionError):
        create_buffer({"type": "grid", "size": 0})

    with pytest.raises(AssertionError):
        create_buffer({"type": "grid", "hash_size": 3, "size": 0})

    with pytest.raises(AssertionError):
        create_buffer(
            {
                "type": "grid",
                "grid_x": 3,
                "grid_y": 3,
                "max_hits": 30,
                "hash_size": 3,
                "size": -1,
            }
        )

    with pytest.raises(AssertionError):
        create_buffer({"type": "hash", "hash_size": 3, "size": -1})

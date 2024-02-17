import pytest


@pytest.fixture(scope="session")
def base_video() -> str:
    return "videos/SmolCat.mp4"


@pytest.fixture
def random_video() -> str:
    return "https://www.youtube.com/watch?v=W86cTIoMv2U"

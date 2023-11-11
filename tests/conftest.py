import pytest 

@pytest.fixture(scope="session")
def base_video() -> str:
    return "videos/SmolCat.mp4"

from collections import namedtuple
from collections.abc import Iterable

import requests
from requests.exceptions import RequestException

from ..logging import Color, console

subtitle_line = namedtuple(
    "subtitle_line", ["start_time", "end_time", "keyword", "content"]
)


def download_sub(sub_url: str, max_retries: int = 2):
    """Download a VTT subtitle file to a string with retry mechanism."""
    for _ in range(max_retries):
        try:
            response = requests.get(url=sub_url)
            response.raise_for_status()
            return parse_srt_subtitle(response.text)
        except RequestException as e:
            console.print(f"Download failed: {str(e)}", style=f"bold {Color.red.value}")
    return None


def parse_srt_subtitle(srt_content):
    """Parse a SRT subtitle file to a list of subtitle segments."""
    try:
        import pysrt
    except ImportError:
        raise ImportError("To use this feature install pysrt by 'pip install pysrt'")

    subtitle_list = []
    if not srt_content:
        return subtitle_list
    subs = pysrt.from_string(srt_content)
    for sub in subs:
        time = (sub.start.ordinal, sub.end.ordinal)
        content = sub.text
        subtitle_list.append((time, content))
    return subtitle_list


class KeywordExtractor:
    """
    Extracts keywords from subtitles using spaCy.

    Args:
        keywords (list[str]): List of keywords to extract.

    Attributes:
        keywords (list[str]): List of keywords to extract.
        nlp: spaCy language model for text processing.
        lemmatized_keywords (set[str]): Set of lemmatized keywords.

    Methods:
        generate_segments: Captures keyword segments from a list of subtitles.

    """

    def __init__(self, keywords: list[str]) -> None:
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "To use this feature install spacy by 'pip install spacy'"
            ) from e

        self.keywords = keywords
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        self.lemmatized_keywords = {
            tok.lemma_ for tok in self.nlp(" ".join(self.keywords))
        }
        console.print(
            f"Keyword capture initialised with: {keywords}",
            style=f"bold {Color.magenta.value}",
        )

    def generate_segments(
        self, subtitle_list: list[tuple[int, int, str]]
    ) -> Iterable[subtitle_line]:
        """
        Captures keyword segments from a list of subtitles.

        Args:
            subtitle_list (list[tuple[int, int, str]]): List of subtitles in the format
                (start_time, end_time, content).

        Yields:
            subtitle_line: A named tuple representing a keyword segment in the format
                (start_time, end_time, lemma, content).

        """
        for (start_time, end_time), content in subtitle_list:
            doc = self.nlp(content.lower())
            for lemma in doc:
                if lemma.lemma_ in self.lemmatized_keywords:
                    console.print(
                        f"Keyword {lemma.lemma_}: {start_time} - {end_time}",
                        style=f"bold {Color.green.value}",
                    )
                    yield subtitle_line(start_time, end_time, lemma, content)
                    break


def create_extractor(config: dict):
    if config["type"] == "keyword":
        return KeywordExtractor(**config["args"])

    raise NotImplementedError(f"{config['type']} not implemented yet")

from collections import namedtuple
from collections.abc import Iterable

import pysrt
import requests
import spacy

from ..logging import Color, console

subtitle_line = namedtuple(
    "subtitle_line", ["start_time", "end_time", "keyword", "content"]
)


def download_sub(sub_url: str):
    """Download a VTT subtitle file to a string."""
    response = requests.get(url=sub_url)
    return parse_srt_subtitle(response.text)


def parse_vtt_subtitle(vtt_content):
    subtitle_list = []
    lines = vtt_content.split("\n")
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.isdigit() and i + 1 < len(lines):
            time = lines[i + 1].strip()
            content = lines[i + 2].strip()
            subtitle_list.append((time, content))
    return subtitle_list


def parse_srt_subtitle(srt_content):
    subtitle_list = []
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
        capture_keyword_segments: Captures keyword segments from a list of subtitles.

    """

    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        self.lemmatized_keywords = {
            tok.lemma_ for tok in self.nlp(" ".join(self.keywords))
        }

    def capture_keyword_segments(
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

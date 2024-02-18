from collections import namedtuple
from collections.abc import Iterable

import nltk
import pysrt
import requests
from nltk.stem import WordNetLemmatizer

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
    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords
        # use the WordNetLemmatizer to lemmatize the words,
        # Stemming is not used because it is too aggressive
        self.wnl = WordNetLemmatizer()
        # expand the keywords with their synonyms
        self.lemmatized_keywords = {
            self.wnl.lemmatize(keyword) for keyword in self.keywords
        }

    def capture_keyword_segments(
        self, subtitle_list: list[tuple[int, int, str]]
    ) -> Iterable[subtitle_line]:
        for (start_time, end_time), content in subtitle_list:
            tokens = nltk.word_tokenize(content.lower())
            # TODO do tagging
            for token in tokens:
                token = self.wnl.lemmatize(token)
                if token in self.lemmatized_keywords:
                    console.print(
                        f"Keyword {token}: {start_time} - {end_time}",
                        style=f"bold {Color.green.value}",
                    )
                    yield subtitle_line(start_time, end_time, token, content)
                    break

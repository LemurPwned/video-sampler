from collections.abc import Iterable

from ..language.keyword_capture import download_sub

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print(
        "yt-dlp not installed, please install it using `pip install yt-dlp` to use this plugin"
    )


# this one generates urls based on the search query
# or the playlist url or channel url
def best_video_best_audio(ctx):
    """Taken from the yt-dlp documentation as-is"""
    """Select the best video and the best audio that won't result in an mkv.
    NOTE: This is just an example and does not handle all cases"""

    # formats are already sorted worst to best
    formats = ctx.get("formats")[::-1]

    # acodec='none' means there is no audio
    best_video = next(
        f for f in formats if f["vcodec"] != "none" and f["acodec"] == "none"
    )

    # find compatible audio extension
    audio_ext = {"mp4": "m4a", "webm": "webm"}[best_video["ext"]]
    # vcodec='none' means there is no video
    best_audio = next(
        f
        for f in formats
        if (f["acodec"] != "none" and f["vcodec"] == "none" and f["ext"] == audio_ext)
    )

    # These are the minimum required fields for a merged format
    yield {
        "format_id": f'{best_video["format_id"]}+{best_audio["format_id"]}',
        "ext": best_video["ext"],
        "requested_formats": [best_video, best_audio],
        # Must be + separated list of protocols
        "protocol": f'{best_video["protocol"]}+{best_audio["protocol"]}',
    }


def best_video_only(ctx):
    """Just best video -- save bandwidth"""
    # formats are already sorted worst to best
    formats = ctx.get("formats")[::-1]

    # acodec='none' means there is no audio
    best_video = next(f for f in formats if f["vcodec"] != "none")
    # These are the minimum required fields for a merged format
    yield {
        "format_id": f'{best_video["format_id"]}',
        "ext": best_video["ext"],
        "requested_formats": [best_video],
        # Must be + separated list of protocols
        "protocol": f'{best_video["protocol"]}',
    }


def no_shorts(info, *, incomplete):
    """Filter out short videos"""
    if url := info.get("url", ""):
        if "/shorts" in url:
            return "This is a short video"


class YTDLPPlugin:
    """
    A plugin for yt-dlp to generate URLs and corresponding titles from the given URL.

    Methods:
        generate_urls(url, extra_yt_constr_args=None, extra_info_extract_opts=None) -> Iterable[str]:
            Generates URLs and corresponding titles from the given URL.

    """

    def __init__(self, ie_key: str = "Generic"):
        """
        Initialize the YTDLPPlugin instance.
        """
        self.ie_key = ie_key
        self.ydl_opts = {
            "format": best_video_only,
        }

    def generate_urls(
        self,
        url: str,
        extra_info_extract_opts: dict = None,
        get_subs: bool = False,
    ) -> Iterable[tuple[str, str, str | None]]:
        """Generate URLs and download subtitles for a given video URL.

        Args:
            url (str): The URL of the video to download subtitles for.
            extra_info_extract_opts (dict, optional): Additional options for extracting video information.

        Yields:
            tuple: A tuple containing the video title, video format URL, and downloaded subtitles.
        """
        if extra_info_extract_opts is None:
            extra_info_extract_opts = {}
        if get_subs:
            extra_info_extract_opts |= self.get_subtitles_opts()
        extr_args = {"ie_key": self.ie_key} if "ytsearch" not in url else {}

        def preproc_entry(info):
            req_format = info["requested_formats"][0]
            subs = None
            if get_subs and "requested_subtitles" in info:
                subs = download_sub(
                    list(info["requested_subtitles"].values())[0]["url"]
                )
            return info["title"], req_format["url"], subs

        with YoutubeDL(params=(self.ydl_opts | extra_info_extract_opts)) as ydl:
            info = ydl.extract_info(url, download=False, **extr_args)
            if "entries" not in info:
                yield preproc_entry(info)
            else:
                for entry in info.get("entries", []):
                    if not entry:
                        continue
                    yield preproc_entry(entry)

    def get_subtitles_opts(self) -> dict:
        return {
            "postprocessors": [
                {
                    "format": "srt",
                    "key": "FFmpegSubtitlesConvertor",
                    "when": "before_dl",
                }
            ],
            "format": best_video_only,
            "subtitleslangs": ["en.*"],
            "writeautomaticsub": True,
            "writesubtitles": True,
        }

from collections.abc import Iterable

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
    """Just best video"""
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


class YTDLPPlugin:
    def __init__(self, ie_key: str = "Generic"):
        self.ie_key = ie_key
        # for now we only support video
        self.ydl_opts = {
            "format": best_video_only,
        }

    def generate_urls(self, url: str) -> Iterable[str]:
        extr_args = {"ie_key": self.ie_key} if "ytsearch" not in url else {}
        with YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False, **extr_args)
            if "entries" not in info:
                yield info["title"], info["url"]
            else:
                for entry in info.get("entries", []):
                    req_format = entry["requested_formats"][0]
                    yield entry["title"], req_format["url"]

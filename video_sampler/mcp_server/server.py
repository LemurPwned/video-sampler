import asyncio
import json
import os
from typing import Any
from urllib.parse import urlparse

import mcp.server.stdio
from mcp import types
from mcp.server import Server
from mcp.server.models import InitializationOptions

from ..integrations.yt_dlp_plugin import YTDLPPlugin
from ..language.keyword_capture import KeywordExtractor, parse_srt_subtitle
from .cache_server import VideoCacheServer


class VideoNavigationMCPServer:
    """MCP Server for video navigation with search capabilities."""

    def __init__(self):
        self.server = Server("video-navigation")
        self.ytdlp_plugin = None
        self.keyword_extractor = None
        self.video_sampler = None
        self.video_database: dict[str, dict] = {}  # Store video metadata
        self.setup_server()
        self.video_cache_server = VideoCacheServer()

    def setup_server(self):
        """Set up the MCP server with resources and tools."""

        # Resource: Video
        @self.server.list_resources()
        async def list_resources() -> list[types.Resource]:
            return [
                types.Resource(**video.model_dump())
                for video in self.video_cache_server.list_all_videos()
            ]

        @self.server.read_resource()
        async def read_resource(video_id_uri: str) -> str:
            """Read a video or imageset resource."""
            if video_id_uri.startswith("video://"):
                video_id = video_id_uri[8:]  # Remove "video://" prefix
                video = self.video_cache_server.get_by_id(video_id)
                if video:
                    return json.dumps(video.model_dump(), indent=2)
                else:
                    raise ValueError(f"Video {video_id} not found")
            else:
                raise ValueError(f"Unknown resource URI: {uri}")

        # Tool: Search
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            """list available tools."""
            return [
                types.Tool(
                    name="search",
                    description=(
                        "Search for content in videos using keyword search (subtitles) or visual search (keyframes). "
                        "If video_path is None, searches over the whole loaded video set. "
                        "Supports YouTube URLs which will be processed automatically."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (keywords for subtitle search or description for visual search)",
                            },
                            "video_path": {
                                "type": ["string", "null"],
                                "description": "Path to video file or YouTube URL. If None, searches over whole set.",
                            },
                            "start": {
                                "type": ["number", "null"],
                                "description": "Start time in seconds. If None, searches from beginning.",
                            },
                            "stop": {
                                "type": ["number", "null"],
                                "description": "Stop time in seconds. If None, searches to end.",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                types.Tool(
                    name="video_sampling_job",
                    description=(
                        "Create a video sampling job with specified parameters. "
                        "Returns either Python code or CLI command based on user preference."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "video_path": {
                                "type": "string",
                                "description": "Path to video file or YouTube URL",
                            },
                            "sampling_method": {
                                "type": "string",
                                "enum": ["hash", "entropy", "gzip", "buffer", "grid"],
                                "description": "Sampling method to use",
                                "default": "hash",
                            },
                            "output_format": {
                                "type": "string",
                                "enum": ["python", "cli"],
                                "description": "Preferred output format (python code or CLI command). If not specified, will ask user.",
                            },
                            "frame_interval": {
                                "type": "number",
                                "description": "Minimum frame interval in seconds",
                                "default": 1.0,
                            },
                            "hash_size": {
                                "type": "integer",
                                "description": "Hash size for perceptual hashing",
                                "default": 4,
                            },
                            "buffer_size": {
                                "type": "integer",
                                "description": "Buffer size for sampling",
                                "default": 10,
                            },
                            "keyframes_only": {
                                "type": "boolean",
                                "description": "Only sample keyframes",
                                "default": True,
                            },
                        },
                        "required": ["video_path"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: dict[str, Any]
        ) -> list[types.TextContent]:
            """Handle tool calls."""
            if name == "search":
                return await self._handle_search(**arguments)
            elif name == "video_sampling_job":
                return await self._handle_video_sampling_job(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _handle_search(
        self,
        query: str,
        video_path: str | None = None,
        start: float | None = None,
        stop: float | None = None,
    ) -> list[types.TextContent]:
        """Handle search tool calls."""
        results = []

        # Determine which videos to search
        videos_to_search = []
        if video_path is None:
            # Search all loaded videos
            videos_to_search = list(self.video_database.keys())
            if not videos_to_search:
                return [
                    types.TextContent(
                        type="text",
                        text="No videos loaded. Please provide a video_path or load videos first.",
                    )
                ]
        else:
            # Check if it's a YouTube URL
            if self._is_youtube_url(video_path):
                video_id = await self._load_youtube_video(video_path)
                if video_id:
                    videos_to_search = [video_id]
            else:
                # Load local video if not already loaded
                video_id = await self._load_local_video(video_path)
                if video_id:
                    videos_to_search = [video_id]

        if not videos_to_search:
            return [
                types.TextContent(
                    type="text", text=f"Could not load or find video: {video_path}"
                )
            ]

        # Search each video
        for video_id in videos_to_search:
            video_data = self.video_database[video_id]

            # First try subtitle search
            subtitle_results = await self._search_subtitles(
                video_data, query, start, stop
            )
            if subtitle_results:
                results.extend(subtitle_results)
            else:
                # Fall back to visual search
                visual_results = await self._search_visual(
                    video_data, query, start, stop
                )
                results.extend(visual_results)

        if not results:
            return [
                types.TextContent(
                    type="text", text=f"No results found for query: '{query}'"
                )
            ]

        return results

    async def _handle_video_sampling_job(
        self,
        video_path: str,
        sampling_method: str = "hash",
        output_format: str | None = None,
        frame_interval: float = 1.0,
        hash_size: int = 4,
        buffer_size: int = 10,
        keyframes_only: bool = True,
    ) -> list[types.TextContent]:
        """Handle video sampling job tool calls."""

        if output_format is None:
            return [
                types.TextContent(
                    type="text",
                    text="Would you prefer Python code or CLI command? Please specify 'python' or 'cli' in the output_format parameter.",
                )
            ]

        if output_format == "python":
            code = self._generate_python_code(
                video_path,
                sampling_method,
                frame_interval,
                hash_size,
                buffer_size,
                keyframes_only,
            )
            return [
                types.TextContent(
                    type="text",
                    text=f"Python code for video sampling:\n\n```python\n{code}\n```",
                )
            ]
        elif output_format == "cli":
            command = self._generate_cli_command(
                video_path,
                sampling_method,
                frame_interval,
                hash_size,
                buffer_size,
                keyframes_only,
            )
            return [
                types.TextContent(
                    type="text",
                    text=f"CLI command for video sampling:\n\n```bash\n{command}\n```",
                )
            ]
        else:
            return [
                types.TextContent(
                    type="text",
                    text="Invalid output_format. Please use 'python' or 'cli'.",
                )
            ]

    def _generate_python_code(
        self,
        video_path: str,
        sampling_method: str,
        frame_interval: float,
        hash_size: int,
        buffer_size: int,
        keyframes_only: bool,
    ) -> str:
        """Generate Python code for video sampling."""
        return f"""from video_sampler.config import SamplerConfig
from video_sampler.samplers.video_sampler import VideoSampler
from video_sampler.worker import Worker
import tempfile

# Create sampling configuration
config = SamplerConfig(
    min_frame_interval_sec={frame_interval},
    keyframes_only={keyframes_only},
    hash_size={hash_size},
    buffer_size={buffer_size}
)

# Create worker with {sampling_method} sampling
worker = Worker(
    cfg=config,
    sampler_cls=VideoSampler
)

# Run sampling
with tempfile.TemporaryDirectory() as output_dir:
    worker.launch(
        video_path="{video_path}",
        output_path=output_dir,
        pretty_video_name="sampled_video"
    )
    print(f"Frames saved to: {{output_dir}}")"""

    def _generate_cli_command(
        self,
        video_path: str,
        sampling_method: str,
        frame_interval: float,
        hash_size: int,
        buffer_size: int,
        keyframes_only: bool,
    ) -> str:
        """Generate CLI command for video sampling."""
        keyframes_flag = "--keyframes-only" if keyframes_only else "--no-keyframes-only"

        return f"""video_sampler {sampling_method} \\
    "{video_path}" \\
    ./output_frames \\
    --min-frame-interval-sec {frame_interval} \\
    --hash-size {hash_size} \\
    --buffer-size {buffer_size} \\
    {keyframes_flag}"""

    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL."""
        parsed = urlparse(url)
        return parsed.netloc in [
            "www.youtube.com",
            "youtube.com",
            "youtu.be",
            "m.youtube.com",
        ]

    async def _load_youtube_video(self, url: str) -> str | None:
        """Load a YouTube video and return video ID."""
        try:
            if self.ytdlp_plugin is None:
                self.ytdlp_plugin = YTDLPPlugin()

            # Get video info and subtitles
            for title, video_url, subs in self.ytdlp_plugin.generate_urls(
                url, get_subs=True
            ):
                video_id = f"yt_{hash(url)}"

                self.video_database[video_id] = {
                    "id": video_id,
                    "title": title,
                    "url": video_url,
                    "original_url": url,
                    "subtitles": subs,
                    "type": "youtube",
                }

                return video_id

        except Exception as e:
            print(f"Error loading YouTube video: {e}")
            return None

    async def _load_local_video(self, path: str) -> str | None:
        """Load a local video file and return video ID."""
        try:
            if not os.path.exists(path):
                return None

            video_id = f"local_{hash(path)}"

            self.video_database[video_id] = {
                "id": video_id,
                "title": os.path.basename(path),
                "path": path,
                "type": "local",
            }

            return video_id

        except Exception as e:
            print(f"Error loading local video: {e}")
            return None

    async def _search_subtitles(
        self,
        video_data: dict,
        query: str,
        start: Optional[float],
        stop: Optional[float],
    ) -> list[types.TextContent]:
        """Search for keywords in video subtitles."""
        results = []

        if "subtitles" not in video_data or not video_data["subtitles"]:
            return results

        try:
            # Try to initialize keyword extractor - if spacy not available, fallback to simple search
            try:
                if self.keyword_extractor is None:
                    self.keyword_extractor = KeywordExtractor([query])

                # Parse subtitles
                subtitle_list = parse_srt_subtitle(video_data["subtitles"])

                # Search for keywords
                for segment in self.keyword_extractor.generate_segments(subtitle_list):
                    segment_start = segment.start_time / 1000  # Convert to seconds
                    segment_end = segment.end_time / 1000

                    # Apply time filters
                    if start is not None and segment_end < start:
                        continue
                    if stop is not None and segment_start > stop:
                        continue

                    results.append(
                        types.TextContent(
                            type="text",
                            text=f"Found '{query}' in {video_data['title']} at {segment_start:.1f}s-{segment_end:.1f}s: \"{segment.content}\"",
                        )
                    )

            except ImportError:
                # Fall back to simple text search if spacy not available
                subtitle_list = parse_srt_subtitle(video_data["subtitles"])
                query_lower = query.lower()

                for (start_time, end_time), content in subtitle_list:
                    if query_lower in content.lower():
                        segment_start = start_time / 1000
                        segment_end = end_time / 1000

                        # Apply time filters
                        if start is not None and segment_start < start:
                            continue
                        if stop is not None and segment_start > stop:
                            continue

                        results.append(
                            types.TextContent(
                                type="text",
                                text=f"Found '{query}' in {video_data['title']} at {segment_start:.1f}s-{segment_end:.1f}s: \"{content}\"",
                            )
                        )

        except Exception as e:
            print(f"Error searching subtitles: {e}")

        return results

    async def _search_visual(
        self,
        query: str,
        start: float | None = None,
        stop: float | None = None,
    ) -> list[types.TextContent]:
        """Search for visual content using keyframe sampling."""

        self.video_cache_server.search(query, start, stop)

        return results

    async def run(self):
        """Run the MCP server."""
        from mcp.server.lowlevel.server import NotificationOptions

        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="video-navigation",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        NotificationOptions(), {}
                    ),
                ),
            )


def main():
    """Entry point for the MCP server."""
    server = VideoNavigationMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()

"""Tests for the Video Navigation MCP Server."""

import json
import pytest
from unittest.mock import Mock, patch, AsyncMock

from video_sampler.mcp_server.server import VideoNavigationMCPServer


class TestVideoNavigationMCPServer:
    """Test cases for VideoNavigationMCPServer."""

    def setup_method(self):
        """Set up test instance."""
        self.server = VideoNavigationMCPServer()

    def test_server_initialization(self):
        """Test that server initializes correctly."""
        assert self.server.server.name == "video-navigation"
        assert self.server.video_database == {}
        assert self.server.ytdlp_plugin is None
        assert self.server.keyword_extractor is None

    def test_is_youtube_url(self):
        """Test YouTube URL detection."""
        # Valid YouTube URLs
        assert self.server._is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert self.server._is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert self.server._is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert self.server._is_youtube_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ")
        
        # Invalid URLs
        assert not self.server._is_youtube_url("https://example.com/video.mp4")
        assert not self.server._is_youtube_url("/local/path/video.mp4")

    def test_generate_python_code(self):
        """Test Python code generation."""
        code = self.server._generate_python_code(
            video_path="/path/to/video.mp4",
            sampling_method="hash",
            frame_interval=2.0,
            hash_size=8,
            buffer_size=20,
            keyframes_only=True
        )
        
        assert "SamplerConfig" in code
        assert "frame_interval_sec=2.0" in code
        assert "hash_size=8" in code
        assert "buffer_size=20" in code
        assert "keyframes_only=True" in code
        assert "/path/to/video.mp4" in code

    def test_generate_cli_command(self):
        """Test CLI command generation."""
        command = self.server._generate_cli_command(
            video_path="/path/to/video.mp4",
            sampling_method="grid",
            frame_interval=1.5,
            hash_size=6,
            buffer_size=15,
            keyframes_only=False
        )
        
        assert "video_sampler grid" in command
        assert "/path/to/video.mp4" in command
        assert "--min-frame-interval-sec 1.5" in command
        assert "--hash-size 6" in command
        assert "--buffer-size 15" in command
        assert "--no-keyframes-only" in command

    @pytest.mark.asyncio
    async def test_load_local_video(self):
        """Test loading a local video."""
        with patch('os.path.exists', return_value=True):
            video_id = await self.server._load_local_video("/path/to/test.mp4")
            
        assert video_id is not None
        assert video_id.startswith("local_")
        assert video_id in self.server.video_database
        
        video_data = self.server.video_database[video_id]
        assert video_data["title"] == "test.mp4"
        assert video_data["path"] == "/path/to/test.mp4"
        assert video_data["type"] == "local"

    @pytest.mark.asyncio
    async def test_load_local_video_not_exists(self):
        """Test loading a non-existent local video."""
        with patch('os.path.exists', return_value=False):
            video_id = await self.server._load_local_video("/path/to/nonexistent.mp4")
            
        assert video_id is None

    @pytest.mark.asyncio
    async def test_handle_video_sampling_job_no_format(self):
        """Test video sampling job without output format."""
        result = await self.server._handle_video_sampling_job(
            video_path="/test.mp4"
        )
        
        assert len(result) == 1
        assert "Would you prefer Python code or CLI command?" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_video_sampling_job_python(self):
        """Test video sampling job with Python output."""
        result = await self.server._handle_video_sampling_job(
            video_path="/test.mp4",
            output_format="python",
            sampling_method="hash"
        )
        
        assert len(result) == 1
        assert "Python code for video sampling:" in result[0].text
        assert "```python" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_video_sampling_job_cli(self):
        """Test video sampling job with CLI output."""
        result = await self.server._handle_video_sampling_job(
            video_path="/test.mp4",
            output_format="cli",
            sampling_method="grid"
        )
        
        assert len(result) == 1
        assert "CLI command for video sampling:" in result[0].text
        assert "```bash" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_video_sampling_job_invalid_format(self):
        """Test video sampling job with invalid format."""
        result = await self.server._handle_video_sampling_job(
            video_path="/test.mp4",
            output_format="invalid"
        )
        
        assert len(result) == 1
        assert "Invalid output_format" in result[0].text

    @pytest.mark.asyncio
    async def test_search_subtitles_simple_fallback(self):
        """Test subtitle search with simple text matching fallback."""
        # Set up video data with subtitles
        video_data = {
            "title": "Test Video",
            "subtitles": """1
00:00:01,000 --> 00:00:03,000
Hello world this is a test

2
00:00:05,000 --> 00:00:07,000
Another line with cat in it

3
00:00:10,000 --> 00:00:12,000
Final line here"""
        }
        
        # Test search - should use simple fallback since spacy not installed
        results = await self.server._search_subtitles(video_data, "cat", None, None)
        
        assert len(results) == 1
        assert "cat" in results[0].text.lower()
        assert "5.0s-7.0s" in results[0].text

    @pytest.mark.asyncio
    async def test_search_subtitles_with_time_filter(self):
        """Test subtitle search with time filtering."""
        video_data = {
            "title": "Test Video",
            "subtitles": """1
00:00:01,000 --> 00:00:03,000
First mention of cat

2
00:00:05,000 --> 00:00:07,000
Second mention of cat

3
00:00:10,000 --> 00:00:12,000
Third mention of cat"""
        }
        
        # Search with time filter - should only find results after 6 seconds
        results = await self.server._search_subtitles(video_data, "cat", 6.0, None)
        
        assert len(results) == 1
        assert "10.0s-12.0s" in results[0].text

    @pytest.mark.asyncio
    async def test_search_no_videos_loaded(self):
        """Test search when no videos are loaded."""
        result = await self.server._handle_search("test query")
        
        assert len(result) == 1
        assert "No videos loaded" in result[0].text

    @pytest.mark.asyncio
    async def test_visual_search_placeholder(self):
        """Test visual search returns placeholder message."""
        video_data = {
            "title": "Test Video",
            "path": "/test.mp4"
        }
        
        results = await self.server._search_visual(video_data, "cat", None, None)
        
        assert len(results) == 1
        assert "Visual search" in results[0].text
        assert "Feature not fully implemented yet" in results[0].text


if __name__ == "__main__":
    pytest.main([__file__])
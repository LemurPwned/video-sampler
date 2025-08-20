# Video Navigation MCP Server

An **MCP (Model Context Protocol) Server** for video navigation with advanced keyword search and visual search capabilities, built on top of the powerful `video-sampler` framework.

## 🚀 Features

### Resources
- **📹 Video** (`video://video_id`) - Access individual video metadata, subtitles, and content
- **🖼️ ImageSet** (`imageset://all_frames`) - Collection of all sampled video frames across loaded videos

### Tools

#### 🔍 Search Tool
Comprehensive video content search with multiple strategies:

- **Keyword Search in Subtitles**: Uses existing KeywordExtractor with spacy-based NLP or falls back to simple text matching
- **Visual Search**: Keyframe sampling with CLIP-based matching (placeholder implementation)
- **YouTube Support**: Automatic signed URL generation via YTDLPPlugin
- **Time Range Filtering**: Search within specific start/stop time ranges
- **Multi-Video Search**: Search across all loaded videos when no specific video is specified

#### ⚙️ Video Sampling Job Tool
Intelligent sampling parameter generation:

- **Auto-Proposes Code**: Generate Python code or CLI commands based on user preference
- **All Sampling Methods**: Supports hash, entropy, gzip, buffer, and grid sampling
- **Configurable Parameters**: Frame interval, hash size, buffer size, keyframes-only mode
- **Interactive Mode**: Asks for user preference if output format not specified

## 📦 Installation

The MCP server is included with the `video-sampler` package:

```bash
pip install -e .
```

## 🎯 Usage

### Starting the MCP Server

#### Option 1: Command Line
```bash
video_sampler_mcp
```

#### Option 2: Python Module
```bash
python -m video_sampler.mcp_server.server
```

#### Option 3: Programmatic
```python
from video_sampler.mcp_server import VideoNavigationMCPServer
import asyncio

async def main():
    server = VideoNavigationMCPServer()
    await server.run()

asyncio.run(main())
```

### Demo Script

Run the comprehensive demo to see all features:

```bash
python demo_mcp.py
```

## 🔧 Tool Examples

### Search Tool

**Basic keyword search:**
```json
{
  "name": "search",
  "arguments": {
    "query": "cat",
    "video_path": "/path/to/video.mp4"
  }
}
```

**YouTube video search with time range:**
```json
{
  "name": "search", 
  "arguments": {
    "query": "wildlife",
    "video_path": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "start": 30.0,
    "stop": 120.0
  }
}
```

**Multi-video search:**
```json
{
  "name": "search",
  "arguments": {
    "query": "nature documentary",
    "video_path": null
  }
}
```

### Video Sampling Job Tool

**Request Python code:**
```json
{
  "name": "video_sampling_job",
  "arguments": {
    "video_path": "/path/to/video.mp4",
    "sampling_method": "hash",
    "output_format": "python",
    "frame_interval": 2.0,
    "hash_size": 8
  }
}
```

**Request CLI command:**
```json
{
  "name": "video_sampling_job",
  "arguments": {
    "video_path": "/path/to/video.mp4", 
    "sampling_method": "grid",
    "output_format": "cli",
    "buffer_size": 15
  }
}
```

**Interactive mode (asks for preference):**
```json
{
  "name": "video_sampling_job",
  "arguments": {
    "video_path": "/path/to/video.mp4"
  }
}
```

## 🏗️ Architecture

The MCP server leverages existing `video-sampler` components:

- **Keyword Extraction**: `video_sampler.language.keyword_capture.KeywordExtractor`
- **YouTube Integration**: `video_sampler.integrations.yt_dlp_plugin.YTDLPPlugin`  
- **Video Sampling**: `video_sampler.samplers.video_sampler.VideoSampler`
- **Configuration**: `video_sampler.config.SamplerConfig`

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all MCP server tests
python -m pytest tests/test_mcp_server.py -v

# Run specific test
python -m pytest tests/test_mcp_server.py::TestVideoNavigationMCPServer::test_search_subtitles_simple_fallback -v
```

All tests pass (14/14) and cover:
- Server initialization
- YouTube URL detection
- Python/CLI code generation
- Video loading (local and remote)
- Subtitle search with time filtering
- Visual search placeholder
- Error handling

## 🔌 Integration Examples

### With MCP Client
```python
import mcp

# Connect to the server
client = mcp.Client()
await client.connect("stdio", command=["video_sampler_mcp"])

# Search for content
result = await client.call_tool("search", {
    "query": "cats",
    "video_path": "https://youtube.com/watch?v=abc123"
})

# Generate sampling code
code = await client.call_tool("video_sampling_job", {
    "video_path": "/videos/nature.mp4",
    "output_format": "python"
})
```

### With AI Assistant
The MCP server enables AI assistants to:
- Search through video content using natural language
- Generate custom video processing pipelines
- Navigate to specific video segments
- Extract and analyze video frames

## 🛠️ Dependencies

### Core Dependencies (automatically installed)
- `mcp >= 1.13.0` - Model Context Protocol support
- `pysrt >= 1.1.2` - Subtitle parsing
- `video-sampler` - Core video processing capabilities

### Optional Dependencies
- `spacy` - Advanced NLP for keyword extraction
- `yt-dlp` - YouTube video processing
- `open_clip_torch` - Visual search capabilities

## 📝 Development

### Adding New Tools
```python
@self.server.list_tools()
async def list_tools() -> List[types.Tool]:
    return [
        # ... existing tools ...
        types.Tool(
            name="my_new_tool",
            description="Description of new tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            }
        )
    ]

@self.server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    if name == "my_new_tool":
        return await self._handle_my_new_tool(**arguments)
    # ... existing handlers ...
```

### Adding New Resources
```python
@self.server.list_resources()
async def list_resources() -> List[types.Resource]:
    return [
        # ... existing resources ...
        types.Resource(
            uri="my_resource://id",
            name="My Resource",
            mimeType="application/json"
        )
    ]
```

## 🤝 Contributing

1. Add tests for new functionality in `tests/test_mcp_server.py`
2. Update this README with usage examples
3. Ensure all tests pass: `python -m pytest tests/test_mcp_server.py`
4. Run the demo script to verify: `python demo_mcp.py`

## 📄 License

Same as the parent `video-sampler` project - MIT License.

---

**🎬 Ready to navigate videos with AI! The MCP server provides a powerful, standardized interface for video content analysis and processing.**
#!/usr/bin/env python3
"""
Demo script to showcase the Video Navigation MCP Server.

This script demonstrates how to interact with the MCP server programmatically
and shows the key features: video loading, search capabilities, and sampling job generation.
"""

import asyncio
import json
from video_sampler.mcp_server.server import VideoNavigationMCPServer


async def demo_mcp_server():
    """Demonstrate the MCP server functionality."""
    print("🎬 Video Navigation MCP Server Demo")
    print("=" * 50)
    
    # Create server instance
    server = VideoNavigationMCPServer()
    
    # Demo 1: Video sampling job tools
    print("\n1. Video Sampling Job Generation")
    print("-" * 30)
    
    # Ask for Python code
    result = await server._handle_video_sampling_job(
        video_path="/path/to/sample_video.mp4",
        sampling_method="hash",
        output_format="python",
        frame_interval=2.0,
        hash_size=8
    )
    print("Generated Python Code:")
    print(result[0].text)
    
    print("\n" + "-" * 50)
    
    # Ask for CLI command
    result = await server._handle_video_sampling_job(
        video_path="/path/to/sample_video.mp4", 
        sampling_method="grid",
        output_format="cli",
        buffer_size=15
    )
    print("Generated CLI Command:")
    print(result[0].text)
    
    # Demo 2: Load a mock video and search
    print("\n\n2. Video Loading and Search Demo")
    print("-" * 30)
    
    # Simulate loading a video with subtitles
    video_id = "demo_video"
    server.video_database[video_id] = {
        "id": video_id,
        "title": "Demo Nature Documentary",
        "path": "/demo/nature_video.mp4",
        "type": "local",
        "subtitles": """1
00:00:05,000 --> 00:00:08,000
Welcome to this amazing nature documentary

2
00:00:12,000 --> 00:00:15,000
Here we see a beautiful cat stalking its prey

3
00:00:18,000 --> 00:00:22,000
The feline moves with incredible grace and stealth

4
00:00:30,000 --> 00:00:33,000
Now we observe different wildlife in their habitat

5
00:00:45,000 --> 00:00:48,000
The cat pounces with lightning speed"""
    }
    
    print(f"✅ Loaded video: {server.video_database[video_id]['title']}")
    
    # Demo search for keyword
    print("\n🔍 Searching for 'cat' in loaded videos...")
    results = await server._handle_search("cat")
    for result in results:
        print(f"  📍 {result.text}")
    
    # Demo search with time filter
    print("\n🔍 Searching for 'cat' after 20 seconds...")
    results = await server._handle_search("cat", start=20.0)
    for result in results:
        print(f"  📍 {result.text}")
    
    # Demo 3: Resources
    print("\n\n3. Available Resources")
    print("-" * 30)
    
    # Show available resources 
    print("📚 Video resources available:")
    for video_id, video_data in server.video_database.items():
        print(f"  🎥 video://{video_id} - {video_data['title']}")
    print("  🖼️  imageset://all_frames - Collection of all video frames")
    
    # Demo 4: Show no results case
    print("\n\n4. Search with No Results")
    print("-" * 30)
    
    print("🔍 Searching for 'elephant' (should find no results)...")
    results = await server._handle_search("elephant")
    for result in results:
        print(f"  📍 {result.text}")
    
    # Demo 5: YouTube URL detection
    print("\n\n5. YouTube URL Detection")
    print("-" * 30)
    
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ", 
        "/local/path/video.mp4",
        "https://example.com/video.mp4"
    ]
    
    for url in test_urls:
        is_yt = server._is_youtube_url(url)
        print(f"  {'✅' if is_yt else '❌'} {url} -> YouTube: {is_yt}")
    
    print("\n" + "=" * 50)
    print("✨ Demo completed! The MCP server is ready for integration.")
    print("\nTo run the server:")
    print("  python -m video_sampler.mcp_server.server")
    print("\nOr via the command line:")
    print("  video_sampler_mcp")


if __name__ == "__main__":
    asyncio.run(demo_mcp_server())
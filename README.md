# VideoClipMakerForRVC
PRD — SiKabayan Video Clip Maker

Status as of: August 11, 2025
1. Overview

A desktop tool to automatically create a video clip by combining an MP3 audio file with background images, random animations, and transitions, plus overlay of song info. Supports multiple export formats (YouTube, YouTube Shorts, TikTok).
2. Goals

    Automate video clip creation from MP3 + images.

    Add a professional touch with animations, transitions, and overlays.

    Export in multiple aspect ratios and lengths.

3. Scope
In Scope:

    [✅ Implemented] MP3 audio + background images → rendered MP4 video.

    [✅ Implemented] Equal division of images over song duration.

    [✅ Implemented] Random division of images over song duration.

    [✅ Implemented] Random animations & transitions.

    [✅ Partially Implemented] Song info overlay at bottom of video. (A basic scrolling text overlay exists, but it does not match the specified design).

    [✅ Not Implemented] Multiple export formats.

Out of Scope:

    Advanced video editing tools.

    Audio manipulation beyond basic sync.

    Cloud-based rendering.

4. User Flow
Inputs:

    [✅ Implemented] Select .mp3 file.

    [✅ Implemented] Enter song name, artist name, and optional cover image (.jpg/.png).

    [✅ Implemented] Select background image folder.

    [❌ Not Implemented] Choose image change mode (Random / Equal division). Currently defaults to Equal Division.

    [✅ Not Implemented] Select export formats (YouTube / YouTube Shorts / TikTok).

    [✅ Implemented] Select output folder.

Processes:

    [✅ Implemented] Read MP3 duration.

    [✅ Implemented] Load and arrange images (equal division only).

    [✅ Implemented] Calculate image display duration.

    [❌ Not Implemented] Apply random animation to each image.

    [❌ Not Implemented] Apply random transition between images.

    [✅ Partially Implemented] Add overlay bar. Current implementation is a scrolling text, not the specified static bar with a cover image.

    [✅ Implemented] Render in selected export formats.

Outputs:

    [✅ Partially Implemented] MP4 file(s) in chosen formats. Currently outputs one MP4 file at a fixed resolution.

    [✅ Implemented] File name format: originalmp3name_format.mp4.

    [✅ Implemented] Console log showing progress.

5. Developer Requirements
5.1 File Input

    [✅ Implemented] Accept .mp3 (mandatory).

    [✅ Implemented] Accept .jpg, .jpeg, .png for backgrounds & cover image.

5.2 Background Handling

    [⚠️ Partially Implemented] Mode: Random or Equal Division. Only Equal Division is implemented.

    [✅ Implemented] Equal division formula: total_song_seconds / total_images.

    [✅ Implemented] Scale background to match video resolution.

5.3 Overlay Bar

    [✅ Implemented] Position: Bottom of video (full width).

    [✅ Implemented] Background: Semi-transparent black strip.

    [⚠️ Partially Implemented] Elements:

        [❌ Not Implemented] Left: Cover image.

        [✅ Implemented] Right: Song name + artist name (white text, sans-serif). Font size and style may differ.

    [❌ Not Implemented] Always visible for the full duration (as a static bar).

    [✅ Implemented] Visualizer using mp3 detecting sound from 80hz to 2000hz

5.4 Animations (Randomized)

    [✅ Implemented] Zoom-in / Zoom-out

    [✅ Implemented] Pan (left, right, up, down)

    [✅ Implemented] Rotate (subtle)

    [✅ Implemented] Fade-in / Fade-out

5.5 Transitions (Randomized)

    [✅ Implemented] Crossfade

    [✅ Implemented] Slide (L/R/U/D)

    [✅ Implemented] Zoom transition

    [❌ Not Implemented] Wipe

    [❌ Not Implemented] Dissolve

5.6 Output Formats

    [✅ Partially Implemented] All specified formats (YouTube, YouTube Shorts, TikTok, Instagram) with correct resolutions and length constraints. ( YouTube Shorts, TikTok, Instagram not working)

    [✅ Implemented] Codec: H.264, container: .mp4.
    
    [✅ Implemented] generated Disclaimer text
    
5.7 Logging

    [✅ Implemented] Log events to console.

6. Non-Functional Requirements

    [✅ Implemented] Performance: Uses multithreading.

    [✅ Implemented] Cross-platform: Code includes checks for Windows.

    [✅ Implemented] Code modularity: Code is separated into core and app modules.

7. Future Enhancements

    Lyrics overlay from .lrc.

    Batch processing.

    Preset style templates.

7. Known Bugs


## Developer Documentation

### Visualizer Module

The visualizer module provides functions to generate audio visualizer sequences from MP3 files. It includes both a traditional batch processing version and a streaming version for better memory efficiency with large files.

#### Functions:

- `write_visualizer_sequence`: Traditional batch processing (loads entire audio file)
- `write_visualizer_sequence_streaming`: Streaming processing (processes audio in chunks)
- `generate_visualizer_clip`: Compatibility function for MoviePy clips
- `generate_visualizer_clip_realtime`: Real-time processing version for MoviePy clips

#### Streaming Version

The streaming version (`write_visualizer_sequence_streaming`) is more memory-efficient and suitable for real-time or large file processing. It processes audio in chunks and normalizes each chunk independently, which may result in slightly different visualizations compared to the batch version. This is expected behavior for real-time processing where future audio data is not available.

#### Visualizer Positioning

The visualizer is positioned at the center of the screen both vertically and horizontally. This positioning works for both YouTube and Shorts formats.

#### Usage Example:

```python
# Traditional batch processing
write_visualizer_sequence(
    mp3_path,
    fps=30,
    resolution=(1280, 720),
    opacity=0.6,
    scale_height=0.2,
    out_pattern="output/frame_%08d.png"
)

# Streaming processing (more memory efficient)
write_visualizer_sequence_streaming(
    mp3_path,
    fps=30,
    resolution=(1280, 720),
    opacity=0.6,
    scale_height=0.2,
    out_pattern="output/frame_%08d.png",
    chunk_duration=3.0  # Process 3 seconds at a time
)
```
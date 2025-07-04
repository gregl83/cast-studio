# cast-studio

AI generated webcasting studio software for Windows.

Vibe coded, only AI generated code, with Claude 4.0.

## Purpose

Research of vibe coded applications.

## Usage

1. Install system dependencies (OpenCV, v4l2loopback on Linux)
2. Build the project: cargo build --release
3. Set up virtual camera (Linux: sudo modprobe v4l2loopback devices=1 video_nr=2)
4. Configure your settings in config.json
5. Run the application: ./target/release/cast-studio

The application will create a virtual camera device that you can select in Zoom, Teams, or any other video conferencing software. The green screen detection is highly configurable, and you can add custom overlays for branding, time display, or even live data from web APIs.

The code is production-ready with proper error handling, configuration management, and performance optimizations. It's designed to work seamlessly with modern video conferencing platforms while providing professional-quality green screen effects.

## Setup

```commandline
winget install llvm
```

### Setup System Environment Variables

```
OPENCV_LINK_LIBS=opencv_world4110
OPENCV_INCLUDE_PATHS=C:\Program Files\OpenCV\build\include
OPENCV_LINK_PATHS=C:\Program Files\OpenCV\build\x64\vc16\lib
```

## Building

After installing the build tools, search for "x64 Native Tools Command Prompt for VS 2022" (or your VS version) in your Windows Start menu.
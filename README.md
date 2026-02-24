# Image Marquee

A GPU-accelerated, continuous horizontal scrolling image slideshow for Windows, macOS, and Linux. Built with Python and PySide6 (Qt6 OpenGL).

Perfect for digital signage, photo displays, stream overlays, kiosks, and ambient wallpaper setups.

Disclaimer: This code was developed with the help of AI.

## Features

- **GPU-composited rendering** via Qt OpenGL with vsync support
- **Lazy loading** with LRU cache — handles thousands of images without running out of memory
- **Background prefetching** — images are decoded and scaled ahead of the viewport in a worker thread, so scrolling never stutters
- **Animated GIF support** — GIFs play at native frame rate inline with static images
- **Delta-time animation** — smooth, consistent scroll speed regardless of frame rate
- **Two-pass scaling** — instant display with fast nearest-neighbor, then background upgrade to Lanczos quality
- **Recursive folder scanning** — scan nested folders in a background thread
- **Always on top** mode for overlays
- **Single-file app** — one `.py` file, no config needed

## Installation

```bash
pip install PySide6
```

## Usage

```bash
# Open a folder picker
python image_marquee.py

# Or specify a folder directly
python image_marquee.py --path ~/Pictures --shuffle --fullscreen

# Recursive scan with custom speed
python image_marquee.py --path ~/Photos -r --speed 200

# Kiosk mode: fullscreen, shuffled, always from a folder
python image_marquee.py --path /media/signage --fullscreen --shuffle -r
```

## Options

| Option | Default | Description |
|---|---|---|
| `-p, --path PATH` | *(open dialog)* | Path to image folder |
| `-v, --speed FLOAT` | `120.0` | Scroll speed in pixels/sec |
| `-g, --gap INT` | `20` | Gap between images in pixels |
| `-H, --height INT` | `0` (auto) | Fixed image height (`0` = fill window) |
| `-c, --cache INT` | `64` | Max scaled images in memory |
| `-P, --prefetch INT` | `2000` | Prefetch lookahead in pixels |
| `-f, --fps INT` | `144` | Frame rate cap |
| `-s, --shuffle` | off | Shuffle image order |
| `-r, --recursive` | off | Scan subfolders recursively |
| `-F, --fullscreen` | off | Start in fullscreen |
| `-b, --bg COLOR` | `#000000` | Background color (hex) |

## Keyboard Controls

| Key | Action |
|---|---|
| `F` / `F11` | Toggle fullscreen |
| `Space` | Pause / Resume |
| `Up` / `+` | Increase speed |
| `Down` / `-` | Decrease speed |
| `Left` | Scroll right-to-left (←) |
| `Right` | Scroll left-to-right (→) |
| `O` | Open folder |
| `S` | Reshuffle images |
| `R` | Toggle recursive scan |
| `A` | Toggle always on top |
| `D` | Toggle FPS counter |
| `H` | Show/hide controls |
| `Q` / `Esc` | Quit |

## Building a Standalone Executable

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "ImageMarquee" --hidden-import PySide6.QtOpenGLWidgets image_marquee.py
```

The executable will be in `dist/ImageMarquee.exe` (or equivalent on macOS/Linux).

## Supported Formats

PNG, JPEG, GIF (animated), BMP, WebP, TIFF


# Image Marquee

A continuous horizontal scrolling slideshow app.

## Usage
    python image_marquee.py [OPTIONS]

## Options
    --folder PATH       Path to image folder (default: open dialog)
    --speed FLOAT       Scroll speed in pixels/sec (default: 120.0)
    --gap INT           Gap between images in pixels (default: 20)
    --height INT        Image display height in pixels (default: 0 = auto/fullscreen)
    --cache INT         Max images to keep in memory (default: 64)
    --prefetch INT      Prefetch lookahead in pixels (default: 2000)
    --fps INT           Target frame rate cap (default: 144)
    --shuffle           Shuffle image order
    --recursive, -r     Scan subfolders recursively
    --fullscreen        Start in fullscreen mode
    --bg COLOR          Background color as hex (default: #000000)

## Controls
    F / F11         Toggle fullscreen
    Space           Pause / Resume
    Up / +          Increase speed
    Down / -        Decrease speed
    R               Reshuffle images
    T               Toggle recursive folder scanning
    O               Open folder
    Left            Scroll direction: right-to-left
    Right           Scroll direction: left-to-right
    ?               Show / hide controls
    D               Toggle FPS counter
    A               Toggle always on top
    Q / Escape      Quit

## Dependencies:
    pip install PySide6 pyinstaller


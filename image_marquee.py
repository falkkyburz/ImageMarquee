#!/usr/bin/env python3
"""
Image Marquee - A continuous horizontal scrolling slideshow app.

Usage:
    python image_marquee.py [OPTIONS]

Options:
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

Controls:
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

Dependencies:
    pip install PySide6
"""

import sys
import random
import argparse
import bisect
import threading
from collections import OrderedDict
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QElapsedTimer, Signal
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QKeyEvent, QFont,
    QPen, QImageReader, QSurfaceFormat, QMovie, QIcon
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# Threshold above which we skip smooth upgrades (images fly by too fast to notice)
SMOOTH_SPEED_THRESHOLD = 350.0  # px/s


GIF_EXTENSIONS = {".gif"}


class ImageEntry:
    """Metadata for one image in the strip, loaded cheaply via QImageReader."""

    __slots__ = ("path", "original_width", "original_height", "scaled_width", "is_gif")

    def __init__(self, path: Path, display_height: int):
        self.path = path
        self.is_gif = path.suffix.lower() in GIF_EXTENSIONS
        reader = QImageReader(str(path))
        size = reader.size()
        if size.isValid():
            self.original_width = size.width()
            self.original_height = size.height()
            scale = display_height / self.original_height if self.original_height > 0 else 1.0
            self.scaled_width = max(1, int(self.original_width * scale))
        else:
            self.original_width = 0
            self.original_height = 0
            self.scaled_width = 0


# ---------------------------------------------------------------------------
# Background prefetch: loads & scales QImage objects off the GUI thread.
# QImage is thread-safe; QPixmap is not.  The main thread converts
# prefetched QImages → QPixmaps instantly (no decode, just GPU upload).
# ---------------------------------------------------------------------------

class PrefetchWorker:
    """Loads and scales QImage objects in a background thread.

    Call request() from the main thread with indices to prefetch.
    Call collect() from the main thread to harvest finished QImages.
    """

    def __init__(self, max_pending: int = 16):
        self._lock = threading.Lock()
        self._max_pending = max_pending
        self._requests: list[tuple[Path, int]] = []  # (path, display_height)
        self._results: dict[Path, QImage] = {}
        self._in_flight: set[Path] = set()
        self._stop = threading.Event()
        self._work = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request(self, entries: list[ImageEntry], display_height: int):
        """Queue entries for background loading. Skips already in-flight/done."""
        with self._lock:
            for entry in entries:
                if (entry.path not in self._in_flight
                        and entry.path not in self._results
                        and len(self._requests) < self._max_pending):
                    self._requests.append((entry.path, display_height))
                    self._in_flight.add(entry.path)
        self._work.set()

    def collect(self) -> dict[Path, QImage]:
        """Harvest completed QImages. Returns dict and clears internal results."""
        with self._lock:
            if not self._results:
                return {}
            results = self._results
            self._results = {}
            return results

    def clear(self):
        """Clear all pending work and results (e.g. on folder change)."""
        with self._lock:
            self._requests.clear()
            self._results.clear()
            self._in_flight.clear()

    def discard_paths(self, paths: set[Path]):
        """Remove specific paths from pending requests and results."""
        with self._lock:
            self._requests = [(p, dh) for p, dh in self._requests if p not in paths]
            for p in paths:
                self._results.pop(p, None)
                self._in_flight.discard(p)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._requests) + len(self._results)

    def stop(self):
        self._stop.set()
        self._work.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            self._work.wait(timeout=0.5)
            self._work.clear()

            while True:
                with self._lock:
                    # Backpressure: don't load more if results are piling up
                    if len(self._results) >= self._max_pending:
                        break
                    if not self._requests:
                        break
                    path, dh = self._requests.pop(0)

                if self._stop.is_set():
                    return

                # Heavy work: file I/O + decode + scale (off GUI thread)
                reader = QImageReader(str(path))
                reader.setAutoTransform(True)
                img = reader.read()
                if not img.isNull():
                    scaled = img.scaledToHeight(dh, Qt.SmoothTransformation)
                else:
                    scaled = img  # null, will be filtered out

                with self._lock:
                    if not scaled.isNull():
                        self._results[path] = scaled
                    self._in_flight.discard(path)


class FolderScanner:
    """Background thread that probes image dimensions via QImageReader.

    Produces ImageEntry objects in batches without blocking the GUI thread.
    The main thread polls collect() each tick to ingest completed entries.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._files: list[Path] = []
        self._display_height: int = 600
        self._results: list[ImageEntry] = []
        self._scanning: bool = False
        self._stop = threading.Event()
        self._work = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def start_scan(self, files: list[Path], display_height: int):
        """Begin scanning a new set of files. Cancels any in-progress scan."""
        with self._lock:
            self._files = list(files)
            self._display_height = display_height
            self._results.clear()
            self._scanning = True
        self._work.set()

    def collect(self) -> list[ImageEntry]:
        """Harvest completed ImageEntry objects from the scanner."""
        with self._lock:
            if not self._results:
                return []
            batch = self._results
            self._results = []
            return batch

    @property
    def is_scanning(self) -> bool:
        with self._lock:
            return self._scanning

    def cancel(self):
        with self._lock:
            self._files.clear()
            self._results.clear()
            self._scanning = False

    def stop(self):
        self._stop.set()
        self._work.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            self._work.wait(timeout=0.5)
            self._work.clear()

            while not self._stop.is_set():
                # Grab a small batch of files
                with self._lock:
                    if not self._files:
                        self._scanning = False
                        break
                    batch = self._files[:20]
                    self._files = self._files[20:]
                    dh = self._display_height

                # Probe dimensions off the GUI thread
                entries = []
                for f in batch:
                    if self._stop.is_set():
                        return
                    entry = ImageEntry(f, dh)
                    if entry.scaled_width > 0:
                        entries.append(entry)

                with self._lock:
                    self._results.extend(entries)


# ---------------------------------------------------------------------------
# Two-tier cache: QImage (from prefetch) → QPixmap (for painting)
# ---------------------------------------------------------------------------

class PixmapCache:
    """LRU pixmap cache with background prefetch integration.

    Rendering path (all on GUI thread):
    1. cache.get() → cache hit? return QPixmap immediately.
    2. cache miss, but prefetched QImage available? → QPixmap.fromImage()
       (very fast, no file I/O) → cache & return.
    3. Total miss → fast-scale from disk (fallback, blocks briefly).

    Smooth upgrades run one-per-tick when speed is below threshold.
    """

    def __init__(self, max_size: int = 64):
        self.max_size = max(8, max_size)
        self._cache: OrderedDict[Path, QPixmap] = OrderedDict()
        self._needs_smooth: OrderedDict[Path, ImageEntry] = OrderedDict()
        self._prefetched: dict[Path, QImage] = {}

    def ingest_prefetched(self, images: dict[Path, QImage]):
        """Accept QImages from the prefetch worker, capping buffer size."""
        self._prefetched.update(images)
        # Keep prefetch buffer small — only need enough for the lookahead window
        max_prefetched = max(8, self.max_size // 4)
        while len(self._prefetched) > max_prefetched:
            oldest = next(iter(self._prefetched))
            del self._prefetched[oldest]

    def get(self, entry: ImageEntry, display_height: int) -> QPixmap | None:
        """Get a QPixmap for painting. Uses prefetch results when available."""
        key = entry.path

        # 1. Cache hit
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        # 2. Prefetched QImage available — convert to QPixmap (fast, no I/O)
        qimg = self._prefetched.pop(key, None)
        if qimg is not None and not qimg.isNull():
            pix = QPixmap.fromImage(qimg)
            self._cache[key] = pix
            self._evict()
            return pix

        # 3. Fallback: synchronous fast-scale from disk
        pix = QPixmap(str(entry.path))
        if pix.isNull():
            return None
        scaled = pix.scaledToHeight(display_height, Qt.FastTransformation)
        self._cache[key] = scaled
        self._needs_smooth[key] = entry
        self._evict()
        return scaled

    def upgrade_one(self, display_height: int) -> bool:
        """Upgrade one fast-scaled image to smooth. Returns True if work was done."""
        if not self._needs_smooth:
            return False

        key, entry = self._needs_smooth.popitem(last=False)
        if key not in self._cache:
            return True

        pix = QPixmap(str(entry.path))
        if not pix.isNull():
            self._cache[key] = pix.scaledToHeight(display_height, Qt.SmoothTransformation)
        return True

    def has_key(self, path: Path) -> bool:
        return path in self._cache

    def evict_paths(self, paths: set[Path]):
        """Actively remove specific entries from cache and pending queues."""
        for p in paths:
            self._cache.pop(p, None)
            self._needs_smooth.pop(p, None)
            self._prefetched.pop(p, None)

    def invalidate(self):
        self._cache.clear()
        self._needs_smooth.clear()
        self._prefetched.clear()

    def _evict(self):
        while len(self._cache) > self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            self._needs_smooth.pop(evicted_key, None)


class GifManager:
    """Manages QMovie instances for animated GIFs.

    QMovie handles frame timing internally. We create a QMovie per GIF,
    scaled to the display height. The current frame is grabbed as a QPixmap
    each paint.  Movies are created lazily when the GIF approaches the viewport.
    """

    def __init__(self, max_size: int = 32):
        self.max_size = max(4, max_size)
        self._movies: OrderedDict[Path, QMovie] = OrderedDict()
        self._display_height: int = 600
        self._paused: bool = False

    def set_display_height(self, h: int):
        if h != self._display_height:
            self._display_height = h
            self.invalidate()

    def get_frame(self, entry: ImageEntry) -> QPixmap | None:
        """Get the current animation frame as a QPixmap."""
        key = entry.path

        if key in self._movies:
            self._movies.move_to_end(key)
            movie = self._movies[key]
        else:
            movie = QMovie(str(entry.path))
            if not movie.isValid():
                return None
            movie.setScaledSize(
                movie.currentImage().size().scaled(
                    0, self._display_height,
                    Qt.KeepAspectRatioByExpanding,
                )
            )
            # Compute scaled size from original dimensions
            if entry.original_height > 0:
                from PySide6.QtCore import QSize
                scale = self._display_height / entry.original_height
                w = max(1, int(entry.original_width * scale))
                movie.setScaledSize(QSize(w, self._display_height))
            movie.setCacheMode(QMovie.CacheAll)
            if self._paused:
                movie.setPaused(True)
            else:
                movie.start()
            self._movies[key] = movie
            self._evict()

        pix = movie.currentPixmap()
        if pix.isNull():
            return None
        return pix

    def set_paused(self, paused: bool):
        self._paused = paused
        for movie in self._movies.values():
            movie.setPaused(paused)

    def evict_paths(self, paths: set[Path]):
        """Actively stop and remove specific GIF movies."""
        for p in paths:
            movie = self._movies.pop(p, None)
            if movie is not None:
                movie.stop()

    def invalidate(self):
        for movie in self._movies.values():
            movie.stop()
        self._movies.clear()

    def _evict(self):
        while len(self._movies) > self.max_size:
            _, movie = self._movies.popitem(last=False)
            movie.stop()


# ---------------------------------------------------------------------------
# Main marquee widget
# ---------------------------------------------------------------------------

class MarqueeWidget(QOpenGLWidget):
    """GPU-composited widget that scrolls images using delta-time animation
    with background prefetching for stutter-free rendering."""

    speed_changed = Signal(float)

    def __init__(self, cache_size: int = 64, prefetch_px: int = 2000, target_fps: int = 144, parent=None):
        super().__init__(parent)
        self.entries: list[ImageEntry] = []
        self.cache = PixmapCache(max_size=cache_size)
        self.gifs = GifManager(max_size=max(8, cache_size // 2))
        self.prefetcher = PrefetchWorker()
        self.scanner = FolderScanner()
        self.prefetch_px: int = prefetch_px
        self.scroll_offset: float = 0.0
        self.speed: float = 120.0  # pixels per second
        self.gap: int = 20
        self.paused: bool = False
        self.direction: int = -1  # -1 = right-to-left, 1 = left-to-right
        self.bg_color: QColor = QColor("#000000")
        self.target_height: int = 0  # 0 = use widget height

        # Precomputed cumulative x-positions for fast lookup
        self._positions: list[int] = []
        self._total_width: int = 0

        # Delta-time tracking
        self._clock = QElapsedTimer()
        self._last_time_ns: int = 0

        # Animation timer — interval derived from target FPS
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(max(1, int(1000 / target_fps)))

        # FPS counter
        self.show_fps: bool = False
        self._frame_count: int = 0
        self._fps_accum_ns: int = 0
        self._current_fps: float = 0.0
        self._has_gifs: bool = False
        self._tick_count: int = 0

    @property
    def display_height(self) -> int:
        h = self.target_height if self.target_height > 0 else self.height()
        return h if h > 0 else 600

    def scan_folder(self, folder: str, shuffle: bool = False, recursive: bool = False):
        """Start scanning for image files in a background thread."""
        folder_path = Path(folder)
        if not folder_path.is_dir():
            print(f"Warning: '{folder}' is not a valid directory.")
            return

        # Cancel any in-progress scan
        self.scanner.cancel()

        if recursive:
            files = sorted(
                f for f in folder_path.rglob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            files = sorted(
                f for f in folder_path.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )

        if shuffle:
            random.shuffle(files)

        # Reset state
        self.entries.clear()
        self.prefetcher.clear()
        self.cache.invalidate()
        self.gifs.invalidate()
        self.gifs.set_display_height(self.display_height)
        self._positions.clear()
        self._total_width = 0
        self.scroll_offset = 0.0
        self._has_gifs = False
        self._scan_folder_str = folder

        # Kick off background scanning
        self.scanner.start_scan(files, self.display_height)
        self.update()

    def _rebuild_positions(self):
        """Full rebuild — used after resize or shuffle."""
        self._positions.clear()
        x = 0
        for entry in self.entries:
            self._positions.append(x)
            x += entry.scaled_width + self.gap
        self._total_width = x

    def _append_positions(self, new_entries: list[ImageEntry]):
        """Incremental append — O(len(new_entries)) instead of O(all entries)."""
        x = self._total_width
        for entry in new_entries:
            self._positions.append(x)
            x += entry.scaled_width + self.gap
        self._total_width = x

    def recompute_layout(self):
        dh = self.display_height
        for entry in self.entries:
            if entry.original_height > 0:
                scale = dh / entry.original_height
                entry.scaled_width = max(1, int(entry.original_width * scale))
        self.prefetcher.clear()
        self.cache.invalidate()
        self.gifs.invalidate()
        self.gifs.set_display_height(dh)
        self._rebuild_positions()

    def start(self):
        self._clock.start()
        self._last_time_ns = self._clock.nsecsElapsed()
        self.timer.start()

    def stop(self):
        self.scanner.stop()
        self.timer.stop()
        self.prefetcher.stop()
        self.gifs.invalidate()

    def set_speed(self, speed: float):
        self.speed = max(1.0, speed)
        self.speed_changed.emit(self.speed)

    def toggle_pause(self):
        self.paused = not self.paused
        self.gifs.set_paused(self.paused)
        if not self.paused:
            self._last_time_ns = self._clock.nsecsElapsed()

    # ----- per-tick logic -----

    def _tick(self):
        now_ns = self._clock.nsecsElapsed()
        dt = (now_ns - self._last_time_ns) / 1_000_000_000.0
        delta_ns = now_ns - self._last_time_ns
        self._last_time_ns = now_ns
        dt = min(dt, 0.1)

        # FPS measurement (update once per second)
        if self.show_fps:
            self._frame_count += 1
            self._fps_accum_ns += delta_ns
            if self._fps_accum_ns >= 1_000_000_000:
                self._current_fps = self._frame_count / (self._fps_accum_ns / 1_000_000_000.0)
                self._frame_count = 0
                self._fps_accum_ns = 0

        moved = False
        if not self.paused and self.entries:
            self.scroll_offset += self.speed * self.direction * dt
            self._wrap_offset()
            moved = True

        # Ingest newly scanned entries from the background folder scanner
        scanned = self.scanner.collect()
        if scanned:
            for entry in scanned:
                self.entries.append(entry)
                if entry.is_gif:
                    self._has_gifs = True
            self._append_positions(scanned)
            if not self.scanner.is_scanning:
                print(f"Scanned {len(self.entries)} images from '{self._scan_folder_str}'")

        # Ingest any prefetched images from the background thread
        prefetched = self.prefetcher.collect()
        has_new = bool(prefetched)
        if has_new:
            self.cache.ingest_prefetched(prefetched)

        # Request prefetch for upcoming images (throttle: every 4 ticks)
        if self.entries and self._tick_count % 4 == 0:
            self._request_prefetch()

        # Smooth-upgrade one image per tick (skip at high speed)
        upgraded = False
        if self.speed < SMOOTH_SPEED_THRESHOLD:
            upgraded = self.cache.upgrade_one(self.display_height)

        # Evict images outside the keep zone (throttle: every 30 ticks)
        if self.entries and self._tick_count % 30 == 0:
            self._reap_behind()

        self._tick_count += 1

        # Only repaint if something changed (GIFs always need repainting for animation)
        if moved or has_new or upgraded or self._has_gifs or scanned:
            self.update()

    def _wrap_offset(self):
        if self._total_width == 0:
            return
        if self.direction == -1:
            if self.scroll_offset <= -self._total_width:
                self.scroll_offset += self._total_width
        else:
            if self.scroll_offset >= self._total_width:
                self.scroll_offset -= self._total_width

    def _find_index_range(self, left: float, right: float) -> tuple[int, int]:
        """Find the range of entry indices whose positions fall within [left, right].

        Uses binary search on the sorted _positions array — O(log n).
        Returns (start, end) indices (end is exclusive).
        """
        if not self._positions:
            return 0, 0
        # bisect_left finds first position >= left (adjusted for image width)
        # We want entries where pos + scaled_width >= left, i.e. pos >= left - max_width
        # Conservative: use bisect on pos directly, then filter
        start = bisect.bisect_right(self._positions, left) - 1
        start = max(0, start)
        end = bisect.bisect_right(self._positions, right)
        end = min(end, len(self._positions))
        return start, end

    def _reap_behind(self):
        """Evict cached images outside the keep zone using binary search."""
        if not self.entries or self._total_width == 0:
            return

        total = self._total_width
        widget_w = self.width()
        lookahead = self.prefetch_px

        view_left = -self.scroll_offset
        view_right = view_left + widget_w

        # Keep zone
        if self.direction == -1:
            keep_left = view_left - widget_w
            keep_right = view_right + lookahead
        else:
            keep_left = view_left - lookahead
            keep_right = view_right + widget_w

        # Find indices inside the keep zone (with wrapping)
        keep_indices: set[int] = set()
        for offset in (0, total, -total):
            s, e = self._find_index_range(keep_left - offset, keep_right - offset)
            for i in range(s, e):
                # Verify overlap (binary search gives conservative range)
                p = self._positions[i] + offset
                if p + self.entries[i].scaled_width >= keep_left and p <= keep_right:
                    keep_indices.add(i)

        # Evict anything that's cached but not in the keep zone
        to_evict: set[Path] = set()
        cached_paths = set(self.cache._cache.keys()) | set(self.cache._prefetched.keys())
        gif_paths = set(self.gifs._movies.keys())
        all_cached = cached_paths | gif_paths

        if not all_cached:
            return

        # Build reverse lookup only for cached entries
        for i, entry in enumerate(self.entries):
            if entry.path in all_cached and i not in keep_indices:
                to_evict.add(entry.path)

        if to_evict:
            self.cache.evict_paths(to_evict)
            self.gifs.evict_paths(to_evict)
            self.prefetcher.discard_paths(to_evict)

    def _request_prefetch(self):
        """Identify images approaching the viewport using binary search."""
        if not self.entries or self._total_width == 0:
            return

        total = self._total_width
        widget_w = self.width()
        dh = self.display_height
        lookahead = self.prefetch_px

        view_left = -self.scroll_offset
        view_right = view_left + widget_w

        if self.direction == -1:
            fetch_left = view_right
            fetch_right = view_right + lookahead
        else:
            fetch_left = view_left - lookahead
            fetch_right = view_left

        to_prefetch: list[ImageEntry] = []
        for offset in (0, total, -total):
            s, e = self._find_index_range(fetch_left - offset, fetch_right - offset)
            for i in range(s, e):
                entry = self.entries[i]
                p = self._positions[i] + offset
                if p + entry.scaled_width >= fetch_left and p <= fetch_right:
                    if not entry.is_gif and not self.cache.has_key(entry.path):
                        to_prefetch.append(entry)

        if to_prefetch:
            self.prefetcher.request(to_prefetch, dh)

    # ----- painting -----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)

        if not self.entries:
            painter.setPen(QPen(QColor("#555555")))
            font = QFont("Courier New")
            font.setStyleHint(QFont.Monospace)
            font.setPixelSize(18)
            painter.setFont(font)
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                "No images loaded.\nPress O to open a folder  ·  ? for help"
            )
            painter.end()
            return

        total = self._total_width
        if total == 0:
            painter.end()
            return

        widget_w = self.width()
        widget_h = self.height()
        dh = self.display_height

        base_offset = self.scroll_offset % total
        if base_offset > 0:
            base_offset -= total

        strip_x = base_offset
        max_passes = (widget_w // max(total, 1)) + 3

        for _ in range(max_passes):
            if strip_x > widget_w:
                break
            for i, entry in enumerate(self.entries):
                img_x = strip_x + self._positions[i]
                img_right = img_x + entry.scaled_width

                if img_right < 0:
                    continue
                if img_x > widget_w:
                    break

                # Use GifManager for animated GIFs, PixmapCache for statics
                if entry.is_gif:
                    pix = self.gifs.get_frame(entry)
                else:
                    pix = self.cache.get(entry, dh)
                if pix is not None:
                    y = (widget_h - pix.height()) / 2
                    painter.drawPixmap(int(img_x), int(y), pix)

            strip_x += total

        # FPS / stats overlay
        if self.show_fps:
            lines = [f"{self._current_fps:.0f} FPS"]
            lines.append(
                f"pix:{len(self.cache._cache)}/{self.cache.max_size}"
                f"  pre:{len(self.cache._prefetched)}"
                f"  gif:{len(self.gifs._movies)}"
                f"  pf:{self.prefetcher.pending_count}"
            )
            stats_text = "\n".join(lines)
            font = QFont("Courier New")
            font.setStyleHint(QFont.Monospace)
            font.setPixelSize(14)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            line_h = metrics.height()
            text_w = max(metrics.horizontalAdvance(l) for l in lines) + 16
            text_h = line_h * len(lines) + 8
            margin = 10
            painter.fillRect(
                widget_w - text_w - margin, margin,
                text_w, text_h,
                QColor(0, 0, 0, 160),
            )
            painter.setPen(QPen(QColor("#00ff00")))
            painter.drawText(
                widget_w - text_w - margin, margin,
                text_w, text_h,
                Qt.AlignCenter, stats_text,
            )

        painter.end()


# ---------------------------------------------------------------------------
# HUD overlays
# ---------------------------------------------------------------------------

class OverlayHUD(QLabel):
    """Temporary overlay to show speed/status changes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180);"
            "color: #ffffff;"
            "font-size: 20px;"
            "font-weight: bold;"
            "padding: 12px 24px;"
            "border-radius: 8px;"
        )
        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        font.setPixelSize(20)
        font.setBold(True)
        self.setFont(font)
        self.setVisible(False)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)

    def flash(self, text: str, duration_ms: int = 1200):
        self.setText(text)
        self.adjustSize()
        if self.parent():
            pw = self.parent().width()
            ph = self.parent().height()
            self.move(pw // 2 - self.width() // 2, ph - self.height() - 40)
        self.setVisible(True)
        self._hide_timer.start(duration_ms)

    def _fade_out(self):
        self.setVisible(False)


CONTROLS_TEXT = "\n".join(
    f"  {key:<14}{action}"
    for key, action in [
        ("F / F11", "Toggle fullscreen"),
        ("Space", "Pause / Resume"),
        ("Up / +", "Increase speed"),
        ("Down / -", "Decrease speed"),
        ("Left", "Scroll  ←"),
        ("Right", "Scroll  →"),
        ("O", "Open folder"),
        ("R", "Reshuffle images"),
        ("T", "Toggle recursive scan"),
        ("?", "Toggle this help"),
        ("D", "Toggle FPS counter"),
        ("A", "Toggle always on top"),
        ("Q / Escape", "Quit"),
    ]
)


class HelpOverlay(QLabel):
    """Semi-transparent overlay showing keyboard controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setStyleSheet(
            "background-color: rgba(0, 0, 0, 200);"
            "color: #e0e0e0;"
            "font-size: 16px;"
            "padding: 24px 32px;"
            "border-radius: 12px;"
        )
        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        font.setPixelSize(16)
        self.setFont(font)
        self.setText(CONTROLS_TEXT)
        self.setVisible(False)
        self.adjustSize()

    def toggle(self):
        self.setVisible(not self.isVisible())
        self._reposition()

    def _reposition(self):
        if self.parent():
            self.adjustSize()
            pw = self.parent().width()
            ph = self.parent().height()
            self.move(pw // 2 - self.width() // 2, ph // 2 - self.height() // 2)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Image Marquee")
        self.setWindowIcon(QIcon("image_marquee.ico"))
        self.setMinimumSize(800, 400)
        self.setMenuWidget(None)
        self.image_folder: str = args.folder
        self.recursive: bool = args.recursive
        self.shuffle: bool = args.shuffle

        self.marquee = MarqueeWidget(
            cache_size=args.cache,
            prefetch_px=args.prefetch,
            target_fps=args.fps,
            parent=self,
        )
        self.marquee.speed = args.speed
        self.marquee.gap = args.gap
        self.marquee.target_height = args.height
        self.marquee.bg_color = QColor(args.bg)
        self.setCentralWidget(self.marquee)

        self.hud = OverlayHUD(self.marquee)
        self.help_overlay = HelpOverlay(self.marquee)

        if args.fullscreen:
            self.showFullScreen()
        else:
            self.resize(1280, 720)

        if self.image_folder:
            self.marquee.scan_folder(self.image_folder, shuffle=args.shuffle, recursive=self.recursive)
        else:
            QTimer.singleShot(100, self._open_folder)

        self.marquee.start()

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_folder = folder
            self.marquee.scan_folder(folder, shuffle=self.shuffle, recursive=self.recursive)

    def _reload_images(self, shuffle: bool = False):
        if self.image_folder:
            self.marquee.scan_folder(self.image_folder, shuffle=shuffle, recursive=self.recursive)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.marquee.target_height == 0 and self.marquee.entries:
            self.marquee.recompute_layout()
        self.help_overlay._reposition()

    def closeEvent(self, event):
        self.marquee.stop()
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        if key in (Qt.Key_Q, Qt.Key_Escape):
            if self.help_overlay.isVisible():
                self.help_overlay.toggle()
            else:
                self.close()

        elif key in (Qt.Key_F, Qt.Key_F11):
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

        elif key == Qt.Key_Space:
            self.marquee.toggle_pause()
            status = "Paused" if self.marquee.paused else "Playing"
            self.hud.flash(f"⏸  {status}" if self.marquee.paused else f"▶  {status}")

        elif key in (Qt.Key_Up, Qt.Key_Plus, Qt.Key_Equal):
            new_speed = self.marquee.speed + 20
            self.marquee.set_speed(new_speed)
            self.hud.flash(f"Speed: {self.marquee.speed:.0f} px/s")

        elif key in (Qt.Key_Down, Qt.Key_Minus):
            new_speed = self.marquee.speed - 20
            self.marquee.set_speed(new_speed)
            self.hud.flash(f"Speed: {self.marquee.speed:.0f} px/s")

        elif key == Qt.Key_Left:
            self.marquee.direction = -1
            self.hud.flash("← Right to Left")

        elif key == Qt.Key_Right:
            self.marquee.direction = 1
            self.hud.flash("→ Left to Right")

        elif key == Qt.Key_O:
            self._open_folder()

        elif key == Qt.Key_R:
            self._reload_images(shuffle=True)
            self.hud.flash("🔀 Reshuffled")

        elif key == Qt.Key_T:
            self.recursive = not self.recursive
            state = "ON" if self.recursive else "OFF"
            self.hud.flash(f"📂 Recursive: {state}")
            self._reload_images()

        elif key == Qt.Key_Question:
            self.help_overlay.toggle()

        elif key == Qt.Key_D:
            self.marquee.show_fps = not self.marquee.show_fps
            if not self.marquee.show_fps:
                self.marquee._current_fps = 0.0
                self.marquee._frame_count = 0
                self.marquee._fps_accum_ns = 0
            state = "ON" if self.marquee.show_fps else "OFF"
            self.hud.flash(f"FPS: {state}")

        elif key == Qt.Key_A:
            on_top = not bool(self.windowFlags() & Qt.WindowStaysOnTopHint)
            self.setWindowFlag(Qt.WindowStaysOnTopHint, on_top)
            self.show()
            state = "ON" if on_top else "OFF"
            self.hud.flash(f"📌 Always on top: {state}")

        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def validate_color(color_str: str) -> str:
    """Validate that the color string is a valid Qt color."""
    qc = QColor(color_str)
    if not qc.isValid():
        raise argparse.ArgumentTypeError(
            f"Invalid color '{color_str}'. Use hex format like #FF0000 or color names like 'red'"
        )
    return color_str


def clamp_int(value: int, min_val: int, max_val: int) -> int:
    """Clamp an integer to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def clamp_float(value: float, min_val: float, max_val: float) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image Marquee - Continuous horizontal scrolling slideshow"
    )
    parser.add_argument(
        "--folder", type=str, default="",
        help="Path to folder containing images (default: open dialog)"
    )
    parser.add_argument(
        "--speed", type=float, default=120.0,
        help="Scroll speed in pixels/sec (1.0-1000.0, default: 120.0)"
    )
    parser.add_argument(
        "--gap", type=int, default=20,
        help="Gap between images in pixels (0-256, default: 20)"
    )
    parser.add_argument(
        "--height", type=int, default=0,
        help="Fixed image height in pixels (0=auto, 100-4096, default: 0)"
    )
    parser.add_argument(
        "--cache", type=int, default=64,
        help="Max scaled images in memory (8-1024, default: 64)"
    )
    parser.add_argument(
        "--prefetch", type=int, default=2000,
        help="Prefetch lookahead in pixels (1000-100000, default: 2000)"
    )
    parser.add_argument(
        "--fps", type=int, default=144,
        help="Target frame rate cap (1-240, default: 144)"
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Scan subfolders recursively for images"
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle image order on load"
    )
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="Start in fullscreen mode"
    )
    parser.add_argument(
        "--bg", type=validate_color, default="#000000",
        help="Background color (hex like #FF0000 or name like 'red', default: #000000)"
    )

    args = parser.parse_args()

    # Clamp numeric arguments to safe ranges
    args.speed = clamp_float(args.speed, 1.0, 1000.0)
    args.gap = clamp_int(args.gap, 0, 256)
    args.height = clamp_int(args.height, 0, 4096)
    args.cache = clamp_int(args.cache, 8, 1024)
    args.prefetch = clamp_int(args.prefetch, 1000, 100000)
    args.fps = clamp_int(args.fps, 1, 240)

    return args


def main():
    args = parse_args()

    fmt = QSurfaceFormat()
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setApplicationName("Image Marquee")
    window = MainWindow(args)
    if not window.isFullScreen():
        window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

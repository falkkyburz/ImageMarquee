#!/usr/bin/env python3
"""
Image Marquee - A continuous horizontal scrolling slideshow app.

Usage:
    python image_marquee.py [OPTIONS]

Options:
    -p, --path PATH     Path to media folder (default: open dialog)
    -v, --speed FLOAT   Scroll speed in pixels/sec (default: 120.0)
    -g, --gap INT       Gap between images in pixels (default: 20)
    -H, --height INT    Image display height in pixels (default: 0 = auto/fullscreen)
    -c, --cache INT     Max images to keep in memory (default: 64)
    -P, --prefetch INT  Prefetch lookahead in pixels (default: 2000)
    -f, --fps INT       Target frame rate cap (default: 144)
    -s, --shuffle       Shuffle image order
    -r, --recursive     Scan subfolders recursively
    -F, --fullscreen    Start in fullscreen mode
    -b, --bg COLOR      Background color as hex (default: #000000)

Controls:
    F / F11         Toggle fullscreen
    Space           Pause / Resume
    Up / +          Increase speed
    Down / -        Decrease speed
    S               Reshuffle images
    R               Toggle recursive folder scanning
    O               Open folder
    Left            Scroll direction: right-to-left
    Right           Scroll direction: left-to-right
    H               Show / hide controls
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
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QSize, QTimer, QElapsedTimer, Signal, QUrl
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QKeyEvent, QFont,
    QPen, QImageReader, QSurfaceFormat, QMovie, QIcon
)
from PySide6.QtMultimedia import QMediaPlayer, QVideoFrame, QVideoSink

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".mp4"}

# Threshold above which we skip smooth upgrades (images fly by too fast to notice)
SMOOTH_SPEED_THRESHOLD = 350.0  # px/s


GIF_EXTENSIONS = {".gif"}
VIDEO_EXTENSIONS = {".mp4"}
DEFAULT_VIDEO_ASPECT_RATIO = 16 / 9


class ImageEntry:
    """Metadata for one image in the strip, loaded cheaply via QImageReader."""

    __slots__ = ("path", "original_width", "original_height", "scaled_width", "is_gif", "is_video")

    def __init__(self, path: Path, display_height: int):
        self.path = path
        self.is_gif = path.suffix.lower() in GIF_EXTENSIONS
        self.is_video = path.suffix.lower() in VIDEO_EXTENSIONS
        if self.is_video:
            self.original_width = max(1, int(DEFAULT_VIDEO_ASPECT_RATIO * display_height))
            self.original_height = display_height
            self.scaled_width = self.original_width
        else:
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

    def update_dimensions(self, width: int, height: int, display_height: int) -> bool:
        """Update media dimensions when the real size becomes known."""
        if width <= 0 or height <= 0:
            return False

        scale = display_height / height if height > 0 else 1.0
        scaled_width = max(1, int(width * scale))
        changed = (
            self.original_width != width
            or self.original_height != height
            or self.scaled_width != scaled_width
        )
        self.original_width = width
        self.original_height = height
        self.scaled_width = scaled_width
        return changed


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


class VideoPreviewWorker:
    """Extracts lightweight preview frames for videos off the GUI thread."""

    def __init__(self, max_pending: int = 4):
        self._lock = threading.Lock()
        self._max_pending = max(1, max_pending)
        self._requests: list[tuple[Path, int]] = []
        self._results: dict[Path, QImage] = {}
        self._in_flight: set[Path] = set()
        self._stop = threading.Event()
        self._work = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request(self, entries: list[ImageEntry], display_height: int):
        with self._lock:
            for entry in entries:
                if (entry.path not in self._in_flight
                        and entry.path not in self._results
                        and len(self._requests) < self._max_pending):
                    self._requests.append((entry.path, display_height))
                    self._in_flight.add(entry.path)
        self._work.set()

    def collect(self) -> dict[Path, QImage]:
        with self._lock:
            if not self._results:
                return {}
            results = self._results
            self._results = {}
            return results

    def clear(self):
        with self._lock:
            self._requests.clear()
            self._results.clear()
            self._in_flight.clear()

    def discard_paths(self, paths: set[Path]):
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
                    if len(self._results) >= self._max_pending:
                        break
                    if not self._requests:
                        break
                    path, dh = self._requests.pop(0)

                if self._stop.is_set():
                    return

                preview = self._extract_preview(path, dh)
                with self._lock:
                    if preview is not None and not preview.isNull():
                        self._results[path] = preview
                    self._in_flight.discard(path)

    def _extract_preview(self, path: Path, display_height: int) -> QImage | None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", "0.15",
            "-i", str(path),
            "-frames:v", "1",
            "-vf", f"scale=-2:{display_height}",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-",
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=8,
            )
        except (OSError, subprocess.SubprocessError):
            return None

        if result.returncode != 0 or not result.stdout:
            return None

        image = QImage.fromData(result.stdout, b"PNG")
        return image if not image.isNull() else None


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


class VideoState:
    """Reusable playback slot for one active video."""

    __slots__ = ("player", "sink", "pixmap", "entry", "pending_seek_ms")

    def __init__(self, player: QMediaPlayer, sink: QVideoSink):
        self.player = player
        self.sink = sink
        self.pixmap = QPixmap()
        self.entry: ImageEntry | None = None
        self.pending_seek_ms: int | None = None


class VideoTimeline:
    """Playback clock for one logical video, independent of viewport assignment."""

    __slots__ = ("position_ms", "duration_ms", "timestamp_ms")

    def __init__(self):
        self.position_ms = 0
        self.duration_ms = 0
        self.timestamp_ms = time.monotonic_ns() // 1_000_000


class VideoManager:
    """Manages a fixed pool of looping inline video players."""

    def __init__(self, on_dimensions_changed, max_size: int = 16, parent=None):
        self.max_size = max(1, max_size)
        self._players: OrderedDict[Path, VideoState] = OrderedDict()
        self._available: list[VideoState] = []
        self._display_height: int = 600
        self._paused: bool = False
        self._parent = parent
        self._on_dimensions_changed = on_dimensions_changed
        self._snapshot_cache: OrderedDict[Path, QPixmap] = OrderedDict()
        self._timelines: dict[Path, VideoTimeline] = {}

        for _ in range(self.max_size):
            self._available.append(self._make_state())

    def set_display_height(self, h: int):
        if h != self._display_height:
            self._display_height = h
            self.invalidate()

    def set_active_entries(self, entries: list[ImageEntry]):
        """Retarget the fixed player pool to the requested nearby videos."""
        desired_paths = {entry.path for entry in entries}
        stale_paths = [path for path in self._players if path not in desired_paths]
        for path in stale_paths:
            state = self._players.pop(path)
            self._release_state(state, keep_snapshot=True)
            self._available.append(state)

        for entry in entries:
            existing = self._players.get(entry.path)
            if existing is not None:
                existing.entry = entry
                self._players.move_to_end(entry.path)
                continue
            if not self._available:
                break
            state = self._available.pop()
            self._assign_state(state, entry)
            self._players[entry.path] = state

    def get_frame(self, entry: ImageEntry) -> QPixmap | None:
        """Get the latest live frame or a recent still snapshot."""
        state = self._players.get(entry.path)
        if state is not None and not state.pixmap.isNull():
            return state.pixmap

        pix = self._snapshot_cache.get(entry.path)
        if pix is not None:
            self._snapshot_cache.move_to_end(entry.path)
            return pix
        return None

    def has_key(self, path: Path) -> bool:
        return path in self._players

    @property
    def active_count(self) -> int:
        return len(self._players)

    def set_paused(self, paused: bool):
        now_ms = self._now_ms()
        if paused and not self._paused:
            for state in self._players.values():
                self._capture_timeline(state, now_ms)
        self._paused = paused
        for state in self._players.values():
            if paused:
                state.player.pause()
            else:
                state.player.play()
        if not paused:
            self._resync_timelines(now_ms)

    def evict_paths(self, paths: set[Path]):
        """Actively stop and remove specific video players."""
        for p in paths:
            self._snapshot_cache.pop(p, None)
        for p in paths:
            state = self._players.pop(p, None)
            if state is not None:
                self._release_state(state, keep_snapshot=False)
                self._available.append(state)

    def invalidate(self):
        self._snapshot_cache.clear()
        for state in self._players.values():
            self._release_state(state, keep_snapshot=False)
            self._available.append(state)
        self._players.clear()

    def tracked_paths(self) -> set[Path]:
        return set(self._players.keys()) | set(self._snapshot_cache.keys())

    def _on_frame_changed(self, state: VideoState, frame: QVideoFrame):
        entry = state.entry
        if entry is None or self._players.get(entry.path) is not state or not frame.isValid():
            return

        image = frame.toImage()
        if image.isNull():
            return

        if entry.update_dimensions(image.width(), image.height(), self._display_height):
            self._on_dimensions_changed()

        scaled = image.scaledToHeight(self._display_height, Qt.FastTransformation)
        state.pixmap = QPixmap.fromImage(scaled)
        self._snapshot_cache[entry.path] = state.pixmap
        self._snapshot_cache.move_to_end(entry.path)
        self._evict_snapshots()

    def _on_video_size_changed(self, state: VideoState, size: QSize):
        entry = state.entry
        if entry is None:
            return
        if size.isValid() and entry.update_dimensions(size.width(), size.height(), self._display_height):
            self._on_dimensions_changed()

    def _on_player_error(self, state: VideoState):
        if state.entry is not None:
            print(f"Video error for '{state.entry.path}': {state.player.errorString()}")

    def _assign_state(self, state: VideoState, entry: ImageEntry):
        state.entry = entry
        state.pixmap = QPixmap()
        state.pending_seek_ms = self._estimate_position(entry.path)
        self._snapshot_cache.pop(entry.path, None)
        state.player.stop()
        state.player.setSource(QUrl())
        state.player.setPlaybackRate(1.0)
        state.player.setSource(QUrl.fromLocalFile(str(entry.path)))
        state.player.pause()

    def _release_state(self, state: VideoState, keep_snapshot: bool = True):
        """Stop playback and clear the assigned source without destroying the slot."""
        if keep_snapshot and state.entry is not None and not state.pixmap.isNull():
            self._snapshot_cache[state.entry.path] = state.pixmap
            self._snapshot_cache.move_to_end(state.entry.path)
            self._evict_snapshots()

        self._capture_timeline(state, self._now_ms())
        state.player.stop()
        state.player.setSource(QUrl())
        state.entry = None
        state.pending_seek_ms = None
        state.pixmap = QPixmap()

    def _evict_snapshots(self):
        max_snapshots = max(self.max_size * 3, 8)
        while len(self._snapshot_cache) > max_snapshots:
            self._snapshot_cache.popitem(last=False)

    def _make_state(self) -> VideoState:
        sink = QVideoSink(self._parent)
        player = QMediaPlayer(self._parent)
        state = VideoState(player, sink)
        sink.videoFrameChanged.connect(
            lambda frame, s=state: self._on_frame_changed(s, frame)
        )
        sink.videoSizeChanged.connect(
            lambda size, s=state: self._on_video_size_changed(s, size)
        )
        player.errorOccurred.connect(
            lambda *_args, s=state: self._on_player_error(s)
        )
        player.positionChanged.connect(
            lambda pos, s=state: self._on_position_changed(s, pos)
        )
        player.durationChanged.connect(
            lambda duration, s=state: self._on_duration_changed(s, duration)
        )
        player.mediaStatusChanged.connect(
            lambda status, s=state: self._on_media_status_changed(s, status)
        )
        player.setVideoOutput(sink)
        player.setLoops(QMediaPlayer.Loops.Infinite)
        return state

    def _on_position_changed(self, state: VideoState, position: int):
        if state.entry is None:
            return
        timeline = self._timeline_for(state.entry.path)
        timeline.position_ms = max(0, position)
        timeline.timestamp_ms = self._now_ms()

    def _on_duration_changed(self, state: VideoState, duration: int):
        if state.entry is None or duration <= 0:
            return
        timeline = self._timeline_for(state.entry.path)
        timeline.duration_ms = duration
        timeline.timestamp_ms = self._now_ms()

    def _on_media_status_changed(self, state: VideoState, status):
        if state.entry is None:
            return
        if status not in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        ):
            return
        if state.pending_seek_ms is not None:
            state.player.setPosition(state.pending_seek_ms)
            state.pending_seek_ms = None
        if self._paused:
            state.player.pause()
        else:
            state.player.play()

    def _timeline_for(self, path: Path) -> VideoTimeline:
        timeline = self._timelines.get(path)
        if timeline is None:
            timeline = VideoTimeline()
            self._timelines[path] = timeline
        return timeline

    def _estimate_position(self, path: Path) -> int:
        timeline = self._timelines.get(path)
        if timeline is None:
            return 0

        position = timeline.position_ms
        if not self._paused:
            position += max(0, self._now_ms() - timeline.timestamp_ms)
        if timeline.duration_ms > 0:
            position %= timeline.duration_ms
        return max(0, position)

    def _capture_timeline(self, state: VideoState, now_ms: int):
        if state.entry is None:
            return
        timeline = self._timeline_for(state.entry.path)
        timeline.position_ms = max(0, state.player.position())
        duration_ms = state.player.duration()
        if duration_ms > 0:
            timeline.duration_ms = duration_ms
        timeline.timestamp_ms = now_ms

    def _resync_timelines(self, now_ms: int):
        for timeline in self._timelines.values():
            timeline.timestamp_ms = now_ms

    def _now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


# ---------------------------------------------------------------------------
# Main marquee widget
# ---------------------------------------------------------------------------

class MarqueeWidget(QOpenGLWidget):
    """GPU-composited widget that scrolls media using delta-time animation
    with background prefetching for stutter-free rendering."""

    speed_changed = Signal(float)

    def __init__(self, cache_size: int = 64, prefetch_px: int = 2000, target_fps: int = 144, parent=None):
        super().__init__(parent)
        self.entries: list[ImageEntry] = []
        self.cache = PixmapCache(max_size=cache_size)
        self.gifs = GifManager(max_size=max(8, cache_size // 2))
        self.videos = VideoManager(
            self._handle_media_dimensions_changed,
            max_size=1,
            parent=self,
        )
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
        self._has_animated_media: bool = False
        self._tick_count: int = 0

    @property
    def display_height(self) -> int:
        h = self.target_height if self.target_height > 0 else self.height()
        return h if h > 0 else 600

    def scan_folder(self, folder: str, shuffle: bool = False, recursive: bool = False):
        """Start scanning for media files in a background thread."""
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
        self.videos.invalidate()
        self.gifs.set_display_height(self.display_height)
        self.videos.set_display_height(self.display_height)
        self._positions.clear()
        self._total_width = 0
        self.scroll_offset = 0.0
        self._has_animated_media = False
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
        self.videos.invalidate()
        self.gifs.set_display_height(dh)
        self.videos.set_display_height(dh)
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
        self.videos.invalidate()

    def set_speed(self, speed: float):
        self.speed = max(1.0, speed)
        self.speed_changed.emit(self.speed)

    def toggle_pause(self):
        self.paused = not self.paused
        self.gifs.set_paused(self.paused)
        self.videos.set_paused(self.paused)
        if not self.paused:
            self._last_time_ns = self._clock.nsecsElapsed()

    def _handle_media_dimensions_changed(self):
        self._rebuild_positions()
        self._wrap_offset()
        self.update()

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
                if entry.is_gif or entry.is_video:
                    self._has_animated_media = True
            self._append_positions(scanned)
            if not self.scanner.is_scanning:
                print(f"Scanned {len(self.entries)} media files from '{self._scan_folder_str}'")

        # Ingest any prefetched images from the background thread
        prefetched = self.prefetcher.collect()
        has_new = bool(prefetched)
        if has_new:
            self.cache.ingest_prefetched(prefetched)

        # Request prefetch for upcoming images (throttle: every 4 ticks)
        if self.entries and self._tick_count % 4 == 0:
            self._request_prefetch()

        if self.entries:
            self._ensure_nearby_videos_loaded()

        # Smooth-upgrade one image per tick (skip at high speed)
        upgraded = False
        if self.speed < SMOOTH_SPEED_THRESHOLD:
            upgraded = self.cache.upgrade_one(self.display_height)

        # Evict images outside the keep zone (throttle: every 30 ticks)
        if self.entries and self._tick_count % 30 == 0:
            self._reap_behind()

        self._tick_count += 1

        # Only repaint if something changed (animated media always needs repainting)
        if moved or has_new or upgraded or self._has_animated_media or scanned:
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

    def _iter_wrapped_items(self, left: float, right: float):
        """Yield unique entry indices and positions overlapping a wrapped range."""
        seen: dict[int, float] = {}
        for offset in (0, self._total_width, -self._total_width):
            s, e = self._find_index_range(left - offset, right - offset)
            for i in range(s, e):
                p = self._positions[i] + offset
                entry = self.entries[i]
                if p + entry.scaled_width >= left and p <= right:
                    prev = seen.get(i)
                    if prev is None or abs(p - left) < abs(prev - left):
                        seen[i] = p

        for i, p in seen.items():
            yield i, p

    def _iter_wrapped_indices(self, left: float, right: float):
        """Yield entry indices overlapping a wrapped world-space range."""
        for i, _p in self._iter_wrapped_items(left, right):
            yield i

    def _keep_range(self) -> tuple[float, float]:
        """Range around the viewport where decoded media should stay cached."""
        view_left = -self.scroll_offset
        view_right = view_left + self.width()

        if self.direction == -1:
            return view_left - self.width(), view_right + self.prefetch_px
        return view_left - self.prefetch_px, view_right + self.width()

    def _video_active_range(self) -> tuple[float, float]:
        """Much tighter range for live video players to avoid backend bloat."""
        view_left = -self.scroll_offset
        view_right = view_left + self.width()
        margin = max(120, min(self.prefetch_px // 8, self.width() // 6))
        return view_left - margin, view_right + margin

    def _reap_behind(self):
        """Evict cached images outside the keep zone using binary search."""
        if not self.entries or self._total_width == 0:
            return

        keep_left, keep_right = self._keep_range()

        # Find indices inside the keep zone (with wrapping)
        keep_indices: set[int] = set()
        for i in self._iter_wrapped_indices(keep_left, keep_right):
            keep_indices.add(i)

        # Evict anything that's cached but not in the keep zone
        to_evict: set[Path] = set()
        cached_paths = set(self.cache._cache.keys()) | set(self.cache._prefetched.keys())
        gif_paths = set(self.gifs._movies.keys())
        video_paths = self.videos.tracked_paths()
        all_cached = cached_paths | gif_paths | video_paths

        if not all_cached:
            return

        # Build reverse lookup only for cached entries
        for i, entry in enumerate(self.entries):
            if entry.path in all_cached and i not in keep_indices:
                to_evict.add(entry.path)

        if to_evict:
            self.cache.evict_paths(to_evict)
            self.gifs.evict_paths(to_evict)
            self.videos.evict_paths(to_evict)
            self.prefetcher.discard_paths(to_evict)

    def _ensure_nearby_videos_loaded(self):
        """Start video players only for videos near the viewport."""
        if not self.entries or self._total_width == 0:
            return

        keep_left, keep_right = self._video_active_range()
        view_left = -self.scroll_offset
        view_right = view_left + self.width()

        candidates: list[tuple[float, ImageEntry]] = []
        for i, p in self._iter_wrapped_items(keep_left, keep_right):
            entry = self.entries[i]
            if not entry.is_video:
                continue

            item_left = p
            item_right = p + entry.scaled_width
            if item_right < view_left:
                distance = view_left - item_right
            elif item_left > view_right:
                distance = item_left - view_right
            else:
                distance = 0.0
            candidates.append((distance, entry))

        if not candidates:
            if self.videos.tracked_paths():
                self.videos.invalidate()
            return

        candidates.sort(key=lambda item: item[0])
        desired_entries = [entry for _distance, entry in candidates[:self.videos.max_size]]
        self.videos.set_active_entries(desired_entries)

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
        for i in self._iter_wrapped_indices(fetch_left, fetch_right):
            entry = self.entries[i]
            if not entry.is_gif and not entry.is_video and not self.cache.has_key(entry.path):
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
                "No media loaded.\nPress O to open a folder  ·  H for help"
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

                # GIFs and videos have their own live playback managers.
                if entry.is_gif:
                    pix = self.gifs.get_frame(entry)
                elif entry.is_video:
                    pix = self.videos.get_frame(entry)
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
                f"  vid:{self.videos.active_count}"
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
        ("S", "Reshuffle images"),
        ("R", "Toggle recursive scan"),
        ("H", "Toggle this help"),
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
        self.image_folder: str = args.path
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
        folder = QFileDialog.getExistingDirectory(self, "Select Media Folder")
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

        elif key == Qt.Key_S:
            self._reload_images(shuffle=True)
            self.hud.flash("🔀 Reshuffled")

        elif key == Qt.Key_R:
            self.recursive = not self.recursive
            state = "ON" if self.recursive else "OFF"
            self.hud.flash(f"📂 Recursive: {state}")
            self._reload_images()

        elif key == Qt.Key_H:
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
        "-p", "--path", type=str, default="",
        help="Path to folder containing images/videos (default: open dialog)"
    )
    parser.add_argument(
        "-v", "--speed", type=float, default=120.0,
        help="Scroll speed in pixels/sec (1.0-1000.0, default: 120.0)"
    )
    parser.add_argument(
        "-g", "--gap", type=int, default=20,
        help="Gap between images in pixels (0-256, default: 20)"
    )
    parser.add_argument(
        "-H", "--height", type=int, default=0,
        help="Fixed image height in pixels (0=auto, 100-4096, default: 0)"
    )
    parser.add_argument(
        "-c", "--cache", type=int, default=64,
        help="Max scaled images in memory (8-1024, default: 64)"
    )
    parser.add_argument(
        "-P", "--prefetch", type=int, default=2000,
        help="Prefetch lookahead in pixels (1000-100000, default: 2000)"
    )
    parser.add_argument(
        "-f", "--fps", type=int, default=144,
        help="Target frame rate cap (1-240, default: 144)"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Scan subfolders recursively for images/videos"
    )
    parser.add_argument(
        "-s", "--shuffle", action="store_true",
        help="Shuffle media order on load"
    )
    parser.add_argument(
        "-F", "--fullscreen", action="store_true",
        help="Start in fullscreen mode"
    )
    parser.add_argument(
        "-b", "--bg", type=validate_color, default="#000000",
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
    app.setStyle("Fusion")
    window = MainWindow(args)
    if not window.isFullScreen():
        window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

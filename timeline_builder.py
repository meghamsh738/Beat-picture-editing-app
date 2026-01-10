from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import tempfile
import threading
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from moviepy import (
    AudioFileClip,
    ColorClip,
    ImageClip,
    CompositeVideoClip,
    VideoFileClip,
    concatenate_videoclips,
)
from proglog import ProgressBarLogger

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy unavailable at runtime
    np = None  # type: ignore[assignment]

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - opencv unavailable at runtime
    cv2 = None  # type: ignore[assignment]

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QSize, Signal

# Pillow 10 removed Image.ANTIALIAS; map it to LANCZOS for older call sites.
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS


DEFAULTS = {
    "MARKERS_CSV": "",
    "AUDIO_PATH": "",
    "IMAGES_DIRS": [],
    "OUT_QUICK": "",
    "OUT_FULL": "",
    "FFMPEG_PATH": "ffmpeg",
    "FFPROBE_PATH": "ffprobe",
    "USE_NVENC": True,
    "USE_SHORTEST": False,
    "PRE_NORMALIZE_IMAGES": True,
    "PRE_SAVE_FORMAT": "JPEG",
    "PRE_JPEG_QUALITY": 90,
    "FPS": 25,
    "FRAME_W": 1920,
    "FRAME_H": 1080,
    "AUDIO_GAIN_DB": 0.0,
    "AUDIO_OFFSET_SEC": 0.0,
    "TIMECODE_BASE_S": None,
    "SNAP_MODE": "round",
    "QUICK_START_MODE": "first_marker",
    "QUICK_PREROLL_S": 2.0,
    "QUICK_START_S": 0.0,
    "QUICK_START_INTERVAL_INDEX": 0,
    "QUICK_DURATION_S": 20.0,
    "QUICK_N_INTERVALS": 10,
    "PROJECT_JSON": str(Path.home() / "timeline_project.json"),
    "LOOP_IMAGES": True,
    "IMPACT_KIND": "disabled",
    "IMPACT_DURATION_S": 0.08,
    "IMPACT_ENABLED": False,
    "FX_GLOW_ENABLED": False,
    "FX_GLOW_COLOR": "#00f6ff",
    "FX_GLOW_WIDTH": 8,
    "FX_GLOW_INTENSITY": 0.75,
    "FX_GLOW_PRESET": "legacy",
    "WAVEFORM_ENABLED": False,
    "WAVEFORM_STYLE": "oscilloscope",
    "WAVEFORM_HEIGHT_PCT": 0.18,
    "WAVEFORM_COLOR": "#00f6ff",
    "WAVEFORM_OPACITY": 0.6,
}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"]
PIXELS_PER_SECOND = 80

PREFERRED_TIMECODE_COLUMNS = [
    "source in",
    "source tc",
    "source timecode",
    "timeline in",
    "timeline timecode",
    "record in",
    "marker in",
    "marker time",
    "tc in",
    "timecode",
    "start",
]

NEON_COLORS = [
    ("Electric Blue", "#00f6ff"),
    ("Magenta Pulse", "#ff3cff"),
    ("Cyber Lime", "#8cff00"),
    ("Sunset Ember", "#ff7b3c"),
    ("Ice White", "#f7fdfa"),
]

PROC_GLOW_PRESETS: Dict[str, Dict[str, object]] = {
    "legacy": {
        "label": "Edge Detect (legacy)",
        "mode": "legacy",
    },
    "neon_frame": {
        "label": "Neon Frame",
        "mode": "frame",
        "thickness_pct": 0.035,
        "blur_px": 28,
        "strength": 1.0,
    },
    "pulse_vignette": {
        "label": "Pulse Vignette",
        "mode": "radial",
        "inner_pct": 0.18,
        "outer_pct": 0.65,
        "blur_px": 42,
        "strength": 0.9,
    },
    "corner_flares": {
        "label": "Corner Flares",
        "mode": "corners",
        "radius_pct": 0.22,
        "blur_px": 30,
        "strength": 0.85,
    },
}

WAVEFORM_STYLE_CHOICES = [
    ("Oscilloscope", "oscilloscope"),
    ("Wave Bars", "bars"),
]

LOG_EVENTS = os.environ.get("TIMELINE_DEBUG_LOG", "0") not in {"0", "", "false", "False"}


def _log(msg: str) -> None:
    if LOG_EVENTS:
        print(f"[Timeline] {msg}")


def _clip_with_duration(clip, duration):
    setter = getattr(clip, "with_duration", None)
    if callable(setter):
        return setter(duration)
    legacy = getattr(clip, "set_duration", None)
    if callable(legacy):
        return legacy(duration)
    try:
        clip.duration = duration
    except Exception:
        pass
    return clip


def _clip_with_opacity(clip, opacity):
    setter = getattr(clip, "with_opacity", None)
    if callable(setter):
        return setter(opacity)
    legacy = getattr(clip, "set_opacity", None)
    if callable(legacy):
        return legacy(opacity)
    return clip


def _clip_with_position(clip, pos):
    setter = getattr(clip, "with_position", None)
    if callable(setter):
        return setter(pos)
    legacy = getattr(clip, "set_position", None)
    if callable(legacy):
        return legacy(pos)
    return clip


def _clip_with_audio(clip, audio_clip):
    setter = getattr(clip, "with_audio", None)
    if callable(setter):
        return setter(audio_clip)
    legacy = getattr(clip, "set_audio", None)
    if callable(legacy):
        return legacy(audio_clip)
    try:
        clip.audio = audio_clip
    except Exception:
        pass
    return clip


def _audio_subclip(audio_clip, start, end):
    worker = getattr(audio_clip, "subclipped", None)
    if callable(worker):
        return worker(start, end)
    legacy = getattr(audio_clip, "subclip", None)
    if callable(legacy):
        return legacy(start, end)
    return audio_clip


def _audio_with_volume(audio_clip, factor: float):
    scaler = getattr(audio_clip, "with_volume_scaled", None)
    if callable(scaler):
        return scaler(factor)
    legacy = getattr(audio_clip, "volumex", None)
    if callable(legacy):
        return legacy(factor)
    return audio_clip


_PERSISTENT_TEMP_DIRS: List[Path] = []
def _resolve_cache_root() -> Path:
    candidates = [
        Path.home() / ".cache" / "timeline_builder_cache",
        Path(tempfile.gettempdir()) / "timeline_builder_cache",
    ]
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            continue
    fallback = Path(tempfile.mkdtemp(prefix="timeline_builder_cache_"))
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


_GLOBAL_CACHE_ROOT = _resolve_cache_root()


def _register_persistent_temp_dir(path: Path, limit: int = 4) -> None:
    _PERSISTENT_TEMP_DIRS.append(path)
    while len(_PERSISTENT_TEMP_DIRS) > limit:
        old = _PERSISTENT_TEMP_DIRS.pop(0)
        shutil.rmtree(old, ignore_errors=True)


def _cleanup_persistent_temp_dirs() -> None:
    while _PERSISTENT_TEMP_DIRS:
        path = _PERSISTENT_TEMP_DIRS.pop()
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_persistent_temp_dirs)


def _make_persistent_temp_dir(prefix: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix=prefix))
    _register_persistent_temp_dir(path)
    return path


def _shared_cache_dir(name: str) -> Path:
    try:
        root = _GLOBAL_CACHE_ROOT
        root.mkdir(parents=True, exist_ok=True)
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        return target
    except Exception:
        # Fall back to per-run temp if cache root cannot be created
        return _make_persistent_temp_dir(f"{name}_")


def _friendly_folder_name(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    text = text.rstrip("/\\") or text
    if ":" in text:
        try:
            name = PureWindowsPath(text).name
            if name:
                return name
        except Exception:
            pass
    try:
        name = Path(text).name
        if name:
            return name
    except Exception:
        pass
    return text

CURSOR_DEBUG = bool(int(os.environ.get("TIMELINE_CURSOR_DEBUG", "0")))
PATH_DEBUG = bool(int(os.environ.get("TIMELINE_PATH_DEBUG", "0")))

IMPACT_PRESETS = {
    "disabled": {
        "label": "Disabled",
        "mode": "none",
    },
    "bump": {
        "label": "Bump Zoom",
        "mode": "bump",
        "zoom": 1.12,
        "brightness": 1.25,
        "contrast": 1.08,
        "tint": (255, 255, 255),
        "tint_strength": 0.12,
    },
    "white_flash": {
        "label": "White Flash",
        "mode": "solid",
        "color": (255, 255, 255),
    },
    "teal_glow": {
        "label": "Teal Glow",
        "mode": "solid",
        "color": (130, 210, 255),
    },
    "neon_pulse": {
        "label": "Neon Pulse",
        "mode": "bump",
        "zoom": 1.16,
        "brightness": 1.3,
        "contrast": 1.12,
        "tint": (0, 250, 255),
        "tint_strength": 0.35,
    },
    "ember_pop": {
        "label": "Ember Pop",
        "mode": "solid",
        "color": (255, 170, 80),
    },
}

IMAGE_BUNDLE_MIME = "application/x-timeline-image-bundle"

APP_STYLE_SHEET = """
QMainWindow, QWidget {
    background-color: #0f141b;
    color: #eff4ff;
}
QGroupBox {
    background-color: #121923;
    border: 1px solid #1f2a39;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    margin-top: -8px;
    color: #88b4ff;
    font-weight: 600;
    letter-spacing: 0.5px;
}
QLabel#sectionTitle {
    font-size: 15px;
    font-weight: 600;
    color: #9ec5ff;
}
QLabel#imagePreview {
    background-color: #0b1118;
    border: 1px solid #1f2a39;
    border-radius: 10px;
    color: #6f7c91;
}
QPushButton {
    background-color: #1f6feb;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    color: #fafcff;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #2f82ff;
}
QPushButton:pressed {
    background-color: #185ed1;
}
QPushButton:disabled {
    background-color: #1b2431;
    color: #6d7382;
}
QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QListWidget,
QTreeWidget,
QTextEdit {
    background-color: #0d131b;
    border: 1px solid #1f2a39;
    border-radius: 6px;
    padding: 4px 6px;
    selection-background-color: #1f6feb;
    selection-color: #f5f9ff;
}
QComboBox QAbstractItemView {
    background-color: #0d131b;
    selection-background-color: #1f6feb;
}
QTabWidget::pane {
    border: 1px solid #1f2a39;
    border-radius: 8px;
    margin-top: 6px;
}
QTabBar::tab {
    background-color: #0f141b;
    color: #9eb0c7;
    border: 1px solid #1f2a39;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 6px 14px;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background-color: #162232;
    color: #dfe8ff;
}
QListWidget::item:hover {
    background-color: #152033;
}
QListWidget::item:selected {
    background-color: #243a5c;
    color: #f6fbff;
}
QStatusBar {
    background-color: #080b11;
    color: #9caabd;
}
QProgressBar {
    border: 1px solid #1f2a39;
    border-radius: 6px;
    background-color: #090c12;
    height: 12px;
}
QProgressBar::chunk {
    border-radius: 6px;
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #3f8efc, stop:1 #64e5ff);
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #0b1018;
    border-radius: 6px;
    margin: 0px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #1b2737;
    border-radius: 6px;
    min-height: 30px;
    min-width: 30px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #2a3a4f;
}
QSplitter::handle {
    background: #111823;
    margin: 0 4px;
}
"""


THUMB_CACHE: Dict[Tuple[str, int, int, str], QtGui.QPixmap] = {}


def install_app_style(app: QtWidgets.QApplication) -> None:
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    palette = QtGui.QPalette()
    base = QtGui.QColor("#0d131b")
    alt = QtGui.QColor("#151c26")
    text = QtGui.QColor("#f1f5ff")
    disabled_text = QtGui.QColor("#6f7784")
    highlight = QtGui.QColor("#1f6feb")
    palette.setColor(QtGui.QPalette.Window, base)
    palette.setColor(QtGui.QPalette.WindowText, text)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#090d13"))
    palette.setColor(QtGui.QPalette.AlternateBase, alt)
    palette.setColor(QtGui.QPalette.Text, text)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#111824"))
    palette.setColor(QtGui.QPalette.ButtonText, text)
    palette.setColor(QtGui.QPalette.Highlight, highlight)
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#fefefe"))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled_text)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled_text)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, disabled_text)
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#111824"))
    palette.setColor(QtGui.QPalette.ToolTipText, text)
    app.setPalette(palette)
    font = app.font()
    if font.pointSize() < 10:
        font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(APP_STYLE_SHEET)
try:
    THUMB_CACHE_DIR = Path(tempfile.gettempdir()) / "timeline_builder_thumbs"
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    THUMB_CACHE_DIR = None


def _thumbnail_cache_path(path: Path, width: int, height: int) -> Optional[Path]:
    if not THUMB_CACHE_DIR:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    try:
        if _is_running_in_wsl() and ":" in str(path):
            key_path = str(_windows_to_wsl_path(str(path)))
        else:
            key_path = str(path)
        key_path = os.path.normcase(os.path.normpath(key_path))
    except Exception:
        key_path = str(path)
    key = f"{key_path}|{stat.st_mtime_ns}|{stat.st_size}|{width}x{height}|1"
    digest = hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()
    return THUMB_CACHE_DIR / f"{digest}.jpg"


def _load_cached_thumbnail(cache_path: Optional[Path]) -> Optional[QtGui.QPixmap]:
    if not cache_path or not cache_path.exists():
        return None
    pix = QtGui.QPixmap(str(cache_path))
    if pix.isNull():
        try:
            cache_path.unlink()
        except Exception:
            pass
        return None
    return pix


def _save_cached_thumbnail(cache_path: Optional[Path], pix: QtGui.QPixmap) -> None:
    if not cache_path or pix.isNull():
        return
    try:
        pix.save(str(cache_path), "JPG", 85)
    except Exception:
        try:
            cache_path.unlink()
        except Exception:
            pass


def _composite_on_canvas(source: QtGui.QPixmap, width: int, height: int) -> QtGui.QPixmap:
    canvas = QtGui.QPixmap(width, height)
    canvas.fill(QtGui.QColor("#050505"))
    if not source.isNull():
        scaled = source.scaled(
            width,
            height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        painter = QtGui.QPainter(canvas)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        x = (width - scaled.width()) // 2
        y = (height - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
    return canvas


def _thumbnail_from_qimage_reader(path: Path, width: int, height: int) -> Optional[QtGui.QPixmap]:
    reader = QtGui.QImageReader(str(path))
    reader.setAutoTransform(True)
    try:
        size = reader.size()
    except Exception:
        size = QtCore.QSize()
    if size.isValid() and size.width() > 0 and size.height() > 0:
        target_w = max(1, int(width * 2))
        target_h = max(1, int(height * 2))
        scale = min(
            target_w / float(size.width()),
            target_h / float(size.height()),
            1.0,
        )
        scaled = QtCore.QSize(
            max(1, int(size.width() * scale)),
            max(1, int(size.height() * scale)),
        )
    else:
        scaled = QtCore.QSize(width * 2, height * 2)
    if scaled.isValid():
        reader.setScaledSize(scaled)
    qimg = reader.read()
    if qimg.isNull():
        return None
    pix = QtGui.QPixmap.fromImage(qimg)
    return _composite_on_canvas(pix, width, height)


def _thumbnail_from_pil(path: Path, width: int, height: int) -> Optional[QtGui.QPixmap]:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((width * 2, height * 2), Image.LANCZOS)
        data = img.tobytes("raw", "RGB")
        qimg = QtGui.QImage(
            data,
            img.width,
            img.height,
            img.width * 3,
            QtGui.QImage.Format_RGB888,
        )
        pix = QtGui.QPixmap.fromImage(qimg)
        return _composite_on_canvas(pix, width, height)
    except Exception:
        return None


def get_thumbnail(path: str, width: int = 120, height: int = 90, placeholder: str = "No\nImg") -> QtGui.QPixmap:
    key = (path or "", width, height, placeholder)
    cached = THUMB_CACHE.get(key)
    if cached:
        return cached

    pix: Optional[QtGui.QPixmap] = None
    p = Path(path) if path else None
    if p and p.exists():
        cache_path = _thumbnail_cache_path(p, width, height)
        pix = _load_cached_thumbnail(cache_path)
        if pix is None:
            pix = _thumbnail_from_qimage_reader(p, width, height)
        if pix is None:
            pix = _thumbnail_from_pil(p, width, height)
        if pix and cache_path:
            _save_cached_thumbnail(cache_path, pix)

    if pix is None or pix.isNull():
        pix = QtGui.QPixmap(width, height)
        pix.fill(QtGui.QColor("#1b1f27"))
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(QtGui.QPen(QtGui.QColor("#dfe3ee")))
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSizeF(min(width, height) / 6.5)
        painter.setFont(font)
        painter.drawText(pix.rect(), Qt.AlignCenter, placeholder)
        painter.end()

    THUMB_CACHE[key] = pix
    return pix


def _is_running_in_wsl() -> bool:
    try:
        release = platform.release().lower()
        version = platform.version().lower()
    except Exception:
        return False
    return "microsoft" in release or "microsoft" in version


def _windows_to_wsl_path(win_path: str) -> Path:
    win_path = win_path.strip()
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return Path("/mnt") / drive / rest.lstrip("/")
    return Path(win_path)


def _default_sample_root() -> Path:
    if _is_running_in_wsl():
        win_home = os.environ.get("USERPROFILE")
        if win_home:
            return _windows_to_wsl_path(win_home) / "BeatTimelineSamples"
    return Path.home() / "BeatTimelineSamples"


def _format_timecode(seconds: float, fps: int) -> str:
    frames = int(round(seconds * fps))
    hh = frames // (3600 * fps)
    rem = frames % (3600 * fps)
    mm = rem // (60 * fps)
    rem = rem % (60 * fps)
    ss = rem // fps
    ff = rem % fps
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _hex_to_rgb(value: str, fallback: Tuple[int, int, int] = (0, 255, 255)) -> Tuple[int, int, int]:
    if not value:
        return fallback
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return fallback
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
        return (r, g, b)
    except ValueError:
        return fallback


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def _create_sample_markers_csv(path: Path, fps: int = 25, segments: int = 6, spacing_s: float = 2.0) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["Marker Name,Source In"]
    for idx in range(segments + 1):
        seconds = spacing_s * idx
        rows.append(f"Marker {idx+1:02d},{_format_timecode(seconds, fps)}")
    path.write_text("\n".join(rows), encoding="utf-8")


def _create_sample_image(path: Path, color: Tuple[int, int, int], label: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1920, 1080), color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 120)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    text = label
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((1920 - text_w) / 2, (1080 - text_h) / 2 - 40),
        text,
        fill=(255, 255, 255),
        font=font,
    )
    subtitle = "Beat Timeline Sample"
    bbox2 = draw.textbbox((0, 0), subtitle, font=sub_font)
    draw.text(
        ((1920 - (bbox2[2] - bbox2[0])) / 2, 1080 - 160),
        subtitle,
        fill=(240, 240, 240),
        font=sub_font,
    )
    img.save(path, "JPEG", quality=92, optimize=True, progressive=True)


def _create_sample_audio(path: Path, duration_s: float = 12.0) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    amplitude = 0.25
    frequencies = [440.0, 554.37, 659.25, 880.0]
    total_frames = int(duration_s * sample_rate)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frame_index = 0
        chunk_size = 2048
        while frame_index < total_frames:
            remaining = total_frames - frame_index
            batch = min(chunk_size, remaining)
            buffer = bytearray()
            for i in range(batch):
                global_index = frame_index + i
                phase = global_index / sample_rate
                freq = frequencies[(global_index // (sample_rate * 3)) % len(frequencies)]
                sample = int(
                    amplitude * 32767 * math.sin(2 * math.pi * freq * phase)
                )
                buffer.extend(struct.pack("<h", sample))
                buffer.extend(struct.pack("<h", sample))
            wav_file.writeframes(buffer)
            frame_index += batch


def ensure_sample_assets() -> Path:
    root = _default_sample_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        root = Path.home() / "BeatTimelineSamples"
        root.mkdir(parents=True, exist_ok=True)

    images_dir = root / "images"
    _create_sample_markers_csv(root / "markers.csv")
    _create_sample_audio(root / "audio.wav")

    colors = [
        ((219, 68, 83), "Sample 01"),
        ((142, 68, 173), "Sample 02"),
        ((65, 131, 215), "Sample 03"),
        ((38, 166, 91), "Sample 04"),
        ((243, 156, 18), "Sample 05"),
        ((230, 126, 34), "Sample 06"),
    ]
    for idx, (color, label) in enumerate(colors, start=1):
        _create_sample_image(images_dir / f"sample_{idx:02d}.jpg", color, label)

    return root


SAMPLE_ROOT = ensure_sample_assets()
DEFAULTS.update(
    {
        "MARKERS_CSV": str(SAMPLE_ROOT / "markers.csv"),
        "AUDIO_PATH": str(SAMPLE_ROOT / "audio.wav"),
        "IMAGES_DIRS": [str(SAMPLE_ROOT / "images")],
        "OUT_QUICK": str(SAMPLE_ROOT / "quick_preview.mp4"),
        "OUT_FULL": str(SAMPLE_ROOT / "final_export.mp4"),
        "PROJECT_JSON": str(SAMPLE_ROOT / "BeatTimelineProject.json"),
    }
)


# =========================
# Time & math utils
# =========================
def parse_tc_to_frames(tc: str, fps: int) -> int:
    tc = str(tc).strip()
    if not tc:
        raise ValueError("Empty timecode string.")

    normalized = re.sub(r"[;,]", ":", tc)
    parts = [p.strip() for p in normalized.split(":")]
    if len(parts) == 4:
        try:
            hh, mm, ss, ff = (int(value) for value in parts)
        except ValueError:
            pass
        else:
            return ((hh * 3600) + (mm * 60) + ss) * fps + ff

    digits = re.findall(r"\d+", tc)
    if len(digits) == 4:
        hh, mm, ss, ff = (int(value) for value in digits)
        return ((hh * 3600) + (mm * 60) + ss) * fps + ff

    raise ValueError(f"Invalid timecode (expected HH:MM:SS:FF): {tc}")


def frames_to_seconds(frames: int, fps: int) -> float:
    return frames / float(fps)


def seconds_to_tc(seconds: float, fps: int) -> str:
    frames = int(round(seconds * fps))
    hh = frames // (3600 * fps)
    rem = frames % (3600 * fps)
    mm = rem // (60 * fps)
    rem = rem % (60 * fps)
    ss = rem // fps
    ff = rem % fps
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def infer_timecode_base_seconds(marker_frames: List[int], fps: int) -> float:
    if not marker_frames:
        return 0.0
    min_s = frames_to_seconds(min(marker_frames), fps)
    if min_s >= 3600.0:
        base_hours = int(min_s // 3600)
        return float(base_hours * 3600)
    return 0.0


def snap_to_frame(seconds: float, fps: int, mode: str = "round") -> float:
    f = seconds * fps
    if mode == "floor":
        f = math.floor(f)
    elif mode == "ceil":
        f = math.ceil(f)
    else:
        f = int(round(f))
    return f / fps


def span_overlap(a0: float, a1: float, b0: float, b1: float) -> Tuple[float, float]:
    start = max(a0, b0)
    end = min(a1, b1)
    if end <= start:
        return 0.0, 0.0
    return start, end


# =========================
# Media & interval helpers
# =========================
def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


IMPACT_EPS = 1e-6


def _impact_enabled(kind: str, dur: float) -> bool:
    preset = IMPACT_PRESETS.get(kind or "disabled")
    return bool(preset) and preset.get("mode") != "none" and dur > IMPACT_EPS


def _impact_tail_length(duration: float, kind: str, has_image: bool, tail_request: float) -> float:
    if not has_image or not _impact_enabled(kind, tail_request):
        return 0.0
    return min(max(0.0, duration), max(0.0, tail_request))


def split_segment_phases(seg: Dict[str, object], impact_kind: str, impact_duration: float):
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    duration = max(0.0, end - start)
    has_image = seg.get("kind") == "image" and bool(seg.get("img"))
    tail = _impact_tail_length(duration, impact_kind, has_image, impact_duration)
    if tail <= IMPACT_EPS:
        return [("base", start, end)]
    pivot = max(start, end - tail)
    phases = []
    if pivot > start + IMPACT_EPS:
        phases.append(("base", start, pivot))
    phases.append(("impact", pivot, end))
    return phases


def split_duration_for_impact(
    duration: float, has_image: bool, impact_kind: str, impact_duration: float
) -> Tuple[float, float]:
    tail = _impact_tail_length(duration, impact_kind, has_image, impact_duration)
    base = max(0.0, duration - tail)
    return base, tail


def _impact_cache_key(kind: str, source_path: str) -> str:
    try:
        real = str(Path(source_path).resolve())
    except Exception:
        real = str(source_path)
    digest = hashlib.sha1(f"{kind}|{real}".encode("utf-8")).hexdigest()
    return digest


def _ensure_solid_variant(kind: str, preset: Dict[str, object], cache_dir: Path) -> Optional[str]:
    color = tuple(preset.get("color", (255, 255, 255)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"impact_{kind}.png"
    if not target.exists():
        img = Image.new("RGB", (8, 8), color)
        img.save(target, "PNG")
    return str(target)


def _ensure_bump_variant(kind: str, source_path: str, preset: Dict[str, object], cache_dir: Path) -> Optional[str]:
    if not source_path or not Path(source_path).exists():
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(source_path).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        ext = ".jpg"
    target = cache_dir / f"{_impact_cache_key(kind, source_path)}{ext}"
    if target.exists():
        return str(target)
    try:
        img = Image.open(source_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return None

    orig_size = img.size
    zoom = float(preset.get("zoom", 1.1))
    if zoom > 1.0:
        crop_w = max(1, int(round(orig_size[0] / zoom)))
        crop_h = max(1, int(round(orig_size[1] / zoom)))
        left = max(0, (orig_size[0] - crop_w) // 2)
        top = max(0, (orig_size[1] - crop_h) // 2)
        img = img.crop((left, top, left + crop_w, top + crop_h))
        img = img.resize(orig_size, Image.LANCZOS)

    brightness = float(preset.get("brightness", 1.0))
    if abs(brightness - 1.0) > 1e-3:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    contrast = float(preset.get("contrast", 1.0))
    if abs(contrast - 1.0) > 1e-3:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    tint = preset.get("tint")
    if tint:
        overlay = Image.new("RGB", img.size, tuple(tint))
        strength = max(0.0, min(1.0, float(preset.get("tint_strength", 0.15))))
        img = Image.blend(img, overlay, strength)
    try:
        img.save(target, "JPEG", quality=95)
    except Exception:
        return None
    return str(target)


def ensure_impact_variant_image(kind: str, source_path: str, cache_dir: Optional[Path]) -> Optional[str]:
    if not cache_dir or not source_path:
        return source_path
    preset = IMPACT_PRESETS.get(kind or "disabled")
    if not preset or preset.get("mode") == "none":
        return source_path
    mode = preset.get("mode")
    if mode == "solid":
        return _ensure_solid_variant(kind, preset, cache_dir)
    if mode == "bump":
        variant = _ensure_bump_variant(kind, source_path, preset, cache_dir)
        if variant:
            return variant
        fallback = IMPACT_PRESETS.get("white_flash")
        if fallback:
            return _ensure_solid_variant("white_flash", fallback, cache_dir)
    return source_path


def _fallback_glow_mask_from_luma(img: Image.Image, spread: int) -> Image.Image:
    base = ImageOps.grayscale(img.convert("RGB"))
    edges = base.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    extrema = edges.getextrema()
    if not extrema or extrema[1] <= 0:
        edges = base.filter(ImageFilter.GaussianBlur(radius=max(1, spread * 0.4)))
    blur_radius = max(1, int(spread * 0.7))
    return edges.filter(ImageFilter.GaussianBlur(radius=blur_radius))


_GLOW_VARIANT_VERSION = "v5_proc_glow"


def _fast_glow_path_available() -> bool:
    return cv2 is not None and np is not None


def _ensure_glow_variant_image_fast(src_path: Path, glow_opts: GlowOptions, target: Path) -> Optional[str]:
    if not _fast_glow_path_available():
        return None
    try:
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if img is None:
        return None
    if img.ndim == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.full(base.shape[:2], 255, dtype=np.uint8)
    else:
        channels = img.shape[2]
        if channels == 4:
            alpha = img[:, :, 3]
            base = img[:, :, :3]
        elif channels == 3:
            base = img[:, :, :3]
            alpha = np.full(base.shape[:2], 255, dtype=np.uint8)
        else:
            return None
    spread = max(1, int(glow_opts.width))
    kernel_size = max(3, spread * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(alpha, kernel, iterations=1)
    blur_outer = cv2.GaussianBlur(
        dilated, (0, 0), sigmaX=max(1.0, spread * 0.65), sigmaY=max(1.0, spread * 0.65)
    )
    blur_inner = cv2.GaussianBlur(
        alpha, (0, 0), sigmaX=max(0.8, spread * 0.4), sigmaY=max(0.8, spread * 0.4)
    )
    halo = cv2.subtract(blur_outer, blur_inner)
    low = max(20, min(140, spread * 8))
    high = max(low * 2, 120)
    edges = cv2.Canny(alpha, low, high)
    halo = cv2.addWeighted(halo, 0.85, edges, 0.5, 0.0)
    halo = cv2.normalize(halo, None, 0, 255, cv2.NORM_MINMAX)
    halo = cv2.convertScaleAbs(halo, alpha=0.8 + glow_opts.intensity * 0.9)
    halo = np.clip(halo.astype(np.float32) * glow_opts.intensity, 0, 255).astype(np.uint8)
    if int(halo.max()) <= 0:
        return None
    alpha_factor = np.clip(halo.astype(np.float32) / 255.0, 0.0, 1.0)
    alpha_factor = alpha_factor[..., None]
    glow_color = np.array(glow_opts.color[::-1], dtype=np.float32)
    overlay = glow_color * alpha_factor
    base = base.astype(np.float32)
    result = np.clip(base * (1.0 - alpha_factor) + overlay, 0, 255).astype(np.uint8)
    try:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    except Exception:
        return None
    try:
        Image.fromarray(result_rgb).save(target, "PNG", optimize=True)
    except Exception:
        return None
    return str(target)


def _ensure_glow_variant_image_pillow(src_path: Path, glow_opts: GlowOptions, target: Path) -> Optional[str]:
    try:
        img = Image.open(src_path).convert("RGBA")
    except Exception:
        return None

    alpha = img.split()[3] if "A" in img.getbands() else Image.new("L", img.size, 255)
    spread = max(1, int(glow_opts.width))
    pad = max(2, spread * 2)
    alpha_padded = ImageOps.expand(alpha, border=pad, fill=0)
    edge = alpha_padded.filter(ImageFilter.FIND_EDGES)
    dilated = alpha_padded.filter(ImageFilter.MaxFilter(spread * 2 + 1))
    blurred = dilated.filter(ImageFilter.GaussianBlur(radius=spread * 0.9))
    inner = alpha_padded.filter(ImageFilter.GaussianBlur(radius=max(1, int(spread * 0.4))))
    halo = ImageChops.subtract(blurred, inner)
    halo = ImageChops.add(halo, edge)
    halo = halo.crop((pad, pad, pad + alpha.width, pad + alpha.height))
    extrema = halo.getextrema()
    if not extrema or extrema[1] <= 0:
        halo = _fallback_glow_mask_from_luma(img, spread)
    halo = ImageEnhance.Brightness(halo).enhance(1.4 + glow_opts.intensity)
    halo = halo.point(lambda p: min(255, int(p * glow_opts.intensity)))

    glow_color = tuple(max(0, min(255, int(c))) for c in glow_opts.color)
    glow_layer = Image.new("RGBA", img.size, glow_color + (0,))
    glow_layer.putalpha(halo)

    canvas = Image.alpha_composite(img, glow_layer)
    canvas = canvas.convert("RGB")
    try:
        canvas.save(target, "PNG", optimize=True)
    except Exception:
        return None
    return str(target)


def _generate_proc_glow_mask(size: Tuple[int, int], preset: Dict[str, object]) -> Optional[Image.Image]:
    w, h = size
    if w <= 2 or h <= 2:
        return None
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    mode = str(preset.get("mode", "frame"))
    if mode == "frame":
        thickness_pct = float(preset.get("thickness_pct", 0.04))
        border = max(1, int(min(w, h) * max(0.005, thickness_pct)))
        draw.rectangle((0, 0, w - 1, h - 1), fill=255)
        if border * 2 < w and border * 2 < h:
            draw.rectangle((border, border, w - border, h - border), fill=0)
    elif mode == "radial":
        min_dim = min(w, h)
        outer_pct = max(0.1, min(1.0, float(preset.get("outer_pct", 0.7))))
        inner_pct = max(0.0, min(outer_pct - 0.05, float(preset.get("inner_pct", 0.2))))
        outer_r = max(4, int(min_dim * outer_pct * 0.5))
        inner_r = max(0, int(min_dim * inner_pct * 0.5))
        cx = w / 2.0
        cy = h / 2.0
        bbox_outer = (cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r)
        draw.ellipse(bbox_outer, fill=255)
        if inner_r > 0:
            bbox_inner = (cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r)
            draw.ellipse(bbox_inner, fill=0)
    elif mode == "corners":
        min_dim = min(w, h)
        radius = max(6, int(min_dim * float(preset.get("radius_pct", 0.25))))
        diameter = radius * 2
        boxes = [
            (0, 0, min(w, diameter), min(h, diameter)),
            (max(0, w - diameter), 0, w, min(h, diameter)),
            (0, max(0, h - diameter), min(w, diameter), h),
            (max(0, w - diameter), max(0, h - diameter), w, h),
        ]
        for box in boxes:
            draw.ellipse(box, fill=255)
    else:
        return None
    blur_px = max(0, int(preset.get("blur_px", 24)))
    if blur_px:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_px))
    strength = float(preset.get("strength", 1.0))
    if abs(strength - 1.0) > 1e-3:
        factor = max(0.05, min(2.0, strength))
        mask = mask.point(lambda p: min(255, int(p * factor)))
    return mask


def _ensure_glow_variant_image_procedural(src_path: Path, glow_opts: GlowOptions, target: Path) -> Optional[str]:
    preset = PROC_GLOW_PRESETS.get(glow_opts.preset or "legacy")
    if not preset or preset.get("mode") == "legacy":
        return None
    try:
        base = Image.open(src_path).convert("RGBA")
    except Exception:
        return None
    mask = _generate_proc_glow_mask(base.size, preset)
    if mask is None:
        return None
    opacity = float(preset.get("opacity", 0.9))
    opacity = max(0.05, min(1.0, opacity))
    alpha = mask.point(lambda p: min(255, int(p * opacity)))
    preset_color = preset.get("color")
    if preset_color:
        color_rgb = tuple(int(c) for c in preset_color)
    else:
        color_rgb = glow_opts.color
    overlay = Image.new("RGBA", base.size, color_rgb + (0,))
    overlay.putalpha(alpha)
    canvas = Image.alpha_composite(base, overlay)
    if "brightness" in preset:
        val = float(preset.get("brightness", 1.0))
        if abs(val - 1.0) > 1e-3:
            canvas = ImageEnhance.Brightness(canvas).enhance(val)
    if "contrast" in preset:
        val = float(preset.get("contrast", 1.0))
        if abs(val - 1.0) > 1e-3:
            canvas = ImageEnhance.Contrast(canvas).enhance(val)
    try:
        canvas.convert("RGB").save(target, "PNG", optimize=True)
    except Exception:
        return None
    return str(target)


def ensure_glow_variant_image(source_path: str, glow_opts: GlowOptions, cache_dir: Optional[Path]) -> str:
    if not glow_opts.enabled or not cache_dir or not source_path:
        return source_path
    src_path = Path(source_path)
    if not src_path.exists():
        return source_path
    try:
        stat = src_path.stat()
    except OSError:
        stat = None
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (
        f"{src_path.resolve()}|{stat.st_mtime_ns if stat else 0}|{stat.st_size if stat else 0}|"
        f"{glow_opts.color}|{glow_opts.width}|{glow_opts.intensity}|{glow_opts.preset}|{_GLOW_VARIANT_VERSION}"
    )
    digest = hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()
    target = cache_dir / f"glow_{digest}.png"
    if target.exists():
        return str(target)
    preset_key = str(glow_opts.preset or "legacy").lower()
    if preset_key != "legacy":
        preset_variant = _ensure_glow_variant_image_procedural(src_path, glow_opts, target)
        if preset_variant:
            return preset_variant
    fast_variant = _ensure_glow_variant_image_fast(src_path, glow_opts, target)
    if fast_variant:
        return fast_variant
    legacy_variant = _ensure_glow_variant_image_pillow(src_path, glow_opts, target)
    if legacy_variant:
        return legacy_variant
    return source_path


def _precache_glow_variants(
    paths: Iterable[str],
    glow_opts: GlowOptions,
    cache_dir: Optional[Path],
    status_cb: Callable[[str], None],
    cancel_requested: Optional[Callable[[], bool]] = None,
    label: str = "Stage: glow cache",
) -> Dict[str, str]:
    if not glow_opts.enabled or not cache_dir:
        return {}
    unique: List[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not raw or raw in seen:
            continue
        p = Path(raw)
        if not p.exists():
            continue
        seen.add(raw)
        unique.append(raw)
    total = len(unique)
    if total == 0:
        return {}
    max_workers = max(1, min(os.cpu_count() or 4, total))
    status_cb(f"{label}: 0/{total} (x{max_workers})")
    results: Dict[str, str] = {}
    step = max(1, total // 8 or 1)

    def _worker(src: str) -> str:
        return ensure_glow_variant_image(src, glow_opts, cache_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, src): src for src in unique}
        completed = 0
        for future in as_completed(futures):
            if cancel_requested and cancel_requested():
                raise RuntimeError("Render cancelled (glow cache)")
            src = futures[future]
            variant = ""
            try:
                variant = future.result()
            except Exception as exc:
                print(f"[Glow] worker failed on {Path(src).name}: {exc}")
            if variant:
                results[src] = variant
            completed += 1
            if completed % step == 0 or completed == total:
                status_cb(f"{label}: {completed}/{total} (x{max_workers})")
    return results


def _build_waveform_filter(width: int, height: int, fps: int, style: str, color: Tuple[int, int, int], opacity: float) -> str:
    style = style or "oscilloscope"
    if style not in {"oscilloscope", "bars"}:
        style = "oscilloscope"
    mode = "line" if style == "oscilloscope" else "cline"
    color_hex = _rgb_to_hex(color).lstrip("#") or "00ffff"
    alpha_val = int(max(0.05, min(1.0, opacity)) * 255)
    alpha_expr = f"if(gt(r(X,Y)+g(X,Y)+b(X,Y),0),{alpha_val},0)"
    filter_chain = (
        f"showwaves=s={width}x{height}:mode={mode}:rate={fps}:colors=0x{color_hex},"
        "format=rgba,"
        f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{alpha_expr}'"
    )
    return filter_chain


def generate_waveform_overlay_video(
    ffmpeg_path: str,
    audio_path: str,
    width: int,
    height: int,
    fps: int,
    style: str,
    color: Tuple[int, int, int],
    opacity: float,
    start_s: float,
    duration_s: float,
    cache_dir: Path,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> Optional[Path]:
    audio_file = Path(audio_path)
    if not audio_path or not audio_file.exists() or duration_s <= 0:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        stat = audio_file.stat()
    except OSError:
        stat = None
    cache_key = (
        f"{audio_file.resolve()}|{stat.st_mtime_ns if stat else 0}|{stat.st_size if stat else 0}|"
        f"{width}x{height}|{fps}|{style}|{color}|{opacity:.4f}|{start_s:.3f}|{duration_s:.3f}"
    )
    digest = hashlib.sha1(cache_key.encode("utf-8", "ignore")).hexdigest()
    output = cache_dir / f"waveform_{digest}.mov"
    if output.exists():
        return output
    if cancel_requested and cancel_requested():
        return None
    filter_chain = _build_waveform_filter(width, height, fps, style, color, opacity)
    cmd = [ffmpeg_path, "-hide_banner", "-y"]
    if start_s > 0:
        cmd += ["-ss", f"{max(0.0, start_s):.6f}"]
    cmd += [
        "-t",
        f"{max(0.01, duration_s):.6f}",
        "-i",
        str(audio_file),
        "-filter_complex",
        f"[0:a]{filter_chain}[wave]",
        "-map",
        "[wave]",
        "-pix_fmt",
        "argb",
        "-r",
        str(fps),
        "-c:v",
        "qtrle",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return output if output.exists() else None

def _detect_timecode_column(df: pd.DataFrame, fps: int) -> str:
    normalized = {c.strip().lower(): c for c in df.columns}
    for key in PREFERRED_TIMECODE_COLUMNS:
        if key in normalized:
            return normalized[key]

    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        hits = 0
        for value in series.astype(str).head(25):
            try:
                parse_tc_to_frames(value, fps)
                hits += 1
                if hits >= 2:
                    return column
            except Exception:
                continue
    raise KeyError("No column with timecode-like values was found in the CSV.")


def read_markers_frames(csv_path: str, fps: int) -> List[int]:
    df = pd.read_csv(csv_path)
    src_col = _detect_timecode_column(df, fps)
    frames = []
    for t in df[src_col].dropna().astype(str):
        try:
            frames.append(parse_tc_to_frames(t, fps))
        except Exception:
            pass
    frames = sorted(set(frames))
    return frames


def build_intervals(marker_frames: List[int]) -> List[Tuple[int, int]]:
    intervals = []
    for i in range(len(marker_frames) - 1):
        a, b = marker_frames[i], marker_frames[i + 1]
        if b > a:
            intervals.append((a, b))
    return intervals


def find_images_recursive(root: str, allowed_exts: Iterable[str]) -> List[Path]:
    root_path = Path(root)
    exts = {e.lower() for e in allowed_exts}
    imgs = [p for p in root_path.rglob("*") if p.suffix.lower() in exts]
    imgs.sort()
    return imgs


def assign_images(
    intervals: List[Tuple[int, int]],
    images: List[Path],
    loop: bool = True,
    prior_assign: Optional[List[str]] = None,
) -> List[str]:
    n = len(intervals)
    out = [""] * n
    if prior_assign:
        for i in range(min(n, len(prior_assign))):
            if prior_assign[i] and Path(prior_assign[i]).exists():
                out[i] = prior_assign[i]
    img_list = [str(p) for p in images]
    if not img_list:
        raise RuntimeError("No images found to assign.")
    idx = 0
    for i in range(n):
        if not out[i]:
            if idx >= len(img_list):
                if loop:
                    idx = 0
                else:
                    out[i] = img_list[-1]
                    continue
            out[i] = img_list[idx]
            idx += 1
    return out


def render_quick_windowed(
    out_path,
    ffmpeg_path: str,
    fps,
    frame_size,
    audio_path,
    audio_gain_db,
    segments,
    window_start_s,
    window_dur_s,
    tc_base_s,
    audio_offset_s,
    snap_mode,
    impact_kind,
    impact_duration,
    glow_opts: GlowOptions,
    wave_opts: WaveformOptions,
    progress_cb,
    status_cb,
    cancel_requested: Optional[Callable[[], bool]] = None,
):
    def _check_cancel(stage: str = "") -> None:
        if cancel_requested and cancel_requested():
            raise RuntimeError("Render cancelled" + (f" ({stage})" if stage else ""))

    window_start_s = snap_to_frame(window_start_s, fps, snap_mode)
    window_end_s = snap_to_frame(window_start_s + window_dur_s, fps, "ceil")
    window_dur_s = max(0.0, window_end_s - window_start_s)
    _check_cancel("init")
    impact_kind = impact_kind or "disabled"
    impact_duration = max(0.0, float(impact_duration))
    impact_cache_dir: Optional[Path] = None
    if _impact_enabled(impact_kind, impact_duration):
        impact_cache_dir = _shared_cache_dir("impact_fx")
    glow_cache_dir: Optional[Path] = None
    glow_variant_cache: Dict[Tuple[str, Tuple[int, int, int], float, float], str] = {}
    if glow_opts.enabled:
        msg = "Quick: generating glow variants…"
        status_cb(msg)
        print(f"[Timeline] {msg}")
        glow_cache_dir = _shared_cache_dir("glow_fx")
        _log(f"Quick glow cache dir: {glow_cache_dir}")
        _check_cancel("glow cache")
        progress_cb(0.05)
        precached = _precache_glow_variants(
            (seg.get("img", "") for seg in segments if seg.get("img")),
            glow_opts,
            glow_cache_dir,
            status_cb,
            cancel_requested,
            label="Quick: glow cache",
        )
        for raw_path, variant in precached.items():
            key = (raw_path, tuple(glow_opts.color), float(glow_opts.width), float(glow_opts.intensity))
            glow_variant_cache[key] = variant
    wave_cache_dir: Optional[Path] = None
    wave_clip: Optional[VideoFileClip] = None
    waveform_height = 0
    audio_window_start = max(0.0, (window_start_s - tc_base_s) + audio_offset_s)
    if wave_opts.enabled and wave_opts.style != "disabled":
        msg = "Quick: synthesizing waveform overlay…"
        status_cb(msg)
        print(f"[Timeline] {msg}")
        waveform_height = max(1, int(frame_size[1] * wave_opts.height_pct))
        wave_cache_dir = _shared_cache_dir("wave_overlay")
        _log(f"Quick waveform dir: {wave_cache_dir}")
        _check_cancel("waveform prep")
        progress_cb(0.12)
        overlay_video = generate_waveform_overlay_video(
            ffmpeg_path,
            audio_path,
            frame_size[0],
            waveform_height,
            fps,
            wave_opts.style,
            wave_opts.color,
            wave_opts.opacity,
            audio_window_start,
            window_dur_s,
            wave_cache_dir,
            cancel_requested=cancel_requested,
        )
        if overlay_video and Path(overlay_video).exists():
            _log(f"Quick waveform clip: {overlay_video}")
            try:
                wave_clip = VideoFileClip(str(overlay_video), audio=False, has_mask=True)
                wave_clip = _clip_with_opacity(wave_clip, wave_opts.opacity)
            except Exception as exc:
                print(f"[Timeline] Waveform overlay load failed: {exc}")
                wave_clip = None

    clips = []
    normalized_cache: Dict[str, str] = {}
    impact_variant_cache: Dict[Tuple[str, str, float], str] = {}
    cursor = 0.0
    status_cb(
        f"Quick: {seconds_to_tc(window_start_s, fps)} → {seconds_to_tc(window_end_s, fps)}"
    )

    total_segments = len(segments)
    processed_segments = 0
    for seg in segments:
        _check_cancel("compositing")
        phases = split_segment_phases(seg, impact_kind, impact_duration)
        for phase_kind, phase_start, phase_end in phases:
            ovl0, ovl1 = span_overlap(phase_start, phase_end, window_start_s, window_end_s)
            if ovl1 <= ovl0:
                continue
            ovl0 = snap_to_frame(ovl0, fps, "ceil")
            ovl1 = snap_to_frame(ovl1, fps, "floor")
            if ovl1 <= ovl0:
                continue
            dur = max(0.0, ovl1 - ovl0)
            if dur <= 0.0:
                continue

            if seg["kind"] == "black" or not seg.get("img"):
                clip = _clip_with_duration(ColorClip(size=frame_size, color=(0, 0, 0)), dur)
            else:
                raw_path = seg.get("img", "")
                base_image = raw_path
                if glow_cache_dir is not None and glow_opts.enabled:
                    glow_key = (raw_path, tuple(glow_opts.color), float(glow_opts.width), float(glow_opts.intensity))
                    glow_img = glow_variant_cache.get(glow_key)
                    if glow_img is None:
                        glow_img = ensure_glow_variant_image(raw_path, glow_opts, glow_cache_dir)
                        glow_variant_cache[glow_key] = glow_img
                    base_image = glow_img or raw_path
                image_variant = base_image
                if phase_kind == "impact" and impact_cache_dir is not None:
                    impact_key = (base_image, impact_kind, float(impact_duration))
                    impact_img = impact_variant_cache.get(impact_key)
                    if impact_img is None:
                        impact_img = ensure_impact_variant_image(impact_kind, base_image, impact_cache_dir) or base_image
                        impact_variant_cache[impact_key] = impact_img
                    image_variant = impact_img or base_image
                normalized_path = normalized_cache.get(image_variant)
                if not normalized_path:
                    normalized_path = ensure_normalized_image_cached(
                        image_variant, frame_size[0], frame_size[1], "JPEG", 92
                    )
                    normalized_cache[image_variant] = normalized_path
                clip = _clip_with_duration(ImageClip(normalized_path), dur)
            clips.append(clip)
            cursor = snap_to_frame(cursor + dur, fps, "round")
            if cursor >= window_dur_s - (1.0 / fps):
                break
        processed_segments += 1
        if total_segments and processed_segments % max(1, total_segments // 10 or 1) == 0:
            pct = min(0.3, (cursor / max(window_dur_s or 1.0, 1e-6)) * 0.3)
            progress_cb(pct)
            status_cb(f"Quick: prepping frames {processed_segments}/{total_segments}")
        if cursor >= window_dur_s - (1.0 / fps):
            break

    if not clips:
        raise RuntimeError(
            "Quick window did not overlap any segment. Try a different start or longer duration."
        )

    timeline = concatenate_videoclips(clips, method="chain")
    total_len = timeline.duration

    if wave_clip is not None:
        wave_clip = _clip_with_duration(wave_clip, total_len)
        wave_clip = _clip_with_position(wave_clip, ("center", frame_size[1] - waveform_height))
        timeline = CompositeVideoClip([timeline, wave_clip])

    audio_start = audio_window_start
    audio_end = audio_start + total_len

    base_audio = None
    audio = None
    try:
        base_audio = AudioFileClip(audio_path)
        audio_end = min(audio_end, base_audio.duration)
        _check_cancel("audio prep")
        if audio_end <= audio_start:
            status_cb("Audio window out of range; silent preview.")
            timeline = _clip_with_audio(timeline, None)
        else:
            audio = _audio_subclip(base_audio, audio_start, audio_end)
            if audio_gain_db and abs(audio_gain_db) > 1e-6:
                audio = _audio_with_volume(audio, db_to_gain(audio_gain_db))
            timeline = _clip_with_audio(timeline, audio)

        logger = TkMoviePyLogger(lambda p: progress_cb(0.35 + 0.6 * p), cancel_requested=cancel_requested)
        timeline.write_videofile(
            out_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            preset="faster",
            threads=4,
            temp_audiofile=str(Path(out_path).with_suffix(".temp-audio.m4a")),
            remove_temp=True,
            logger=logger,
        )
    finally:
        if audio:
            audio.close()
        if base_audio:
            base_audio.close()
        timeline.close()
        if wave_clip:
            wave_clip.close()
        for c in clips:
            try:
                c.close()
            except Exception:
                pass


# =========================
# FFmpeg FULL export
# =========================
def _ensure_black_image(path: Path, w: int, h: int, fmt: str, jpeg_quality: int = 90):
    if path.exists():
        return
    img = Image.new("RGB", (w, h), (0, 0, 0))
    if fmt.upper() == "JPEG":
        img.save(path, "JPEG", quality=jpeg_quality, optimize=True, progressive=True)
    else:
        img.save(path, "PNG", optimize=True)


def _concat_min_duration(dur: float, fps: int) -> float:
    return max(1.0 / max(1, fps), float(dur))


def _build_concat_file(txt_path: Path, items: List[Tuple[str, float]], fps: int):
    lines = ["ffconcat version 1.0"]
    for fp, dur in items:
        d = _concat_min_duration(dur, fps)
        safe = fp.replace("\\", "/").replace("'", r"'\\''")
        lines.append(f"file '{safe}'")
        lines.append(f"duration {d:.6f}")
    if items:
        fp_last = items[-1][0]
        safe = fp_last.replace("\\", "/").replace("'", r"'\\''")
        lines.append(f"file '{safe}'")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def _short(p: str, n: int = 64) -> str:
    return p if len(p) <= n else ("…" + p[-(n - 1) :])


def _run_ffprobe_duration(ffprobe: str, media_path: str) -> Optional[float]:
    try:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            media_path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        s = out.decode("utf-8", errors="replace").strip()
        return float(s)
    except Exception as e:
        print("[FFPROBE] failed:", e)
        return None


def _normalize_one(
    src_path: str,
    out_dir: Path,
    W: int,
    H: int,
    save_format: str,
    jpeg_quality: int,
) -> str:
    p = Path(src_path)
    stem = p.stem
    ext = ".jpg" if save_format.upper() == "JPEG" else ".png"
    out_path = out_dir / f"{stem}{ext}"
    i = 1
    while (
        out_path.exists()
        and out_path.stat().st_size > 0
        and (out_path.resolve() != p.resolve())
    ):
        out_path = out_dir / f"{stem}_{i}{ext}"
        i += 1

    try:
        im = Image.open(p)
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (0, 0, 0))
            alpha = im.getchannel("A") if "A" in im.getbands() else None
            bg.paste(im.convert("RGB"), mask=alpha)
            im = bg
        else:
            im = im.convert("RGB")

        iw, ih = im.size
        scale = min(W / iw, H / ih)
        new_w = max(1, int(round(iw * scale)))
        new_h = max(1, int(round(ih * scale)))
        im_resized = im.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        canvas.paste(im_resized, ((W - new_w) // 2, (H - new_h) // 2))
        save_canvas = canvas.convert("RGB")
        if save_format.upper() == "JPEG":
            save_canvas.save(out_path, "JPEG", quality=jpeg_quality, optimize=True, progressive=True)
        else:
            save_canvas.save(out_path, "PNG", optimize=True)
        return str(out_path)
    except Exception as e:
        print(f"[NORMALIZE] Failed on {_short(str(p))}: {e}")
        return ""


def _build_parallel_normalized_cache(
    sources: List[str],
    width: int,
    height: int,
    save_format: str,
    jpeg_quality: int,
    status_cb: Callable[[str], None],
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> Dict[str, str]:
    unique: List[str] = []
    seen: set[str] = set()
    for src in sources:
        if not src:
            continue
        if src not in seen:
            seen.add(src)
            unique.append(src)
    total = len(unique)
    if total == 0:
        return {}
    max_workers = max(1, min(os.cpu_count() or 4, total))
    _log(
        f"[Normalize] starting pool workers={max_workers} unique={total} target={width}x{height} fmt={save_format}"
    )
    progress_step = max(1, total // 10 or 1)
    cache: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                ensure_normalized_image_cached,
                src,
                width,
                height,
                save_format,
                jpeg_quality,
            ): src
            for src in unique
        }
        completed = 0
        for future in as_completed(futures):
            if cancel_requested and cancel_requested():
                raise RuntimeError("Render cancelled (normalize)")
            src = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[NORMALIZE] worker failed on {_short(src)}: {exc}")
                result = ""
            if result:
                cache[src] = result
            completed += 1
            if completed % progress_step == 0 or completed == total:
                status_cb(
                    f"Prepping glow/normalize cache {completed}/{total} (x{max_workers})"
                )
                _log(f"[Normalize] progress {completed}/{total}")
    status_cb("Stage: normalize/glow cache finalizing…")
    _log("[Normalize] all workers done; finalizing cache map")
    return cache


def ensure_normalized_image_cached(
    src_path: str,
    width: int,
    height: int,
    save_format: str = "JPEG",
    jpeg_quality: int = 92,
) -> str:
    if not src_path:
        return ""
    source = Path(src_path)
    if not source.exists():
        return src_path
    try:
        stat = source.stat()
    except OSError:
        stat = None
    cache_dir = _shared_cache_dir("normalized")
    ext = ".jpg" if save_format.upper() == "JPEG" else ".png"
    key = (
        f"{source.resolve()}|{stat.st_mtime_ns if stat else 0}|{stat.st_size if stat else 0}|"
        f"{width}x{height}|{save_format.upper()}|{jpeg_quality}"
    )
    digest = hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()
    target = cache_dir / f"norm_{digest}{ext}"
    if target.exists():
        return str(target)
    tmp_path = _normalize_one(src_path, cache_dir, width, height, save_format, jpeg_quality)
    if not tmp_path:
        return src_path
    tmp = Path(tmp_path)
    if tmp.resolve() == target.resolve():
        return str(target)
    try:
        if not target.exists():
            tmp.replace(target)
    except Exception:
        target = tmp
    return str(target)


def ffmpeg_full_export(
    ffmpeg_path: str,
    ffprobe_path: str,
    out_path: str,
    fps: int,
    frame_size: Tuple[int, int],
    audio_path: str,
    audio_gain_db: float,
    use_nvenc: bool,
    use_shortest: bool,
    pre_normalize: bool,
    pre_fmt: str,
    pre_q: int,
    intervals: List[Tuple[int, int]],
    assignments: List[str],
    first_marker_frames: int,
    tc_base_s: float,
    impact_kind: str,
    impact_duration: float,
    glow_opts: GlowOptions,
    wave_opts: WaveformOptions,
    progress_cb,
    status_cb,
    cancel_requested: Optional[Callable[[], bool]] = None,
):
    def _check_cancel(stage: str = "") -> None:
        if cancel_requested and cancel_requested():
            raise RuntimeError("Render cancelled" + (f" ({stage})" if stage else ""))

    def _apply_glow_cached(path: str) -> str:
        if not glow_cache_dir or not path:
            return path
        cached = glow_variant_map.get(path)
        if cached:
            return cached
        variant = ensure_glow_variant_image(path, glow_opts, glow_cache_dir)
        glow_variant_map[path] = variant
        return variant

    W, H = frame_size
    tmpdir = Path(tempfile.mkdtemp(prefix="imgseq_"))
    concat_txt = tmpdir / "images_concat.txt"
    impact_kind = impact_kind or "disabled"
    impact_duration = max(0.0, float(impact_duration))
    impact_cache_dir = (
        _shared_cache_dir("impact_fx") if _impact_enabled(impact_kind, impact_duration) else None
    )
    glow_cache_dir = _shared_cache_dir("glow_fx") if glow_opts.enabled else None
    glow_variant_map: Dict[str, str] = {}
    waveform_enabled = wave_opts.enabled and wave_opts.style != "disabled"
    waveform_height = max(1, int(H * wave_opts.height_pct)) if waveform_enabled else 0
    waveform_filter = None
    if waveform_enabled:
        waveform_filter = _build_waveform_filter(
            W, waveform_height, fps, wave_opts.style, wave_opts.color, wave_opts.opacity
        )
        print("[Timeline] Export: waveform overlay enabled")

    black_ext = ".jpg" if (pre_normalize and pre_fmt.upper() == "JPEG") else ".png"
    black_path = tmpdir / f"black{black_ext}"
    _ensure_black_image(black_path, W, H, ("JPEG" if black_ext == ".jpg" else "PNG"), jpeg_quality=pre_q)

    print("=== FULL EXPORT DIAGNOSTICS ===")
    print(
        f"[CFG] FFmpeg='{ffmpeg_path}' | FFprobe='{ffprobe_path}' | NVENC={use_nvenc} | FPS={fps} | Size={W}x{H}"
    )
    print(f"[PATHS] audio={_short(str(audio_path))}")
    print(f"[COUNTS] intervals={len(intervals)} | assignments={len(assignments)}")

    non_empty = sum(1 for a in assignments if a)
    missing = [a for a in assignments if a and not Path(a).exists()]
    print(f"[ASSIGN] non-empty={non_empty} | missing={len(missing)}")
    if missing[:5]:
        print("[ASSIGN] missing samples:", ", ".join(_short(m, 48) for m in missing[:5]))

    if intervals:
        d_frames = [(b - a) for (a, b) in intervals if b > a]
        d_secs = [df / float(fps) for df in d_frames]
        print(
            f"[DURS] frames: min={min(d_frames)} | med={int(statistics.median(d_frames))} | max={max(d_frames)}"
        )
        print(
            f"[DURS] secs:   min={min(d_secs):.6f} | med={statistics.median(d_secs):.6f} | max={max(d_secs):.6f}"
        )

    base_frames = int(round(max(0.0, tc_base_s) * fps))
    head_frames = max(0, first_marker_frames - base_frames)
    head_dur = head_frames / float(fps)
    print(
        f"[TIMECODE] base_s={tc_base_s:.3f} ({base_frames}f) | first_marker={first_marker_frames}f "
        f"({seconds_to_tc(frames_to_seconds(first_marker_frames, fps), fps)}) | black_head={head_dur:.6f}s"
    )
    if head_dur > 30:
        status_cb(f"Note: long black head {head_dur:.2f}s (check TC base).")

    status_cb("Stage: normalize/glow cache")
    progress_cb(0.05)
    items: List[Tuple[str, float]] = []
    if head_frames > 0:
        items.append((str(black_path), head_dur))

    preprocess_start = time.time()
    if pre_normalize:
        _check_cancel("normalize")
        normalized_root = _shared_cache_dir("normalized")
        sources_for_norm = [
            assignments[i]
            for i in range(len(assignments))
            if i < len(assignments) and assignments[i] and Path(assignments[i]).exists()
        ]
        normalized_map = _build_parallel_normalized_cache(
            sources_for_norm,
            W,
            H,
            pre_fmt,
            pre_q,
            status_cb,
            cancel_requested,
        )
        if glow_cache_dir:
            glow_variant_map.update(
                _precache_glow_variants(
                    normalized_map.values(),
                    glow_opts,
                    glow_cache_dir,
                    status_cb,
                    cancel_requested,
                    label="Stage: glow cache (normalized)",
                )
            )
        normalized_count = len(normalized_map)
        mismatched = sum(
            1
            for src in normalized_map
            if (
                (Path(src).suffix.lower() == ".png" and pre_fmt.upper() == "JPEG")
                or (Path(src).suffix.lower() in {".jpg", ".jpeg"} and pre_fmt.upper() == "PNG")
            )
        )
        finalize_step = 1
        for i, (a, b) in enumerate(intervals):
            _check_cancel("segment prep")
            dur = max(0.0, (b - a) / float(fps))
            src = assignments[i] if (i < len(assignments) and assignments[i]) else ""
            if not src:
                items.append((str(black_path), dur))
                continue
            norm_path = normalized_map.get(src)
            if not norm_path:
                items.append((str(black_path), dur))
                continue
            final_path = norm_path
            if glow_cache_dir:
                final_path = _apply_glow_cached(final_path)
            base_dur, tail_dur = split_duration_for_impact(dur, True, impact_kind, impact_duration)
            if base_dur > IMPACT_EPS:
                items.append((final_path, base_dur))
            if tail_dur > IMPACT_EPS and impact_cache_dir is not None:
                variant_path = ensure_impact_variant_image(impact_kind, final_path, impact_cache_dir) or final_path
                items.append((variant_path, tail_dur))
            if (i + 1) % finalize_step == 0 or (i + 1) == len(intervals):
                pct = min(1.0, (i + 1) / max(1, len(intervals)))
                status_cb(f"Stage: normalize/glow finalize {i+1}/{len(intervals)}")
                _log(
                    f"[Normalize] finalize progress {i+1}/{len(intervals)} ({pct*100:.1f}%)"
                )
        elapsed = time.time() - preprocess_start
        print(
            f"[NORMALIZE] wrote={normalized_count} files to '{normalized_root}' (fmt={pre_fmt}, q={pre_q}) | mismatched_ext≈{mismatched} | elapsed={elapsed:.1f}s"
        )
        status_cb(f"Stage: normalize/glow complete ({elapsed:.1f}s)")
        progress_cb(0.25)
        vf = f"fps={fps},setsar=1"
    else:
        if glow_cache_dir:
            glow_variant_map.update(
                _precache_glow_variants(
                    assignments,
                    glow_opts,
                    glow_cache_dir,
                    status_cb,
                    cancel_requested,
                    label="Stage: glow cache",
                )
            )
        for i, (a, b) in enumerate(intervals):
            dur = max(0.0, (b - a) / float(fps))
            img = assignments[i] if (i < len(assignments) and assignments[i]) else str(black_path)
            if not Path(img).exists():
                img = str(black_path)
            has_image = img != str(black_path)
            if has_image and glow_cache_dir:
                img = _apply_glow_cached(img)
            base_dur, tail_dur = split_duration_for_impact(dur, has_image, impact_kind, impact_duration)
            if base_dur > IMPACT_EPS:
                items.append((img, base_dur))
            if tail_dur > IMPACT_EPS and has_image and impact_cache_dir is not None:
                variant_path = ensure_impact_variant_image(impact_kind, img, impact_cache_dir) or img
                items.append((variant_path, tail_dur))
        vf = (
            f"scale=w={W}:h={H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={fps},setsar=1"
        )

    est_total = sum(max(1.0 / fps, d) for _, d in items)
    print(f"[BUILD] concat items={len(items)} | est_total_video={est_total:.6f}s")
    if items[:5]:
        print("[BUILD] first items:")
        for fp_, d in items[:5]:
            print("  -", _short(Path(fp_).name, 40), f"{d:.6f}s")
    if len(items) > 5:
        tail_sample = " | ".join(f"{_short(Path(fp).name, 24)} {d:.3f}s" for fp, d in items[-3:])
        print(f"[BUILD] last items: {tail_sample}")

    _build_concat_file(concat_txt, items, fps)
    txt_lines = concat_txt.read_text(encoding="utf-8").splitlines()
    status_cb("Stage: building concat list")
    print(f"[CONCAT] path={concat_txt}")
    print("[CONCAT] head:")
    for ln in txt_lines[:10]:
        print(" ", ln)
    if len(txt_lines) > 20:
        print("[CONCAT] tail:")
        for ln in txt_lines[-10:]:
            print(" ", ln)

    a_dur = _run_ffprobe_duration(ffprobe_path, audio_path)
    if a_dur is not None:
        print(f"[FFPROBE] audio duration ≈ {a_dur:.3f}s")

    af = []
    if abs(audio_gain_db) > 1e-6:
        af.append(f"volume={10**(audio_gain_db/20.0):.6f}")
    af_filter = ",".join(af) if af else "anull"

    if use_nvenc:
        vcodec = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "19"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "faster"]

    status_cb("Stage: launching FFmpeg")
    progress_cb(0.3)
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-y",
        "-fflags",
        "+genpts",
        "-safe",
        "0",
        "-f",
        "concat",
        "-i",
        str(concat_txt),
        "-i",
        audio_path,
    ]

    if waveform_enabled and waveform_filter:
        overlay_y = max(0, H - waveform_height)
        filter_complex = ";".join(
            [
                f"[0:v]{vf}[base]",
                f"[1:a]{waveform_filter}[wave]",
                f"[base][wave]overlay=0:{overlay_y}:format=auto[outv]",
            ]
        )
        print(f"[Timeline] Export filter_complex: {filter_complex}")
        cmd += ["-filter_complex", filter_complex, "-map", "[outv]", "-map", "1:a"]
    else:
        cmd += ["-vf", vf]

    cmd += [
        "-filter:a",
        af_filter,
        "-r",
        str(fps),
        *vcodec,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]
    if use_shortest:
        cmd += ["-shortest"]
    cmd += ["-progress", "pipe:1", "-nostats", out_path]

    print("[CMD]", " ".join(cmd))
    print("=== FFmpeg progress === (watch for warnings/errors below)")

    total_video_sec = est_total
    time_re = re.compile(rb"out_time_ms=(\d+)")
    warn_re = re.compile(r"(i)(error|invalid|no such|failed|decode|corrupt)")
    buf_lines: List[str] = []
    warn_lines: List[str] = []

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    status_cb("Stage: ffmpeg encoding")
    progress_cb(0.31)
    start_ff = time.time()
    try:
        for raw in iter(proc.stdout.readline, b""):
            if cancel_requested and cancel_requested():
                status_cb("Stopping FFmpeg…")
                proc.terminate()
                proc.wait(timeout=5)
                raise RuntimeError("Render cancelled")
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            buf_lines.append(line)
            if warn_re.search(line):
                warn_lines.append(line)
            m = time_re.search(raw)
            if m:
                ms = int(m.group(1))
                t = ms / 1_000_000.0
                if total_video_sec > 0:
                    pct = min(1.0, t / total_video_sec)
                    progress_cb(pct)
                    status_cb(f"Stage: ffmpeg encoding {pct*100:.1f}% ({seconds_to_tc(t, fps)})")
    finally:
        proc.wait()
        elapsed_ff = time.time() - start_ff
        progress_cb(1.0)

        if warn_lines:
            print("=== FFmpeg warnings/errors (non-fatal) ===")
            for ln in warn_lines[-80:]:
                print(ln)

        out_dur = _run_ffprobe_duration(ffprobe_path, out_path)
        if out_dur is not None:
            print(
                f"[FFPROBE] output duration ≈ {out_dur:.3f}s (expected ≈ {est_total:.3f}s)"
            )

        if proc.returncode != 0:
            tail = "\n".join(buf_lines[-100:])
            print("=== FFmpeg log (tail) ===\n", tail)
            raise RuntimeError(f"FFmpeg returned {proc.returncode}. See console for the tail log.")
        status_cb(f"Stage: ffmpeg finished ({elapsed_ff:.1f}s)")
        print("=== FULL EXPORT DONE ===")
        print("Output:", out_path)
        print("=========================")


@dataclass
class GlowOptions:
    enabled: bool = False
    color: Tuple[int, int, int] = (0, 255, 255)
    width: int = 6
    intensity: float = 0.6
    preset: str = "legacy"


@dataclass
class WaveformOptions:
    enabled: bool = False
    color: Tuple[int, int, int] = (0, 255, 255)
    opacity: float = 0.6
    height_pct: float = 0.18
    style: str = "oscilloscope"


@dataclass
class Segment:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    duration: float = 1.0
    kind: str = "image"
    image_path: str = ""
    start: float = 0.0
    end: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "duration": self.duration,
            "kind": self.kind,
            "image_path": self.image_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Segment":
        seg = cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            name=str(data.get("name", "")),
            duration=float(data.get("duration", 1.0)),
            kind=str(data.get("kind", "image")),
            image_path=str(data.get("image_path") or ""),
        )
        return seg

    def ensure_min_duration(self, min_duration: float) -> None:
        self.duration = max(min_duration, float(self.duration))


def timeline_to_render_segments(segments: List[Segment]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for seg in segments:
        kind = "image"
        img = seg.image_path
        if seg.kind != "image" or not img:
            kind = "black"
            img = ""
        out.append(
            {
                "kind": kind,
                "start": seg.start,
                "end": seg.end,
                "img": img,
                "name": seg.name,
            }
        )
    return out


def timeline_to_intervals(
    segments: List[Segment], fps: int
) -> Tuple[List[Tuple[int, int]], List[str]]:
    intervals: List[Tuple[int, int]] = []
    assignments: List[str] = []
    current = 0
    for seg in segments:
        frames = max(1, int(round(seg.duration * fps)))
        intervals.append((current, current + frames))
        assignments.append(seg.image_path if seg.kind == "image" else "")
        current += frames
    return intervals, assignments


class TkMoviePyLogger(ProgressBarLogger):
    def __init__(self, on_update, cancel_requested: Optional[Callable[[], bool]] = None):
        super().__init__()
        self.on_update = on_update
        self._cancel_requested = cancel_requested

    def bars_callback(self, bar, attr, value, old_value=None):
        if self._cancel_requested and self._cancel_requested():
            raise RuntimeError("Render cancelled")
        try:
            total = self.bars[bar]["total"] or 1
            index = self.bars[bar]["index"] or 0
            self.on_update(index / total)
        except Exception:
            pass


class RenderWorker(QtCore.QObject):
    progress = Signal(float)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._cancelled = False

    @QtCore.Slot()
    def run(self):
        try:
            kwargs = dict(self.kwargs)
            kwargs.setdefault("cancel_requested", self.is_cancelled)
            self.func(
                *self.args,
                progress_cb=self.progress.emit,
                status_cb=self.status.emit,
                **kwargs,
            )
        except Exception as exc:  # noqa: broad-except
            self.error.emit(str(exc))
        else:
            self.finished.emit()

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled or QtCore.QThread.currentThread().isInterruptionRequested()


class FolderListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder_images: Dict[str, List[str]] = {}
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)

    def set_folder_images(self, mapping: Dict[str, List[Path]]) -> None:
        normalized: Dict[str, List[str]] = {}
        for folder, paths in mapping.items():
            normalized[str(folder)] = [str(p) for p in paths]
        self._folder_images = normalized

    def mimeTypes(self):
        types = super().mimeTypes()
        if IMAGE_BUNDLE_MIME not in types:
            types.append(IMAGE_BUNDLE_MIME)
        return types

    def mimeData(self, items):  # noqa: D401 - Qt override
        mime = QtCore.QMimeData()
        if not items:
            return mime
        folder = items[0].data(Qt.UserRole) or items[0].text()
        mime.setText(str(folder))
        bundle = self._folder_images.get(str(folder), [])
        if bundle:
            payload = json.dumps({"folder": folder, "paths": bundle}).encode("utf-8")
            mime.setData(IMAGE_BUNDLE_MIME, payload)
        return mime


class ThumbnailListWidget(QtWidgets.QListWidget):
    imageActivated = Signal(str)
    imageDoubleClicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suppress_scroll_depth = 0
        self._bulk_active = False
        self._bulk_resize_mode: Optional[QtWidgets.QListView.ResizeMode] = None
        self._bulk_scroll_snapshot: Optional[Tuple[int, int]] = None
        self._user_scrolling = False
        self._frozen_scroll: Optional[Tuple[int, int]] = None
        self._freeze_timer = QtCore.QTimer(self)
        self._freeze_timer.setSingleShot(True)
        self._freeze_timer.timeout.connect(self._apply_frozen_scroll)
        self._scrollbar_hooks_installed = False

        self.setViewMode(QtWidgets.QListView.IconMode)
        self.setMovement(QtWidgets.QListView.Static)
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setIconSize(QSize(120, 90))
        self.setGridSize(QSize(140, 140))
        self.setSpacing(8)
        self.setWordWrap(True)
        self.setUniformItemSizes(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setLayoutMode(QtWidgets.QListView.Batched)
        self.setBatchSize(120)
        self.setAutoScroll(False)

        model = self.model()
        if model is not None:
            try:
                model.rowsInserted.connect(self._on_rows_inserted)
            except Exception:
                pass

        self.itemSelectionChanged.connect(self._emit_selected)
        self.itemDoubleClicked.connect(self._emit_double_clicked)
        self.verticalScrollBar().valueChanged.connect(self._log_vscroll_change)
        self.horizontalScrollBar().valueChanged.connect(self._log_hscroll_change)
        self._connect_scrollbar_user_signals()

    def add_thumbnail_item(self, path: Path) -> QtWidgets.QListWidgetItem:
        item = QtWidgets.QListWidgetItem(Path(path).name)
        item.setToolTip(str(path))
        item.setData(Qt.UserRole, str(path))
        exists = Path(path).exists()
        placeholder = "Missing" if not exists else "No\nImg"
        item.setIcon(QtGui.QIcon(get_thumbnail(str(path), 120, 90, placeholder)))
        self.addItem(item)
        return item

    def _index_visible(self, index: QtCore.QModelIndex) -> bool:
        if not index.isValid():
            return False
        rect = self.visualRect(index)
        return rect.isValid() and self.viewport().rect().intersects(rect)

    def mimeTypes(self):
        types = super().mimeTypes()
        if IMAGE_BUNDLE_MIME not in types:
            types.append(IMAGE_BUNDLE_MIME)
        return types

    def mimeData(self, items):  # noqa: D401 - Qt override
        mime = QtCore.QMimeData()
        paths = [it.data(Qt.UserRole) for it in items if it and it.data(Qt.UserRole)]
        if not paths:
            return mime
        if len(paths) == 1:
            path = paths[0]
            mime.setUrls([QtCore.QUrl.fromLocalFile(path)])
            mime.setText(path)
        else:
            payload = json.dumps({"paths": paths}).encode("utf-8")
            mime.setData(IMAGE_BUNDLE_MIME, payload)
            mime.setText("\n".join(paths))
        return mime

    def _emit_selected(self) -> None:
        item = self.currentItem()
        if item:
            self.imageActivated.emit(item.data(Qt.UserRole))

    def _emit_double_clicked(self, item) -> None:
        if item:
            self.imageDoubleClicked.emit(item.data(Qt.UserRole))

    def scrollTo(self, index, hint=QtWidgets.QAbstractItemView.EnsureVisible) -> None:  # noqa: D401
        try:
            item = self.itemFromIndex(index)
            path = item.data(Qt.UserRole) if item else "<none>"
        except Exception:
            path = "<error>"
        _log(
            f"[LibraryScroll] scrollTo widget={self.objectName() or '<unnamed>'} path={path} hint={hint}"
        )
        if self._suppress_scroll_depth > 0:
            _log(
                f"[LibraryScroll] scrollTo suppressed widget={self.objectName() or '<unnamed>'} path={path}"
            )
            return
        if hint == QtWidgets.QAbstractItemView.PositionAtTop:
            if not self._user_scrolling:
                _log("[LibraryScroll] ignore PositionAtTop (programmatic)")
                return
            if self._index_visible(index):
                _log("[LibraryScroll] ignore PositionAtTop (already visible)")
                return
        super().scrollTo(index, hint)

    def _snapshot_scroll(self) -> Tuple[int, int]:
        return (
            self.verticalScrollBar().value(),
            self.horizontalScrollBar().value(),
        )

    def _apply_frozen_scroll(self) -> None:
        if self._frozen_scroll is None:
            return
        v_val, h_val = self._frozen_scroll
        self.verticalScrollBar().setValue(v_val)
        self.horizontalScrollBar().setValue(h_val)
        _log(
            f"[LibraryScroll] reapplied frozen widget={self.objectName() or '<unnamed>'} v={v_val} h={h_val}"
        )
        self._frozen_scroll = None

    def _freeze_scroll_next_tick(self) -> None:
        self._frozen_scroll = self._snapshot_scroll()
        self._freeze_timer.start(0)

    def _on_rows_inserted(self, parent: QtCore.QModelIndex, start: int, end: int) -> None:  # noqa: D401
        if self._bulk_active or self._suppress_scroll_depth > 0:
            self._freeze_scroll_next_tick()

    def push_scroll_suppression(self) -> None:
        self._suppress_scroll_depth += 1
        _log(
            f"[LibraryScroll] suppress++ widget={self.objectName() or '<unnamed>'} depth={self._suppress_scroll_depth}"
        )

    def pop_scroll_suppression(self) -> None:
        self._suppress_scroll_depth = max(0, self._suppress_scroll_depth - 1)
        _log(
            f"[LibraryScroll] suppress-- widget={self.objectName() or '<unnamed>'} depth={self._suppress_scroll_depth}"
        )

    def begin_bulk_insert(self) -> None:
        if self._bulk_active:
            return
        self._bulk_active = True
        self._bulk_resize_mode = self.resizeMode()
        self._bulk_scroll_snapshot = (
            self.verticalScrollBar().value(),
            self.horizontalScrollBar().value(),
        )
        self._freeze_scroll_next_tick()
        self._connect_scrollbar_user_signals()
        self.push_scroll_suppression()
        self.setUpdatesEnabled(False)
        self.setResizeMode(QtWidgets.QListView.Fixed)

    def end_bulk_insert(self) -> None:
        if not self._bulk_active:
            return
        snapshot = self._bulk_scroll_snapshot
        self._bulk_scroll_snapshot = None
        try:
            self.doItemsLayout()
        finally:
            self.setUpdatesEnabled(True)
            if self._bulk_resize_mode is not None:
                self.setResizeMode(self._bulk_resize_mode)
            self._bulk_active = False
            self.pop_scroll_suppression()
            if snapshot:
                v_val, h_val = snapshot
                self.verticalScrollBar().setValue(v_val)
                self.horizontalScrollBar().setValue(h_val)
                QtCore.QTimer.singleShot(0, self._apply_frozen_scroll)
            else:
                self._apply_frozen_scroll()

    def _log_vscroll_change(self, value: int) -> None:
        _log(f"[LibraryScroll] value widget={self.objectName() or '<unnamed>'} v={value}")

    def _log_hscroll_change(self, value: int) -> None:
        _log(f"[LibraryScroll] value widget={self.objectName() or '<unnamed>'} h={value}")

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: D401
        self._user_scrolling = True
        super().wheelEvent(event)
        QtCore.QTimer.singleShot(0, self._reset_user_scroll_flag)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: D401
        navigation_keys = {
            Qt.Key_Up,
            Qt.Key_Down,
            Qt.Key_PageUp,
            Qt.Key_PageDown,
            Qt.Key_Home,
            Qt.Key_End,
        }
        if event.key() in navigation_keys:
            self._user_scrolling = True
            super().keyPressEvent(event)
            QtCore.QTimer.singleShot(0, self._reset_user_scroll_flag)
            return
        super().keyPressEvent(event)

    def _reset_user_scroll_flag(self) -> None:
        self._user_scrolling = False

    def _on_scrollbar_pressed(self) -> None:
        self._user_scrolling = True

    def _on_scrollbar_released(self) -> None:
        QtCore.QTimer.singleShot(0, self._reset_user_scroll_flag)

    def _connect_scrollbar_user_signals(self) -> None:
        if self._scrollbar_hooks_installed:
            return
        for scrollbar in (self.verticalScrollBar(), self.horizontalScrollBar()):
            if scrollbar is None:
                continue
            scrollbar.sliderPressed.connect(self._on_scrollbar_pressed)
            scrollbar.sliderReleased.connect(self._on_scrollbar_released)
        self._scrollbar_hooks_installed = True


class ImageLibraryWidget(QtWidgets.QTabWidget):
    imageActivated = Signal(str)
    imageDoubleClicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lists: Dict[str, ThumbnailListWidget] = {}
        self._path_map: Dict[str, Tuple[ThumbnailListWidget, QtWidgets.QListWidgetItem]] = {}
        self._pending_batches: List[Tuple[ThumbnailListWidget, List[str]]] = []
        self._desired_tab_index = 0
        self._block_selection = False
        self._pending_restore_path = ''
        self._last_widget: Optional[QtWidgets.QWidget] = None

        self._loader_timer = QtCore.QTimer(self)
        self._loader_timer.setInterval(0)
        self._loader_timer.timeout.connect(self._process_thumbnail_queue)
        self.currentChanged.connect(self._handle_tab_changed)

    def _create_list_widget(self, folder: str) -> ThumbnailListWidget:
        widget = ThumbnailListWidget()
        widget.setObjectName(f"Library[{len(self._lists)}]")
        widget.imageActivated.connect(self._relay_activated)
        widget.imageDoubleClicked.connect(self._relay_double_clicked)
        widget.setProperty('folder_path', folder)
        return widget

    def set_images(self, images_by_dir: Dict[str, List[Path]]) -> None:
        images_by_dir = images_by_dir or {}
        current_widget = self.currentWidget()
        if isinstance(current_widget, QtWidgets.QListWidget):
            self._remember_scroll(current_widget)
        current_folder = ''
        if isinstance(current_widget, ThumbnailListWidget):
            current_folder = str(current_widget.property('folder_path') or '')
        previous_path = self.current_image_path()
        self._pending_restore_path = previous_path or ''
        self._loader_timer.stop()
        self._pending_batches.clear()
        self._path_map.clear()

        existing = list(self._lists.keys())
        for key in existing:
            if key not in images_by_dir:
                widget = self._lists.pop(key)
                idx = self.indexOf(widget)
                if idx >= 0:
                    self.removeTab(idx)
                widget.deleteLater()

        for folder, paths in images_by_dir.items():
            list_widget = self._lists.get(folder)
            if list_widget is None:
                list_widget = self._create_list_widget(folder)
                self._lists[folder] = list_widget
                idx = self.addTab(list_widget, _friendly_folder_name(folder) or 'Folder')
                self.setTabToolTip(idx, folder)
            else:
                self._remember_scroll(list_widget)
                list_widget.clear()
                idx = self.indexOf(list_widget)
                if idx >= 0:
                    self.setTabText(idx, _friendly_folder_name(folder) or 'Folder')
                    self.setTabToolTip(idx, folder)
            list_widget.setProperty('folder_path', folder)
            queue = [str(Path(p)) for p in paths]
            self._pending_batches.append((list_widget, queue))

        if current_folder:
            for idx in range(self.count()):
                widget = self.widget(idx)
                if isinstance(widget, ThumbnailListWidget) and str(widget.property('folder_path') or '') == current_folder:
                    self._desired_tab_index = idx
                    break
        else:
            self._desired_tab_index = min(self._desired_tab_index, max(0, self.count() - 1)) if self.count() else 0

        if self._pending_batches:
            self._loader_timer.start()
        else:
            self._finish_images_population()

    def _process_thumbnail_queue(self) -> None:
        if not self._pending_batches:
            self._loader_timer.stop()
            self._finish_images_population()
            return
        batch = 24
        while batch > 0 and self._pending_batches:
            widget, queue = self._pending_batches[0]
            if not queue:
                widget.end_bulk_insert()
                self._pending_batches.pop(0)
                continue
            widget.begin_bulk_insert()
            path = queue.pop(0)
            item = widget.add_thumbnail_item(Path(path))
            self._path_map[str(path)] = (widget, item)
            batch -= 1
        if not self._pending_batches:
            self._loader_timer.stop()
            self._finish_images_population()

    def _finish_images_population(self) -> None:
        restore_path = self._pending_restore_path
        self._pending_restore_path = ''
        if restore_path:
            restored = self.select_image(restore_path, center=False)
            if restored:
                return
        if self.count():
            idx = max(0, min(self._desired_tab_index, self.count() - 1))
            self.setCurrentIndex(idx)

    def _handle_tab_changed(self, index: int) -> None:
        if self._last_widget and isinstance(self._last_widget, QtWidgets.QListWidget):
            self._remember_scroll(self._last_widget)
        widget = self.widget(index)
        if isinstance(widget, QtWidgets.QListWidget):
            self._restore_scroll(widget)
        self._last_widget = widget
        if index >= 0:
            self._desired_tab_index = index

    def _relay_activated(self, path: str) -> None:
        if not self._block_selection:
            self.imageActivated.emit(path)

    def _relay_double_clicked(self, path: str) -> None:
        self.imageDoubleClicked.emit(path)

    def current_image_path(self) -> str:
        widget = self.currentWidget()
        if isinstance(widget, ThumbnailListWidget):
            selected = widget.selectedItems()
            if selected:
                return selected[-1].data(Qt.UserRole)
            item = widget.currentItem()
            if item:
                return item.data(Qt.UserRole)
        return ''

    def _remember_scroll(self, widget: QtWidgets.QListWidget) -> None:
        if not isinstance(widget, QtWidgets.QListWidget):
            return
        v_val = widget.verticalScrollBar().value()
        h_val = widget.horizontalScrollBar().value()
        widget.setProperty('_last_scroll', (v_val, h_val))
        _log(
            f"[LibraryScroll] remember widget={widget.objectName() or '<unnamed>'} v={v_val} h={h_val}"
        )

    def _restore_scroll(self, widget: QtWidgets.QListWidget) -> None:
        if not isinstance(widget, QtWidgets.QListWidget):
            return
        last = widget.property('_last_scroll')
        if not last or not isinstance(last, tuple) or len(last) != 2:
            return
        v_val, h_val = last
        widget.verticalScrollBar().setValue(v_val)
        widget.horizontalScrollBar().setValue(h_val)
        _log(
            f"[LibraryScroll] restore widget={widget.objectName() or '<unnamed>'} v={v_val} h={h_val}"
        )

    def select_image(self, path: str, *, center: bool = True) -> bool:
        if not path:
            return False
        entry = self._path_map.get(path)
        if not entry and _is_running_in_wsl():
            alt = str(_windows_to_wsl_path(path))
            entry = self._path_map.get(alt)
        if not entry:
            return False
        list_widget, item = entry
        index = self.indexOf(list_widget)
        if index == -1:
            return False
        suppress_widget = isinstance(list_widget, ThumbnailListWidget)
        same_tab = self.currentWidget() is list_widget
        already_selected = (
            same_tab and list_widget.currentItem() is item and item.isSelected()
        )
        if not center and already_selected:
            _log("[LibrarySelect] skip – already selected and no centering requested")
            return True
        self._block_selection = True
        if suppress_widget and not center:
            list_widget.push_scroll_suppression()
        try:
            if not same_tab:
                self.setCurrentIndex(index)
            if list_widget.currentItem() is not item:
                list_widget.setCurrentItem(item)
            else:
                # Ensure selection state matches request
                item.setSelected(True)
        finally:
            if suppress_widget and not center:
                QtCore.QTimer.singleShot(0, list_widget.pop_scroll_suppression)
            self._block_selection = False
        if center:
            list_widget.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
        return True


class ImagePreviewDialog(QtWidgets.QDialog):
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(Path(image_path).name)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._pixmap = QtGui.QPixmap(image_path)
        self._first_show = True
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QtWidgets.QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._label)
        if self._pixmap.isNull():
            self._label.setText(f"Failed to load image:\n{image_path}")
        else:
            self._label.setPixmap(self._pixmap)
        self.installEventFilter(self)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: D401
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: D401
        super().showEvent(event)
        if self._first_show:
            self._first_show = False
            self.showMaximized()
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap.isNull():
            return
        target = self.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._label.setPixmap(scaled)


class ImageScanWorker(QtCore.QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, directories: List[Tuple[str, str]], allowed_exts: Iterable[str]):
        super().__init__()
        self.directories = directories
        self.allowed_exts = list(allowed_exts)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result: Dict[str, List[str]] = {}
            for original, resolved in self.directories:
                if not resolved:
                    continue
                path = Path(resolved)
                if not path.exists():
                    continue
                found = find_images_recursive(str(path), self.allowed_exts)
                if found:
                    result[original] = [str(p) for p in found]
            self.finished.emit(result)
        except Exception as exc:  # noqa: broad-except
            self.error.emit(str(exc))
class TimelineListWidget(QtWidgets.QListWidget):
    imageDropped = Signal(str, str)
    bundleDropped = Signal(list)
    orderChanged = Signal()
    segmentActivated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QtWidgets.QListView.IconMode)
        self.setMovement(QtWidgets.QListView.Snap)
        self.setWrapping(False)
        self.setSpacing(8)
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setIconSize(QSize(160, 100))
        self.setWordWrap(True)
        self.setFlow(QtWidgets.QListView.LeftToRight)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setLayoutMode(QtWidgets.QListView.Batched)
        self.setBatchSize(120)
        self.setUniformItemSizes(True)
        self.itemDoubleClicked.connect(self._handle_double_click)

    def _handle_double_click(self, item):
        seg_id = item.data(Qt.UserRole) if item else None
        if seg_id:
            self.segmentActivated.emit(seg_id)

    def dragEnterEvent(self, event):  # noqa: D401 - Qt override
        if (
            event.mimeData().hasUrls()
            or event.mimeData().hasFormat(IMAGE_BUNDLE_MIME)
            or event.source() == self
        ):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):  # noqa: D401 - Qt override
        if (
            event.mimeData().hasUrls()
            or event.mimeData().hasFormat(IMAGE_BUNDLE_MIME)
            or event.source() == self
        ):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):  # noqa: D401 - Qt override
        if event.mimeData().hasFormat(IMAGE_BUNDLE_MIME):
            try:
                data = bytes(event.mimeData().data(IMAGE_BUNDLE_MIME)).decode("utf-8")
                payload = json.loads(data)
                paths = payload.get("paths", [])
            except Exception:
                paths = []
            if paths:
                self.bundleDropped.emit([str(p) for p in paths if p])
                event.acceptProposedAction()
                return
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                item = self.itemAt(pos)
                if item is None and self.count():
                    item = self.item(self.count() - 1)
                if item:
                    seg_id = item.data(Qt.UserRole)
                    self.imageDropped.emit(seg_id, path)
                    event.acceptProposedAction()
                    return
        super().dropEvent(event)
        self.orderChanged.emit()


class SegmentItem(QtWidgets.QListWidgetItem):
    def __init__(self, segment: Segment, fps: int):
        super().__init__()
        self.update_from_segment(segment, fps)

    def update_from_segment(self, segment: Segment, fps: int) -> None:
        label = segment.name or "Segment"
        tc_start = seconds_to_tc(segment.start, fps)
        tc_end = seconds_to_tc(segment.end, fps)
        subtitle = f"{segment.duration:.3f}s ({tc_start} → {tc_end})"
        detail = ""
        if segment.kind == "black":
            thumb = get_thumbnail("", 160, 100, "Black")
        else:
            resolved_path = segment.image_path or ""
            exists = False
            if segment.image_path:
                candidate = Path(segment.image_path)
                if not candidate.exists() and _is_running_in_wsl() and ":" in segment.image_path:
                    candidate = _windows_to_wsl_path(segment.image_path)
                exists = candidate.exists()
                if exists:
                    resolved_path = str(candidate)
            placeholder = "Missing" if segment.image_path and not exists else "No\nImg"
            thumb = get_thumbnail(resolved_path, 160, 100, placeholder)
        self.setIcon(QtGui.QIcon(thumb))
        if segment.kind == "black":
            detail = "[Black]"
            base_color = QtGui.QColor("#444444")
        else:
            if segment.image_path:
                filename = Path(segment.image_path).name
                detail = filename if exists else f"{filename} (missing)"
                base_color = QtGui.QColor("#2d8f6f" if exists else "#a94442")
            else:
                detail = "[No image]"
                base_color = QtGui.QColor("#8c6b1f")
        self.setText("\n".join([label, subtitle, detail]))
        self.setData(Qt.UserRole, segment.id)
        self.setToolTip(
            f"{label}\nStart: {tc_start}\nEnd: {tc_end}\nDuration: {segment.duration:.3f}s\nImage: {segment.image_path or 'None'}"
        )
        self.setBackground(QtGui.QBrush(base_color.lighter(130)))
        width = max(120, int(segment.duration * PIXELS_PER_SECOND))
        self.setSizeHint(QSize(width, 90))


class TimelineApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg: Dict[str, object] = DEFAULTS.copy()
        self.marker_frames: List[int] = []
        self.marker_frames_raw: List[int] = []
        self.head_frames: int = 0
        self.segments: List[Segment] = []
        self.image_dirs: List[str] = list(DEFAULTS.get("IMAGES_DIRS", []))
        self.images: List[Path] = []
        self.images_by_dir: Dict[str, List[Path]] = {}
        self.current_project_path: Optional[str] = None
        self.worker_thread: Optional[QtCore.QThread] = None
        self.worker: Optional[RenderWorker] = None
        self._busy = False
        self._loading_segment_ui = False
        self._image_scan_thread: Optional[QtCore.QThread] = None
        self._image_scan_worker: Optional[ImageScanWorker] = None
        self._pending_image_scan = False
        self._suppress_library_activation = False
        self._suppress_library_sync = False
        self._library_recent_click = False
        self._library_recent_click_path = ""
        self._library_scroll_guard = QtCore.QTimer(self)
        self._library_scroll_guard.setSingleShot(True)
        self._library_scroll_guard.timeout.connect(self._end_library_click_sync)
        self._cursor_guard = QtCore.QTimer(self)
        self._cursor_guard.setInterval(1500)
        self._cursor_guard.timeout.connect(self._guard_system_cursor)
        self._main_splitter: Optional[QtWidgets.QSplitter] = None
        self._timeline_splitter: Optional[QtWidgets.QSplitter] = None
        self._nvenc_cache: Dict[str, bool] = {}

        self._build_ui()
        self._apply_config(self.cfg)
        self._update_image_dirs_list()
        self.rescan_images()
        sample_csv = str(self.cfg.get("MARKERS_CSV") or "")
        loaded_sample = False
        if sample_csv:
            candidate = Path(sample_csv)
            if not candidate.exists() and _is_running_in_wsl():
                candidate = _windows_to_wsl_path(sample_csv)
            if candidate.exists():
                self.csv_edit.setText(str(candidate))
                self.load_markers_from_csv(str(candidate))
                loaded_sample = True
        else:
            self._refresh_timeline(preserve_selection=False)
        if loaded_sample:
            self._set_status("Loaded sample demo project.")
        else:
            self._set_status("Ready.")

    # --- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("Beat Image Timeline Builder — Timeline Edition")
        self.resize(1480, 900)
        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self._create_central_layout()
        self._create_status_bar()
        self._connect_ui()

    def _create_actions(self) -> None:
        self.new_project_action = QtGui.QAction("New Project", self)
        self.open_project_action = QtGui.QAction("Open Project…", self)
        self.save_project_action = QtGui.QAction("Save Project", self)
        self.save_project_as_action = QtGui.QAction("Save Project As…", self)
        self.exit_action = QtGui.QAction("Exit", self)

        self.load_csv_action = QtGui.QAction("Load Markers CSV…", self)
        self.add_images_dir_action = QtGui.QAction("Add Images Folder…", self)
        self.rescan_images_action = QtGui.QAction("Rescan Images", self)
        self.auto_fill_action = QtGui.QAction("Auto-Fill Images", self)
        self.quick_render_action = QtGui.QAction("Quick Render", self)
        self.full_export_action = QtGui.QAction("Full Export (FFmpeg)", self)

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_project_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        assets_menu = self.menuBar().addMenu("Assets")
        assets_menu.addAction(self.load_csv_action)
        assets_menu.addAction(self.add_images_dir_action)
        assets_menu.addAction(self.rescan_images_action)
        assets_menu.addAction(self.auto_fill_action)

        render_menu = self.menuBar().addMenu("Render")
        render_menu.addAction(self.quick_render_action)
        render_menu.addAction(self.full_export_action)

    def _create_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction(self.new_project_action)
        toolbar.addAction(self.open_project_action)
        toolbar.addAction(self.save_project_action)
        toolbar.addSeparator()
        toolbar.addAction(self.load_csv_action)
        toolbar.addAction(self.add_images_dir_action)
        toolbar.addAction(self.auto_fill_action)
        toolbar.addSeparator()
        toolbar.addAction(self.quick_render_action)
        toolbar.addAction(self.full_export_action)

    def _end_library_click_sync(self) -> None:
        self._suppress_library_sync = False
        self._library_recent_click = False
        self._library_recent_click_path = ""

    def _create_status_bar(self) -> None:
        status = QtWidgets.QStatusBar()
        self.setStatusBar(status)
        self.status_label = QtWidgets.QLabel("Ready.")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status.addPermanentWidget(self.status_label)
        status.addPermanentWidget(self.progress_bar)

    def _create_central_layout(self) -> None:
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        timeline_panel = self._build_timeline_panel()
        library_panel = self._build_library_panel()
        splitter.addWidget(timeline_panel)
        splitter.addWidget(library_panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setHandleWidth(8)
        splitter.setSizes([900, 520])
        self._main_splitter = splitter
        self._schedule_initial_splitter_sizes()
        self.setCentralWidget(central)

    def _schedule_initial_splitter_sizes(self) -> None:
        QtCore.QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def _apply_initial_splitter_sizes(self) -> None:
        if getattr(self, "_main_splitter", None):
            total_width = self.width() or self.sizeHint().width() or 1480
            left = int(total_width * 0.58)
            right = max(320, total_width - left)
            self._main_splitter.setSizes([left, right])
        if getattr(self, "_timeline_splitter", None):
            total_height = self.height() or self.sizeHint().height() or 900
            top = int(total_height * 0.55)
            bottom = max(280, total_height - top)
            self._timeline_splitter.setSizes([top, bottom])

    def _build_library_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(360)
        panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QtWidgets.QLabel("Image Library")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        dir_row = QtWidgets.QHBoxLayout()
        self.add_dir_btn = QtWidgets.QPushButton("Add Folder…")
        self.remove_dir_btn = QtWidgets.QPushButton("Remove")
        self.rescan_btn = QtWidgets.QPushButton("Rescan")
        dir_row.addWidget(self.add_dir_btn)
        dir_row.addWidget(self.remove_dir_btn)
        dir_row.addWidget(self.rescan_btn)
        layout.addLayout(dir_row)

        self.image_dirs_list = FolderListWidget()
        layout.addWidget(self.image_dirs_list, 1)

        self.image_library = ImageLibraryWidget()
        self.image_library.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_library.setMinimumHeight(200)
        layout.addWidget(self.image_library, 3)

        preview_box = QtWidgets.QGroupBox("Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        self.image_preview_label = QtWidgets.QLabel("Drop or select an image")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumHeight(220)
        self.image_preview_label.setObjectName("imagePreview")
        preview_layout.addWidget(self.image_preview_label)
        layout.addWidget(preview_box)
        return panel

    def _build_timeline_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        outer_layout = QtWidgets.QVBoxLayout(panel)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(6)

        splitter = QtWidgets.QSplitter(Qt.Vertical)
        splitter.setHandleWidth(6)
        outer_layout.addWidget(splitter)
        self._timeline_splitter = splitter

        timeline_section = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(timeline_section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Timeline")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch()
        self.total_duration_label = QtWidgets.QLabel("Total: 0.000s (00:00:00:00)")
        header.addWidget(self.total_duration_label)
        layout.addLayout(header)

        self.timeline_list = TimelineListWidget()
        self.timeline_list.setMinimumHeight(280)
        layout.addWidget(self.timeline_list, 4)

        buttons = QtWidgets.QHBoxLayout()
        self.split_segment_btn = QtWidgets.QPushButton("Split…")
        self.duplicate_segment_btn = QtWidgets.QPushButton("Duplicate")
        self.delete_segment_btn = QtWidgets.QPushButton("Delete")
        self.make_black_btn = QtWidgets.QPushButton("Convert to Black")
        self.auto_fill_btn = QtWidgets.QPushButton("Auto-Fill")
        buttons.addWidget(self.split_segment_btn)
        buttons.addWidget(self.duplicate_segment_btn)
        buttons.addWidget(self.delete_segment_btn)
        buttons.addWidget(self.make_black_btn)
        buttons.addWidget(self.auto_fill_btn)
        layout.addLayout(buttons)

        prop_box = QtWidgets.QGroupBox("Segment Properties")
        form = QtWidgets.QFormLayout(prop_box)
        self.segment_name_edit = QtWidgets.QLineEdit()
        form.addRow("Name", self.segment_name_edit)
        self.segment_duration_spin = QtWidgets.QDoubleSpinBox()
        self.segment_duration_spin.setRange(0.04, 600.0)
        self.segment_duration_spin.setDecimals(3)
        self.segment_duration_spin.setSingleStep(0.04)
        form.addRow("Duration (s)", self.segment_duration_spin)
        self.segment_kind_combo = QtWidgets.QComboBox()
        self.segment_kind_combo.addItems(["Image", "Black"])
        form.addRow("Type", self.segment_kind_combo)
        self.segment_image_label = QtWidgets.QLabel("[No image]")
        form.addRow("Assigned", self.segment_image_label)
        assign_row = QtWidgets.QHBoxLayout()
        self.assign_selected_btn = QtWidgets.QPushButton("Assign Selected Image")
        self.clear_image_btn = QtWidgets.QPushButton("Clear Image")
        self.clear_cache_btn = QtWidgets.QPushButton("Clear Cache")
        assign_row.addWidget(self.assign_selected_btn)
        assign_row.addWidget(self.clear_image_btn)
        assign_row.addWidget(self.clear_cache_btn)
        form.addRow(assign_row)
        self.segment_timecode_label = QtWidgets.QLabel("Start 00:00:00:00 → End 00:00:00:00")
        form.addRow("Timecode", self.segment_timecode_label)
        self.segment_preview_label = QtWidgets.QLabel("Select a segment to preview")
        self.segment_preview_label.setAlignment(Qt.AlignCenter)
        self.segment_preview_label.setMinimumHeight(180)
        self.segment_preview_label.setObjectName("imagePreview")
        form.addRow("Preview", self.segment_preview_label)
        prop_container = QtWidgets.QWidget()
        prop_container_layout = QtWidgets.QVBoxLayout(prop_container)
        prop_container_layout.setContentsMargins(0, 0, 0, 0)
        prop_container_layout.setSpacing(8)
        prop_container_layout.addWidget(prop_box)
        prop_container_layout.addStretch(1)
        prop_scroll = QtWidgets.QScrollArea()
        prop_scroll.setWidgetResizable(True)
        prop_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        prop_scroll.setObjectName("segmentPropertiesScroll")
        prop_scroll.setWidget(prop_container)
        layout.addWidget(prop_scroll)
        splitter.addWidget(timeline_section)

        settings_panel = self._build_settings_panel()
        splitter.addWidget(settings_panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return panel

    def _build_settings_panel(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        tabs.addTab(self._wrap_tab_with_scroll(self._build_paths_tab()), "Project")
        tabs.addTab(self._wrap_tab_with_scroll(self._build_render_tab()), "Render")
        tabs.addTab(self._wrap_tab_with_scroll(self._build_quick_tab()), "Quick Test")
        tabs.addTab(self._wrap_tab_with_scroll(self._build_fx_tab()), "Visual FX")
        return tabs

    def _wrap_tab_with_scroll(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(widget)
        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _create_path_row(self) -> Tuple[QtWidgets.QLineEdit, QtWidgets.QPushButton, QtWidgets.QWidget]:
        line = QtWidgets.QLineEdit()
        button = QtWidgets.QPushButton("Browse…")
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        row_layout.addWidget(line)
        row_layout.addWidget(button)
        return line, button, row

    def _build_paths_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(12)

        self.csv_edit, self.csv_browse_btn, csv_row = self._create_path_row()
        form.addRow("Markers CSV", csv_row)

        self.audio_edit, self.audio_browse_btn, audio_row = self._create_path_row()
        form.addRow("Audio File", audio_row)

        self.quick_out_edit, self.quick_out_btn, quick_row = self._create_path_row()
        form.addRow("Quick Render Out", quick_row)

        self.full_out_edit, self.full_out_btn, full_row = self._create_path_row()
        form.addRow("Full Export Out", full_row)

        self.project_label = QtWidgets.QLabel("Project: (unsaved)")
        form.addRow(self.project_label)

        self.loop_images_check = QtWidgets.QCheckBox("Loop images when auto-filling")
        form.addRow(self.loop_images_check)
        return tab

    def _build_render_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(12)

        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 240)
        form.addRow("FPS", self.fps_spin)

        size_row = QtWidgets.QWidget()
        size_layout = QtWidgets.QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_w_spin = QtWidgets.QSpinBox()
        self.frame_w_spin.setRange(16, 7680)
        self.frame_h_spin = QtWidgets.QSpinBox()
        self.frame_h_spin.setRange(16, 4320)
        size_layout.addWidget(self.frame_w_spin)
        size_layout.addWidget(QtWidgets.QLabel("×"))
        size_layout.addWidget(self.frame_h_spin)
        form.addRow("Frame size", size_row)

        self.audio_gain_spin = QtWidgets.QDoubleSpinBox()
        self.audio_gain_spin.setRange(-60.0, 12.0)
        self.audio_gain_spin.setSingleStep(0.5)
        form.addRow("Audio gain (dB)", self.audio_gain_spin)

        self.audio_offset_spin = QtWidgets.QDoubleSpinBox()
        self.audio_offset_spin.setRange(-10.0, 10.0)
        self.audio_offset_spin.setSingleStep(0.01)
        form.addRow("Audio offset (s)", self.audio_offset_spin)

        self.tc_base_edit = QtWidgets.QLineEdit()
        self.tc_base_edit.setPlaceholderText("Leave blank to infer")
        form.addRow("TC base (s)", self.tc_base_edit)

        self.snap_mode_combo = QtWidgets.QComboBox()
        self.snap_mode_combo.addItems(["round", "floor", "ceil"])
        form.addRow("Snap mode", self.snap_mode_combo)

        self.ffmpeg_path_edit = QtWidgets.QLineEdit()
        self.ffmpeg_path_btn = QtWidgets.QPushButton("Browse…")
        ffmpeg_row = QtWidgets.QWidget()
        ffmpeg_layout = QtWidgets.QHBoxLayout(ffmpeg_row)
        ffmpeg_layout.setContentsMargins(0, 0, 0, 0)
        ffmpeg_layout.addWidget(self.ffmpeg_path_edit)
        ffmpeg_layout.addWidget(self.ffmpeg_path_btn)
        form.addRow("FFmpeg", ffmpeg_row)

        self.ffprobe_path_edit = QtWidgets.QLineEdit()
        self.ffprobe_path_btn = QtWidgets.QPushButton("Browse…")
        ffprobe_row = QtWidgets.QWidget()
        ffprobe_layout = QtWidgets.QHBoxLayout(ffprobe_row)
        ffprobe_layout.setContentsMargins(0, 0, 0, 0)
        ffprobe_layout.addWidget(self.ffprobe_path_edit)
        ffprobe_layout.addWidget(self.ffprobe_path_btn)
        form.addRow("FFprobe", ffprobe_row)

        self.use_nvenc_check = QtWidgets.QCheckBox("Use NVIDIA NVENC (if available)")
        form.addRow(self.use_nvenc_check)
        self.use_shortest_check = QtWidgets.QCheckBox("Use -shortest safeguard")
        form.addRow(self.use_shortest_check)
        self.pre_norm_check = QtWidgets.QCheckBox("Pre-normalize images (recommended)")
        form.addRow(self.pre_norm_check)

        pre_row = QtWidgets.QWidget()
        pre_layout = QtWidgets.QHBoxLayout(pre_row)
        pre_layout.setContentsMargins(0, 0, 0, 0)
        self.pre_format_combo = QtWidgets.QComboBox()
        self.pre_format_combo.addItems(["JPEG", "PNG"])
        self.pre_quality_spin = QtWidgets.QSpinBox()
        self.pre_quality_spin.setRange(1, 100)
        pre_layout.addWidget(QtWidgets.QLabel("Format"))
        pre_layout.addWidget(self.pre_format_combo)
        pre_layout.addWidget(QtWidgets.QLabel("Quality"))
        pre_layout.addWidget(self.pre_quality_spin)
        form.addRow("Pre-save", pre_row)

        return tab

    def _build_quick_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(12)

        self.quick_mode_combo = QtWidgets.QComboBox()
        self.quick_mode_combo.addItems(["first_marker", "absolute", "interval_index"])
        form.addRow("Start mode", self.quick_mode_combo)

        self.preroll_spin = QtWidgets.QDoubleSpinBox()
        self.preroll_spin.setRange(0.0, 30.0)
        self.preroll_spin.setSingleStep(0.25)
        form.addRow("Preroll (s)", self.preroll_spin)

        self.quick_start_spin = QtWidgets.QDoubleSpinBox()
        self.quick_start_spin.setRange(0.0, 10_000.0)
        self.quick_start_spin.setSingleStep(0.25)
        form.addRow("Start @ (s)", self.quick_start_spin)

        self.quick_index_spin = QtWidgets.QSpinBox()
        self.quick_index_spin.setRange(0, 10_000)
        form.addRow("Start interval #", self.quick_index_spin)

        self.quick_dur_spin = QtWidgets.QDoubleSpinBox()
        self.quick_dur_spin.setRange(0.0, 600.0)
        self.quick_dur_spin.setSingleStep(0.5)
        form.addRow("Duration (s)", self.quick_dur_spin)

        self.quick_n_spin = QtWidgets.QSpinBox()
        self.quick_n_spin.setRange(1, 1000)
        form.addRow("Fallback intervals", self.quick_n_spin)
        return tab

    def _build_fx_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        glow_group = QtWidgets.QGroupBox("Neon Glow")
        glow_form = QtWidgets.QFormLayout(glow_group)
        self.glow_enable_check = QtWidgets.QCheckBox("Enable neon glow on image edges")
        glow_form.addRow(self.glow_enable_check)
        self.glow_color_combo = QtWidgets.QComboBox()
        for name, value in NEON_COLORS:
            self.glow_color_combo.addItem(name, value)
        self.glow_preset_combo = QtWidgets.QComboBox()
        for key, preset in PROC_GLOW_PRESETS.items():
            self.glow_preset_combo.addItem(str(preset.get("label", key.title())), key)
        self.glow_width_spin = QtWidgets.QSpinBox()
        self.glow_width_spin.setRange(1, 40)
        self.glow_width_spin.setSingleStep(1)
        self.glow_intensity_spin = QtWidgets.QDoubleSpinBox()
        self.glow_intensity_spin.setRange(0.10, 2.00)
        self.glow_intensity_spin.setSingleStep(0.05)
        self.glow_intensity_spin.setDecimals(2)
        glow_form.addRow("Color", self.glow_color_combo)
        glow_form.addRow("Preset style", self.glow_preset_combo)
        glow_form.addRow("Edge width (px)", self.glow_width_spin)
        glow_form.addRow("Intensity", self.glow_intensity_spin)
        layout.addWidget(glow_group)

        beat_group = QtWidgets.QGroupBox("Beat / Transition Pops")
        beat_form = QtWidgets.QFormLayout(beat_group)
        self.impact_enable_check = QtWidgets.QCheckBox("Enable beat impacts & transition pops")
        beat_form.addRow(self.impact_enable_check)
        self.impact_kind_combo = QtWidgets.QComboBox()
        for key, preset in IMPACT_PRESETS.items():
            self.impact_kind_combo.addItem(preset["label"], key)
        self.impact_duration_spin = QtWidgets.QDoubleSpinBox()
        self.impact_duration_spin.setRange(0.01, 1.0)
        self.impact_duration_spin.setSingleStep(0.01)
        self.impact_duration_spin.setDecimals(3)
        beat_form.addRow("Preset", self.impact_kind_combo)
        beat_form.addRow("Tail duration (s)", self.impact_duration_spin)
        layout.addWidget(beat_group)

        waveform_group = QtWidgets.QGroupBox("Audio Waveform Overlay")
        waveform_form = QtWidgets.QFormLayout(waveform_group)
        self.waveform_enable_check = QtWidgets.QCheckBox("Show waveform along bottom edge")
        waveform_form.addRow(self.waveform_enable_check)
        self.waveform_style_combo = QtWidgets.QComboBox()
        for label, value in WAVEFORM_STYLE_CHOICES:
            self.waveform_style_combo.addItem(label, value)
        idx = self.waveform_style_combo.findData(DEFAULTS.get("WAVEFORM_STYLE", "oscilloscope"))
        if idx >= 0:
            self.waveform_style_combo.setCurrentIndex(idx)
        self.waveform_color_combo = QtWidgets.QComboBox()
        for name, value in NEON_COLORS:
            self.waveform_color_combo.addItem(name, value)
        self.waveform_height_spin = QtWidgets.QDoubleSpinBox()
        self.waveform_height_spin.setRange(0.05, 0.6)
        self.waveform_height_spin.setSingleStep(0.01)
        self.waveform_height_spin.setDecimals(2)
        self.waveform_opacity_spin = QtWidgets.QDoubleSpinBox()
        self.waveform_opacity_spin.setRange(0.1, 1.0)
        self.waveform_opacity_spin.setSingleStep(0.05)
        self.waveform_opacity_spin.setDecimals(2)
        self.waveform_height_spin.setValue(DEFAULTS["WAVEFORM_HEIGHT_PCT"])
        self.waveform_opacity_spin.setValue(DEFAULTS["WAVEFORM_OPACITY"])
        waveform_form.addRow("Style", self.waveform_style_combo)
        waveform_form.addRow("Color", self.waveform_color_combo)
        waveform_form.addRow("Height (fraction of frame)", self.waveform_height_spin)
        waveform_form.addRow("Opacity", self.waveform_opacity_spin)
        layout.addWidget(waveform_group)

        layout.addStretch(1)

        self.glow_enable_check.toggled.connect(self._update_glow_controls_enabled)
        self.impact_enable_check.toggled.connect(self._update_impact_controls_enabled)
        self.waveform_enable_check.toggled.connect(self._update_waveform_controls_enabled)
        return tab

    def _connect_ui(self) -> None:
        self.new_project_action.triggered.connect(self.new_project)
        self.open_project_action.triggered.connect(self.load_project_dialog)
        self.save_project_action.triggered.connect(self.save_project)
        self.save_project_as_action.triggered.connect(self.save_project_as)
        self.exit_action.triggered.connect(self.close)

        self.load_csv_action.triggered.connect(self.load_markers_dialog)
        self.add_images_dir_action.triggered.connect(self.add_image_dir_dialog)
        self.rescan_images_action.triggered.connect(self.rescan_images)
        self.auto_fill_action.triggered.connect(self.auto_fill_segments)

        self.quick_render_action.triggered.connect(self.start_quick_render)
        self.full_export_action.triggered.connect(self.start_full_export)

        self.add_dir_btn.clicked.connect(self.add_image_dir_dialog)
        self.remove_dir_btn.clicked.connect(self.remove_selected_dir)
        self.rescan_btn.clicked.connect(self.rescan_images)

        self.image_library.imageActivated.connect(self._on_library_image_activated)
        self.image_library.imageDoubleClicked.connect(self._on_library_image_double_clicked)

        self.timeline_list.itemSelectionChanged.connect(self._on_timeline_selection_changed)
        self.timeline_list.orderChanged.connect(self._handle_timeline_order_changed)
        self.timeline_list.imageDropped.connect(self._assign_image_to_segment_id)
        self.timeline_list.bundleDropped.connect(self._assign_image_bundle)
        self.timeline_list.segmentActivated.connect(self._focus_segment_properties)

        self.split_segment_btn.clicked.connect(self.split_selected_segment)
        self.duplicate_segment_btn.clicked.connect(self.duplicate_selected_segment)
        self.delete_segment_btn.clicked.connect(self.delete_selected_segments)
        self.make_black_btn.clicked.connect(self.convert_selected_to_black)
        self.auto_fill_btn.clicked.connect(self.auto_fill_segments)

        self.assign_selected_btn.clicked.connect(self.assign_selected_image_to_segment)
        self.clear_image_btn.clicked.connect(self.clear_segment_image)
        self.clear_cache_btn.clicked.connect(self._clear_render_cache)
        self.segment_name_edit.editingFinished.connect(self._on_segment_name_changed)
        self.segment_duration_spin.valueChanged.connect(self._on_segment_duration_changed)
        self.segment_kind_combo.currentIndexChanged.connect(self._on_segment_kind_changed)

        self.csv_browse_btn.clicked.connect(self._browse_csv)
        self.audio_browse_btn.clicked.connect(self._browse_audio)
        self.quick_out_btn.clicked.connect(self._browse_quick_out)
        self.full_out_btn.clicked.connect(self._browse_full_out)
        self.ffmpeg_path_btn.clicked.connect(self._browse_ffmpeg)
        self.ffprobe_path_btn.clicked.connect(self._browse_ffprobe)

        self.fps_spin.valueChanged.connect(lambda _: self._recalculate_segment_timings())

    # --- Utility ---------------------------------------------------------
    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusBar().showMessage(message, 4000)

    def _current_glow_options(self) -> GlowOptions:
        color_value = self.glow_color_combo.currentData() if hasattr(self, "glow_color_combo") else "#00f6ff"
        return GlowOptions(
            enabled=self.glow_enable_check.isChecked() if hasattr(self, "glow_enable_check") else False,
            color=_hex_to_rgb(str(color_value or "#00f6ff")),
            width=int(self.glow_width_spin.value()) if hasattr(self, "glow_width_spin") else 6,
            intensity=float(self.glow_intensity_spin.value()) if hasattr(self, "glow_intensity_spin") else 0.6,
            preset=str(self.glow_preset_combo.currentData() or "legacy") if hasattr(self, "glow_preset_combo") else "legacy",
        )

    def _current_waveform_options(self) -> WaveformOptions:
        color_value = self.waveform_color_combo.currentData() if hasattr(self, "waveform_color_combo") else "#00f6ff"
        style_value = self.waveform_style_combo.currentData() if hasattr(self, "waveform_style_combo") else "oscilloscope"
        return WaveformOptions(
            enabled=self.waveform_enable_check.isChecked() if hasattr(self, "waveform_enable_check") else False,
            color=_hex_to_rgb(str(color_value or "#00f6ff")),
            opacity=float(self.waveform_opacity_spin.value()) if hasattr(self, "waveform_opacity_spin") else 0.6,
            height_pct=max(0.05, min(0.6, float(self.waveform_height_spin.value()) if hasattr(self, "waveform_height_spin") else 0.18)),
            style=str(style_value or "oscilloscope"),
        )

    def _selected_impact_kind(self) -> str:
        enabled = self.impact_enable_check.isChecked() if hasattr(self, "impact_enable_check") else False
        if not enabled:
            return "disabled"
        return str(self.impact_kind_combo.currentData() or "disabled")

    def _update_glow_controls_enabled(self) -> None:
        enabled = self.glow_enable_check.isChecked()
        for widget in (
            self.glow_color_combo,
            self.glow_preset_combo,
            self.glow_width_spin,
            self.glow_intensity_spin,
        ):
            widget.setEnabled(enabled)

    def _update_waveform_controls_enabled(self) -> None:
        enabled = self.waveform_enable_check.isChecked()
        for widget in (
            self.waveform_style_combo,
            self.waveform_color_combo,
            self.waveform_height_spin,
            self.waveform_opacity_spin,
        ):
            widget.setEnabled(enabled)

    def _update_impact_controls_enabled(self) -> None:
        enabled = self.impact_enable_check.isChecked()
        for widget in (
            self.impact_kind_combo,
            self.impact_duration_spin,
        ):
            widget.setEnabled(enabled)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for action in (
            self.new_project_action,
            self.open_project_action,
            self.save_project_action,
            self.save_project_as_action,
            self.load_csv_action,
            self.add_images_dir_action,
            self.auto_fill_action,
            self.quick_render_action,
            self.full_export_action,
        ):
            action.setEnabled(not busy)
        self.assign_selected_btn.setEnabled(not busy)
        self.clear_image_btn.setEnabled(not busy)

    def _update_progress(self, value: float) -> None:
        pct = max(0.0, min(1.0, value)) * 100.0
        self.progress_bar.setValue(int(pct))

    def _clear_progress(self) -> None:
        self.progress_bar.setValue(0)

    def _focus_segment_properties(self, seg_id: str) -> None:
        for i in range(self.timeline_list.count()):
            item = self.timeline_list.item(i)
            if item.data(Qt.UserRole) == seg_id:
                self.timeline_list.setCurrentRow(i)
                self.segment_name_edit.setFocus(Qt.TabFocusReason)
                break

    def _resolve_path(self, path_str: str) -> Path:
        if _is_running_in_wsl() and ":" in path_str:
            return _windows_to_wsl_path(path_str)
        return Path(path_str)

    def _path_exists_runtime(self, path_str: str) -> Tuple[bool, str]:
        if not path_str:
            return False, ""
        candidate = Path(path_str)
        if candidate.exists():
            return True, str(candidate)
        if _is_running_in_wsl() and ":" in path_str:
            alt = _windows_to_wsl_path(path_str)
            if alt.exists():
                return True, str(alt)
        return False, str(candidate)

    def _to_runtime_path(self, path_str: str) -> str:
        if _is_running_in_wsl() and ":" in path_str:
            return str(_windows_to_wsl_path(path_str))
        return path_str

    # --- Loading assets --------------------------------------------------
    def new_project(self) -> None:
        if self._busy:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "New project",
            "Start a new project? Unsaved changes will be lost.",
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.marker_frames = []
        self.marker_frames_raw = []
        self.segments = []
        self.current_project_path = None
        self.project_label.setText("Project: (unsaved)")
        self.csv_edit.clear()
        self.audio_edit.clear()
        self.quick_out_edit.clear()
        self.full_out_edit.clear()
        self._refresh_timeline(preserve_selection=False)
        self._set_status("New project")

    def load_markers_dialog(self) -> None:
        if self._busy:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Resolve Markers CSV",
            str(Path(self.csv_edit.text()).parent if self.csv_edit.text() else Path.home()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not path:
            return
        self.csv_edit.setText(path)
        self.load_markers_from_csv(path)

    def load_markers_from_csv(self, path: str) -> None:
        try:
            fps = self.fps_spin.value()
            raw_frames = read_markers_frames(path, fps)
            if not raw_frames:
                QtWidgets.QMessageBox.warning(
                    self, "Markers", "No valid markers detected in this CSV."
                )
                return
            raw_frames = sorted(set(raw_frames))
            self.marker_frames_raw = raw_frames

            base_text = self.tc_base_edit.text().strip()
            base_seconds = 0.0
            if base_text:
                try:
                    base_seconds = float(base_text)
                except ValueError:
                    base_seconds = infer_timecode_base_seconds(raw_frames, fps)
                    if base_seconds:
                        self.tc_base_edit.setText(str(base_seconds))
            else:
                base_seconds = infer_timecode_base_seconds(raw_frames, fps)
                if base_seconds:
                    self.tc_base_edit.setText(str(base_seconds))
            base_frames = int(round(max(0.0, base_seconds) * fps))

            adjusted_frames = [max(0, frame - base_frames) for frame in raw_frames]
            adjusted_frames = sorted(set(adjusted_frames))
            head_frames = adjusted_frames[0] if adjusted_frames else 0
            relative_frames = [frame - head_frames for frame in adjusted_frames]
            self.marker_frames = relative_frames
            self.head_frames = head_frames

            intervals = build_intervals(self.marker_frames)
            self.segments = []
            min_duration = 1.0 / max(1, fps)
            if head_frames > 0:
                self.segments.append(
                    Segment(
                        name="Intro Black",
                        duration=max(min_duration, head_frames / fps),
                        kind="black",
                    )
                )
            for idx, (a, b) in enumerate(intervals, start=1):
                dur = max(min_duration, (b - a) / fps)
                self.segments.append(
                    Segment(name=f"Segment {idx:03d}", duration=dur, kind="image")
                )
            self._refresh_timeline(preserve_selection=False)
            self._set_status(
                f"Loaded {len(self.marker_frames)} markers → {len(intervals)} intervals"
            )
        except Exception as exc:  # noqa: broad-except
            QtWidgets.QMessageBox.critical(self, "CSV error", str(exc))

    def add_image_dir_dialog(self) -> None:
        if self._busy:
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select images folder", str(Path.home())
        )
        if not directory:
            return
        if directory not in self.image_dirs:
            self.image_dirs.append(directory)
            self._update_image_dirs_list()
            self.rescan_images()
            self._set_status(f"Added images folder: {directory}")

    def remove_selected_dir(self) -> None:
        sel = self.image_dirs_list.currentRow()
        if 0 <= sel < len(self.image_dirs):
            removed = self.image_dirs.pop(sel)
            self._update_image_dirs_list()
            self.rescan_images()
            self._set_status(f"Removed folder: {removed}")

    def _update_image_dirs_list(self) -> None:
        self.image_dirs_list.clear()
        for directory in self.image_dirs:
            display = _friendly_folder_name(directory) or directory
            item = QtWidgets.QListWidgetItem(display)
            item.setData(Qt.UserRole, directory)
            item.setToolTip(directory)
            self.image_dirs_list.addItem(item)
        if isinstance(self.image_dirs_list, FolderListWidget):
            self.image_dirs_list.set_folder_images(self.images_by_dir)

    def rescan_images(self) -> None:
        if self._image_scan_thread and self._image_scan_thread.isRunning():
            self._pending_image_scan = True
            return

        if not self.image_dirs:
            self.images = []
            self.images_by_dir = {}
            self._update_image_library()
            if isinstance(self.image_dirs_list, FolderListWidget):
                self.image_dirs_list.set_folder_images(self.images_by_dir)
            self._set_status("No image folders configured.")
            return

        resolved_dirs: List[Tuple[str, str]] = []
        for directory in self.image_dirs:
            resolved = self._resolve_path(directory)
            resolved_dirs.append((directory, str(resolved)))

        self.rescan_btn.setEnabled(False)
        self._set_status("Scanning images…")

        self._image_scan_worker = ImageScanWorker(resolved_dirs, IMAGE_EXTS)
        self._image_scan_thread = QtCore.QThread(self)
        self._image_scan_worker.moveToThread(self._image_scan_thread)
        self._image_scan_thread.started.connect(self._image_scan_worker.run)
        self._image_scan_worker.finished.connect(self._on_image_scan_finished)
        self._image_scan_worker.error.connect(self._on_image_scan_error)
        self._image_scan_thread.start()

    def _update_image_library(self) -> None:
        self.image_library.set_images(self.images_by_dir)
        if not self.images:
            self._clear_image_preview()
            return
        current = self._current_segment()
        if current and current.kind == "image" and current.image_path:
            self.image_library.select_image(current.image_path, center=False)

    def _cleanup_image_scan(self) -> None:
        if self._image_scan_thread:
            self._image_scan_thread.quit()
            self._image_scan_thread.wait()
            self._image_scan_thread.deleteLater()
        if self._image_scan_worker:
            self._image_scan_worker.deleteLater()
        self._image_scan_thread = None
        self._image_scan_worker = None
        if hasattr(self, "rescan_btn"):
            self.rescan_btn.setEnabled(True)

    def _on_image_scan_finished(self, data: Dict[str, List[str]]) -> None:
        self._cleanup_image_scan()
        images_by_dir: Dict[str, List[Path]] = {}
        collected: List[str] = []
        for original, paths in data.items():
            path_objs = [self._resolve_path(p) for p in paths]
            images_by_dir[original] = path_objs
            collected.extend(paths)
        self.images_by_dir = images_by_dir
        if isinstance(self.image_dirs_list, FolderListWidget):
            self.image_dirs_list.set_folder_images(images_by_dir)
        if collected:
            dedup: List[Path] = []
            seen: set[str] = set()
            for p in collected:
                norm = self._normalize_compare_path(p)
                if norm in seen:
                    continue
                seen.add(norm)
                dedup.append(self._resolve_path(p))
            self.images = dedup
        else:
            self.images = []
        self._update_image_library()
        self._set_status(f"Loaded {len(self.images)} image(s)")
        if self._pending_image_scan:
            self._pending_image_scan = False
            QtCore.QTimer.singleShot(0, self.rescan_images)

    def _on_image_scan_error(self, message: str) -> None:
        self._cleanup_image_scan()
        QtWidgets.QMessageBox.warning(self, "Image scan", message)
        self._set_status("Image scan failed.")
        if self._pending_image_scan:
            self._pending_image_scan = False
            QtCore.QTimer.singleShot(0, self.rescan_images)

    def _normalize_compare_path(self, path: str) -> str:
        if not path:
            return ""
        text = str(path).strip()
        if not text:
            return ""
        if _is_running_in_wsl() and ":" in text:
            text = str(_windows_to_wsl_path(text))
        text = text.replace("\\", "/")
        while "//" in text:
            text = text.replace("//", "/")
        normalized = os.path.normcase(os.path.normpath(text))
        if PATH_DEBUG:
            print(f"[PATH] normalize '{path}' -> '{normalized}'")
        return normalized

    def _find_segment_indices_for_image(self, path: str) -> List[int]:
        target = self._normalize_compare_path(path)
        if not target:
            return []
        matches: List[int] = []
        for idx, seg in enumerate(self.segments):
            if seg.kind != "image" or not seg.image_path:
                continue
            if self._normalize_compare_path(seg.image_path) == target:
                matches.append(idx)
        return matches

    def _highlight_segments_for_image(self, path: str, scroll_timeline: bool = True) -> None:
        indices = self._find_segment_indices_for_image(path)
        if not indices:
            return
        _log(f"Highlight segments for image: {path} -> {indices}")
        try:
            self.timeline_list.blockSignals(True)
            self.timeline_list.clearSelection()
            for idx in indices:
                item = self.timeline_list.item(idx)
                if item:
                    item.setSelected(True)
            if scroll_timeline and indices:
                self.timeline_list.setCurrentRow(indices[0])
        finally:
            self.timeline_list.blockSignals(False)
        self._on_timeline_selection_changed()

    def _on_library_image_activated(self, path: str) -> None:
        if self._suppress_library_activation:
            return
        current_widget = self.image_library.currentWidget()
        if isinstance(current_widget, ThumbnailListWidget):
            self.image_library._remember_scroll(current_widget)  # type: ignore[attr-defined]
            current_widget.push_scroll_suppression()
        self._suppress_library_sync = True
        self._library_recent_click = True
        self._library_recent_click_path = path
        _log(
            f"[LibraryClick] path={path} suppress_sync={self._suppress_library_sync} widget={type(current_widget).__name__}"
        )
        try:
            _log(f"Library image clicked: {path}")
            self._update_image_preview(path)
            # No timeline sync when clicking in the library; only preview updates.
        finally:
            if isinstance(current_widget, ThumbnailListWidget):
                self.image_library._restore_scroll(current_widget)  # type: ignore[attr-defined]
                QtCore.QTimer.singleShot(0, current_widget.pop_scroll_suppression)
            if hasattr(self, "_library_scroll_guard"):
                self._library_scroll_guard.start(350)
            else:
                self._end_library_click_sync()

    def _on_library_image_double_clicked(self, path: str) -> None:
        if not path:
            return
        resolved = self._resolve_path(path)
        candidate = resolved if resolved.exists() else Path(path)
        dialog = ImagePreviewDialog(str(candidate), self)
        dialog.exec()

    def _auto_fill_default_images(self) -> None:
        if not self.segments or not self.images:
            return
        try:
            intervals, _ = timeline_to_intervals(
                self.segments, max(1, self.fps_spin.value())
            )
            prior = [seg.image_path for seg in self.segments]
            assigned = assign_images(
                intervals,
                self.images,
                loop=self.loop_images_check.isChecked(),
                prior_assign=prior,
            )
        except Exception:
            return
        for seg, path in zip(self.segments, assigned):
            if seg.kind == "image":
                seg.image_path = path
        self._refresh_timeline(preserve_selection=False)

    def _update_image_preview(self, path: str) -> None:
        if not path or not Path(path).exists():
            self._clear_image_preview()
            return
        try:
            target = self.image_preview_label.size()
            width = max(360, target.width())
            height = max(240, target.height())
            pix = get_thumbnail(path, width, height, "Preview")
            self.image_preview_label.setPixmap(pix)
            self.image_preview_label.setText("")
        except Exception as exc:  # noqa: broad-except
            self.image_preview_label.setText(f"Preview error: {exc}")

    def _clear_image_preview(self) -> None:
        self.image_preview_label.setPixmap(QtGui.QPixmap())
        self.image_preview_label.setText("Drop or select an image")

    def _update_segment_preview(self, path: str) -> None:
        if not hasattr(self, "segment_preview_label"):
            return
        if not path:
            self._clear_segment_preview()
            return
        resolved = self._resolve_path(path)
        candidate = resolved if resolved.exists() else Path(path)
        placeholder = "Missing" if not candidate.exists() else "Preview"
        try:
            pixmap = get_thumbnail(str(candidate), 240, 160, placeholder)
        except Exception:
            pixmap = QtGui.QPixmap()
        if pixmap and not pixmap.isNull():
            self.segment_preview_label.setPixmap(pixmap)
            self.segment_preview_label.setText("")
        else:
            self.segment_preview_label.setPixmap(QtGui.QPixmap())
            self.segment_preview_label.setText("Preview unavailable")

    def _clear_segment_preview(self) -> None:
        if hasattr(self, "segment_preview_label"):
            self.segment_preview_label.setPixmap(QtGui.QPixmap())
            self.segment_preview_label.setText("Select a segment to preview")

    def _nvenc_supported(self, ffmpeg_path: str) -> bool:
        key = str(ffmpeg_path or "").strip() or "ffmpeg"
        cached = self._nvenc_cache.get(key)
        if cached is not None:
            return cached
        try:
            proc = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=6,
            )
            stdout = proc.stdout.lower()
            available = "h264_nvenc" in stdout or "hevc_nvenc" in stdout
        except Exception:
            available = False
        self._nvenc_cache[key] = available
        if available:
            _log(f"[NVENC] NVENC encoders detected for '{ffmpeg_path}'.")
        else:
            _log(f"[NVENC] NVENC not available for '{ffmpeg_path}'.")
        return available

    # --- Timeline operations --------------------------------------------
    def _refresh_timeline(self, preserve_selection: bool = True) -> None:
        self._recalculate_segment_timings()
        selected_ids = set()
        if preserve_selection:
            for item in self.timeline_list.selectedItems():
                selected_ids.add(item.data(Qt.UserRole))

        fps = max(1, self.fps_spin.value())
        current_count = self.timeline_list.count()
        needed = len(self.segments)

        self.timeline_list.setUpdatesEnabled(False)
        self.timeline_list.blockSignals(True)
        try:
            if current_count < needed:
                for i in range(current_count, needed):
                    self.timeline_list.addItem(SegmentItem(self.segments[i], fps))
            elif current_count > needed:
                for _ in range(current_count - needed):
                    self.timeline_list.takeItem(needed)

            first_selected_row = -1
            for idx, seg in enumerate(self.segments):
                item = self.timeline_list.item(idx)
                if isinstance(item, SegmentItem):
                    item.update_from_segment(seg, fps)
                else:
                    item = SegmentItem(seg, fps)
                    self.timeline_list.takeItem(idx)
                    self.timeline_list.insertItem(idx, item)
                should_select = seg.id in selected_ids
                item.setSelected(should_select)
                if should_select and first_selected_row == -1:
                    first_selected_row = idx
            if first_selected_row >= 0:
                self.timeline_list.setCurrentRow(first_selected_row)
                current_item = self.timeline_list.item(first_selected_row)
                if current_item:
                    self.timeline_list.scrollToItem(
                        current_item, QtWidgets.QAbstractItemView.PositionAtCenter
                    )
        finally:
            self.timeline_list.blockSignals(False)
            self.timeline_list.setUpdatesEnabled(True)

        self._on_timeline_selection_changed()

    def _refresh_timeline_items_only(self) -> None:
        fps = max(1, self.fps_spin.value())
        count = min(self.timeline_list.count(), len(self.segments))
        self.timeline_list.setUpdatesEnabled(False)
        try:
            for i in range(count):
                item = self.timeline_list.item(i)
                seg = self.segments[i]
                if isinstance(item, SegmentItem):
                    item.update_from_segment(seg, fps)
                else:
                    self.timeline_list.takeItem(i)
                    self.timeline_list.insertItem(i, SegmentItem(seg, fps))
        finally:
            self.timeline_list.setUpdatesEnabled(True)

    def _recalculate_segment_timings(self) -> None:
        fps = max(1, self.fps_spin.value())
        min_duration = 1.0 / fps
        current = 0.0
        for seg in self.segments:
            seg.ensure_min_duration(min_duration)
            seg.start = current
            seg.end = current + seg.duration
            current = seg.end
        self.total_duration_label.setText(
            f"Total: {current:.3f}s ({seconds_to_tc(current, fps)})"
        )
        self._refresh_timeline_items_only()

    def _normalize_segment_image_paths(self) -> None:
        if not _is_running_in_wsl():
            return
        for seg in self.segments:
            if not seg.image_path:
                continue
            resolved = self._resolve_path(seg.image_path)
            if resolved.exists():
                seg.image_path = str(resolved)

    def _current_segment_ids(self) -> List[str]:
        return [item.data(Qt.UserRole) for item in self.timeline_list.selectedItems()]

    def _find_segment(self, seg_id: str) -> Optional[Segment]:
        for seg in self.segments:
            if seg.id == seg_id:
                return seg
        return None

    def _current_segment(self) -> Optional[Segment]:
        ids = self._current_segment_ids()
        return self._find_segment(ids[0]) if ids else None

    def _on_timeline_selection_changed(self) -> None:
        seg = self._current_segment()
        self._load_segment_into_form(seg)

    def _load_segment_into_form(self, segment: Optional[Segment]) -> None:
        self._loading_segment_ui = True
        enabled = segment is not None
        for widget in (
            self.segment_name_edit,
            self.segment_duration_spin,
            self.segment_kind_combo,
            self.assign_selected_btn,
            self.clear_image_btn,
        ):
            widget.setEnabled(enabled)
        if not segment:
            self.segment_name_edit.clear()
            self.segment_duration_spin.setValue(1.0)
            self.segment_kind_combo.setCurrentIndex(0)
            self.segment_image_label.setText("[No image]")
            self.segment_timecode_label.setText("Start 00:00:00:00 → End 00:00:00:00")
            self._clear_segment_preview()
            self._loading_segment_ui = False
            return
        self.segment_name_edit.setText(segment.name)
        self.segment_duration_spin.setValue(segment.duration)
        self.segment_kind_combo.setCurrentIndex(0 if segment.kind == "image" else 1)
        img_label = Path(segment.image_path).name if segment.image_path else "[No image]"
        self.segment_image_label.setText(img_label)
        fps = max(1, self.fps_spin.value())
        self.segment_timecode_label.setText(
            f"{seconds_to_tc(segment.start, fps)} → {seconds_to_tc(segment.end, fps)}"
        )
        if segment.kind == "image" and segment.image_path:
            self._update_segment_preview(segment.image_path)
            matched = False
            if not self._suppress_library_sync:
                target_norm = self._normalize_compare_path(segment.image_path)
                recent_norm = self._normalize_compare_path(
                    getattr(self, "_library_recent_click_path", "")
                )
                current_path = self.image_library.current_image_path()
                current_norm = self._normalize_compare_path(current_path)
                _log(
                    "[LibrarySync] target=%s current=%s recent=%s suppress_click=%s"
                    % (
                        target_norm,
                        current_norm,
                        recent_norm,
                        getattr(self, "_library_recent_click", False),
                    )
                )
                if recent_norm and recent_norm == target_norm:
                    matched = True
                    self._library_recent_click_path = ""
                    self._library_recent_click = False
                    _log("[LibrarySync] matched recent click; no scroll")
                elif current_norm == target_norm:
                    matched = True
                    _log("[LibrarySync] already selected; no action")
                else:
                    self._suppress_library_activation = True
                    try:
                        _log(
                            f"[LibrarySync] selecting path={segment.image_path} center=False"
                        )
                        matched = self.image_library.select_image(
                            segment.image_path, center=False
                        )
                    finally:
                        self._suppress_library_activation = False
                if matched:
                    self._library_recent_click_path = ""
                    self._library_recent_click = False
            if not matched:
                _log("[LibrarySync] preview unchanged for timeline selection")
        else:
            self._clear_segment_preview()
        self._loading_segment_ui = False

    def _assign_image_to_segment_id(self, seg_id: str, path: str) -> None:
        seg = self._find_segment(seg_id)
        if not seg:
            return
        resolved = self._resolve_path(path)
        if resolved.exists():
            path = str(resolved)
        seg.kind = "image"
        seg.image_path = path
        self._refresh_segment_item(seg.id)
        if seg == self._current_segment():
            self._load_segment_into_form(seg)
        self._suppress_library_activation = True
        try:
            self.image_library.select_image(path, center=False)
        finally:
            self._suppress_library_activation = False
        self._set_status(f"Assigned {Path(path).name} to {seg.name}")
        self._notify_duplicate_image_usage(path)

    def _bundle_insert_index(self) -> int:
        last = -1
        for idx, seg in enumerate(self.segments):
            if seg.kind == "image" and seg.image_path:
                last = idx
        return max(0, last + 1)

    def _next_available_segment_index(self, start_index: int) -> Optional[int]:
        for idx in range(max(0, start_index), len(self.segments)):
            seg = self.segments[idx]
            if seg.kind != "image":
                continue
            if seg.image_path:
                continue
            return idx
        return None

    def _assign_image_bundle(self, paths: List[str]) -> None:
        cleaned = [str(p) for p in paths if p]
        if not cleaned:
            return
        if not self.segments:
            QtWidgets.QMessageBox.information(
                self, "Folder bundle", "Load or create segments before assigning images."
            )
            return
        start_idx = self._bundle_insert_index()
        if start_idx >= len(self.segments):
            QtWidgets.QMessageBox.information(
                self, "Folder bundle", "No empty image segments remain in the timeline."
            )
            return
        assigned = 0
        cursor = start_idx
        for candidate in cleaned:
            target_idx = self._next_available_segment_index(cursor)
            if target_idx is None:
                break
            seg = self.segments[target_idx]
            resolved = self._resolve_path(candidate)
            seg.kind = "image"
            seg.image_path = str(resolved) if resolved.exists() else candidate
            self._notify_duplicate_image_usage(seg.image_path)
            cursor = target_idx + 1
            assigned += 1
        if assigned == 0:
            QtWidgets.QMessageBox.information(
                self, "Folder bundle", "No empty image segments remain after current assignments."
            )
            return
        self._refresh_timeline()
        target_row = min(cursor - 1, self.timeline_list.count() - 1)
        if target_row >= 0:
            self.timeline_list.setCurrentRow(target_row)
        note = " (ran out of segments)" if assigned < len(cleaned) else ""
        self._set_status(f"Added {assigned} image(s) from folder bundle{note}")

    def _notify_duplicate_image_usage(self, path: str) -> None:
        if not path:
            return
        try:
            target = Path(path).name.lower()
        except Exception:
            target = str(path).lower()
        if not target:
            return
        count = 0
        for seg in self.segments:
            if not seg.image_path:
                continue
            try:
                name = Path(seg.image_path).name.lower()
            except Exception:
                name = str(seg.image_path).lower()
            if name == target:
                count += 1
        if count > 1:
            self._set_status(
                f"Image '{Path(path).name}' is used {count} times in the timeline."
            )

    def _refresh_segment_item(self, seg_id: str) -> None:
        fps = max(1, self.fps_spin.value())
        for i in range(self.timeline_list.count()):
            item = self.timeline_list.item(i)
            if item.data(Qt.UserRole) == seg_id:
                seg = self.segments[i]
                if isinstance(item, SegmentItem):
                    item.update_from_segment(seg, fps)
                else:
                    new_item = SegmentItem(seg, fps)
                    self.timeline_list.takeItem(i)
                    self.timeline_list.insertItem(i, new_item)
                break

    def assign_selected_image_to_segment(self) -> None:
        seg = self._current_segment()
        if not seg:
            QtWidgets.QMessageBox.information(
                self, "Assign image", "Select a segment in the timeline first."
            )
            return
        path = self.image_library.current_image_path()
        if not path:
            QtWidgets.QMessageBox.information(
                self, "Assign image", "Select an image from the library."
            )
            return
        self._assign_image_to_segment_id(seg.id, path)

    def clear_segment_image(self) -> None:
        seg = self._current_segment()
        if not seg:
            return
        seg.image_path = ""
        if self.segment_kind_combo.currentText().lower() == "image":
            seg.kind = "image"
        self._refresh_segment_item(seg.id)
        self._load_segment_into_form(seg)

    def _clear_render_cache(self) -> None:
        if self._busy:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "Wait for the current render to finish before clearing the cache.",
            )
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear render cache",
            "This will delete cached thumbnails, normalized images, glow/impact variants, and waveform overlays.\n"
            "They will be regenerated on the next render.\n\nContinue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        cache_root = Path(_GLOBAL_CACHE_ROOT)
        try:
            if cache_root.exists():
                shutil.rmtree(cache_root, ignore_errors=True)
            cache_root.mkdir(parents=True, exist_ok=True)
            THUMB_CACHE.clear()
            self._set_status("Render cache cleared. It will rebuild on next run.")
            QtWidgets.QMessageBox.information(self, "Clear cache", "Cache cleared successfully.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Clear cache", f"Failed to clear cache: {exc}")

    def _on_segment_name_changed(self) -> None:
        if self._loading_segment_ui:
            return
        seg = self._current_segment()
        if not seg:
            return
        seg.name = self.segment_name_edit.text().strip()
        self._refresh_segment_item(seg.id)

    def _on_segment_duration_changed(self, value: float) -> None:
        if self._loading_segment_ui:
            return
        seg = self._current_segment()
        if not seg:
            return
        seg.duration = max(value, 0.01)
        self._recalculate_segment_timings()
        self._refresh_segment_item(seg.id)
        self._load_segment_into_form(seg)

    def _on_segment_kind_changed(self, index: int) -> None:
        if self._loading_segment_ui:
            return
        seg = self._current_segment()
        if not seg:
            return
        seg.kind = "image" if index == 0 else "black"
        if seg.kind != "image":
            seg.image_path = ""
        self._refresh_segment_item(seg.id)
        self._load_segment_into_form(seg)

    def split_selected_segment(self) -> None:
        seg = self._current_segment()
        if not seg:
            return
        fps = max(1, self.fps_spin.value())
        min_dur = 1.0 / fps
        max_split = seg.duration - min_dur
        if max_split <= min_dur:
            QtWidgets.QMessageBox.information(
                self,
                "Split segment",
                "Segment is too short to split further.",
            )
            return
        value, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Split segment",
            "Split after (seconds):",
            round(seg.duration / 2, 3),
            min_dur,
            max_split,
            3,
        )
        if not ok:
            return
        idx = self.segments.index(seg)
        seg.duration = value
        new_seg = Segment(
            name=f"{seg.name} (cont)",
            duration=max(min_dur, max_split - (value - min_dur)),
            kind=seg.kind,
            image_path=seg.image_path if seg.kind == "image" else "",
        )
        self.segments.insert(idx + 1, new_seg)
        self._refresh_timeline()
        self.timeline_list.setCurrentRow(idx + 1)

    def duplicate_selected_segment(self) -> None:
        seg = self._current_segment()
        if not seg:
            return
        idx = self.segments.index(seg)
        new_seg = Segment(
            name=f"{seg.name} (dup)",
            duration=seg.duration,
            kind=seg.kind,
            image_path=seg.image_path,
        )
        self.segments.insert(idx + 1, new_seg)
        self._refresh_timeline()
        self.timeline_list.setCurrentRow(idx + 1)

    def delete_selected_segments(self) -> None:
        ids = self._current_segment_ids()
        if not ids:
            return
        self.segments = [seg for seg in self.segments if seg.id not in ids]
        self._refresh_timeline(preserve_selection=False)

    def convert_selected_to_black(self) -> None:
        ids = self._current_segment_ids()
        for seg in self.segments:
            if seg.id in ids:
                seg.kind = "black"
                seg.image_path = ""
        self._refresh_timeline()

    def auto_fill_segments(self) -> None:
        if not self.segments:
            QtWidgets.QMessageBox.information(self, "Auto-fill", "Load markers first.")
            return
        if not self.images:
            QtWidgets.QMessageBox.information(self, "Auto-fill", "No images available.")
            return
        intervals, _ = timeline_to_intervals(self.segments, max(1, self.fps_spin.value()))
        prior = [seg.image_path for seg in self.segments]
        try:
            assigned = assign_images(
                intervals,
                self.images,
                loop=self.loop_images_check.isChecked(),
                prior_assign=prior,
            )
            for seg, path in zip(self.segments, assigned):
                if seg.kind == "image":
                    seg.image_path = path
            self._refresh_timeline()
            self._set_status("Auto-filled image assignments")
        except Exception as exc:  # noqa: broad-except
            QtWidgets.QMessageBox.critical(self, "Auto-fill error", str(exc))

    def _handle_timeline_order_changed(self) -> None:
        ordered_ids: List[str] = []
        for i in range(self.timeline_list.count()):
            ordered_ids.append(self.timeline_list.item(i).data(Qt.UserRole))
        mapping = {seg.id: seg for seg in self.segments}
        if len(ordered_ids) != len(self.segments):
            return
        self.segments = [mapping[sid] for sid in ordered_ids if sid in mapping]
        self._refresh_timeline()

    # --- Config serialization -------------------------------------------
    def _browse_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Resolve Markers CSV",
            self.csv_edit.text() or str(Path.home()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if path:
            self.csv_edit.setText(path)
            self.load_markers_from_csv(path)

    def _browse_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            self.audio_edit.text() or str(Path.home()),
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.aac *.ogg);;All Files (*.*)",
        )
        if path:
            self.audio_edit.setText(path)

    def _browse_quick_out(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select quick render output",
            self.quick_out_edit.text() or str(Path.home()),
            "MP4 Files (*.mp4);;All Files (*.*)",
        )
        if path:
            self.quick_out_edit.setText(path)

    def _browse_full_out(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select full export output",
            self.full_out_edit.text() or str(Path.home()),
            "MP4 Files (*.mp4);;All Files (*.*)",
        )
        if path:
            self.full_out_edit.setText(path)

    def _browse_ffmpeg(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ffmpeg executable",
            self.ffmpeg_path_edit.text() or str(Path.home()),
        )
        if path:
            self.ffmpeg_path_edit.setText(path)

    def _browse_ffprobe(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ffprobe executable",
            self.ffprobe_path_edit.text() or str(Path.home()),
        )
        if path:
            self.ffprobe_path_edit.setText(path)

    def _collect_config(self) -> Dict[str, object]:
        cfg = {
            "MARKERS_CSV": self.csv_edit.text().strip(),
            "AUDIO_PATH": self.audio_edit.text().strip(),
            "OUT_QUICK": self.quick_out_edit.text().strip(),
            "OUT_FULL": self.full_out_edit.text().strip(),
            "FFMPEG_PATH": self.ffmpeg_path_edit.text().strip() or "ffmpeg",
            "FFPROBE_PATH": self.ffprobe_path_edit.text().strip() or "ffprobe",
            "FPS": int(self.fps_spin.value()),
            "FRAME_W": int(self.frame_w_spin.value()),
            "FRAME_H": int(self.frame_h_spin.value()),
            "AUDIO_GAIN_DB": float(self.audio_gain_spin.value()),
            "AUDIO_OFFSET_SEC": float(self.audio_offset_spin.value()),
            "SNAP_MODE": self.snap_mode_combo.currentText(),
            "USE_NVENC": self.use_nvenc_check.isChecked(),
            "USE_SHORTEST": self.use_shortest_check.isChecked(),
            "PRE_NORMALIZE_IMAGES": self.pre_norm_check.isChecked(),
            "PRE_SAVE_FORMAT": self.pre_format_combo.currentText(),
            "PRE_JPEG_QUALITY": int(self.pre_quality_spin.value()),
            "QUICK_START_MODE": self.quick_mode_combo.currentText(),
            "QUICK_PREROLL_S": float(self.preroll_spin.value()),
            "QUICK_START_S": float(self.quick_start_spin.value()),
            "QUICK_START_INTERVAL_INDEX": int(self.quick_index_spin.value()),
            "QUICK_DURATION_S": float(self.quick_dur_spin.value()),
            "QUICK_N_INTERVALS": int(self.quick_n_spin.value()),
            "IMAGES_DIRS": list(self.image_dirs),
            "LOOP_IMAGES": self.loop_images_check.isChecked(),
            "IMPACT_KIND": self.impact_kind_combo.currentData() or "disabled",
            "IMPACT_DURATION_S": float(self.impact_duration_spin.value()),
            "IMPACT_ENABLED": self.impact_enable_check.isChecked(),
            "FX_GLOW_ENABLED": self.glow_enable_check.isChecked(),
            "FX_GLOW_COLOR": self.glow_color_combo.currentData() or "#00f6ff",
            "FX_GLOW_WIDTH": int(self.glow_width_spin.value()),
            "FX_GLOW_INTENSITY": float(self.glow_intensity_spin.value()),
            "FX_GLOW_PRESET": str(self.glow_preset_combo.currentData() or "legacy"),
            "WAVEFORM_ENABLED": self.waveform_enable_check.isChecked(),
            "WAVEFORM_STYLE": self.waveform_style_combo.currentData() or "oscilloscope",
            "WAVEFORM_COLOR": self.waveform_color_combo.currentData() or "#00f6ff",
            "WAVEFORM_HEIGHT_PCT": float(self.waveform_height_spin.value()),
            "WAVEFORM_OPACITY": float(self.waveform_opacity_spin.value()),
        }
        tc_text = self.tc_base_edit.text().strip()
        cfg["TIMECODE_BASE_S"] = None if not tc_text else float(tc_text)
        return cfg

    def _apply_config(self, cfg: Dict[str, object]) -> None:
        self.cfg.update(cfg)
        self.csv_edit.setText(str(cfg.get("MARKERS_CSV", "")))
        self.audio_edit.setText(str(cfg.get("AUDIO_PATH", "")))
        self.quick_out_edit.setText(str(cfg.get("OUT_QUICK", "")))
        self.full_out_edit.setText(str(cfg.get("OUT_FULL", "")))
        self.ffmpeg_path_edit.setText(str(cfg.get("FFMPEG_PATH", "ffmpeg")))
        self.ffprobe_path_edit.setText(str(cfg.get("FFPROBE_PATH", "ffprobe")))
        self.fps_spin.setValue(int(cfg.get("FPS", 25)))
        self.frame_w_spin.setValue(int(cfg.get("FRAME_W", 1920)))
        self.frame_h_spin.setValue(int(cfg.get("FRAME_H", 1080)))
        self.audio_gain_spin.setValue(float(cfg.get("AUDIO_GAIN_DB", 0.0)))
        self.audio_offset_spin.setValue(float(cfg.get("AUDIO_OFFSET_SEC", 0.0)))
        tc_val = cfg.get("TIMECODE_BASE_S", None)
        self.tc_base_edit.setText("" if tc_val is None else str(tc_val))
        snap = cfg.get("SNAP_MODE", "round")
        idx = self.snap_mode_combo.findText(str(snap))
        if idx >= 0:
            self.snap_mode_combo.setCurrentIndex(idx)
        self.use_nvenc_check.setChecked(bool(cfg.get("USE_NVENC", True)))
        self.use_shortest_check.setChecked(bool(cfg.get("USE_SHORTEST", False)))
        self.pre_norm_check.setChecked(bool(cfg.get("PRE_NORMALIZE_IMAGES", True)))
        fmt = str(cfg.get("PRE_SAVE_FORMAT", "JPEG"))
        idx = self.pre_format_combo.findText(fmt)
        if idx >= 0:
            self.pre_format_combo.setCurrentIndex(idx)
        self.pre_quality_spin.setValue(int(cfg.get("PRE_JPEG_QUALITY", 90)))
        quick_mode = str(cfg.get("QUICK_START_MODE", "first_marker"))
        idx = self.quick_mode_combo.findText(quick_mode)
        if idx >= 0:
            self.quick_mode_combo.setCurrentIndex(idx)
        self.preroll_spin.setValue(float(cfg.get("QUICK_PREROLL_S", 2.0)))
        self.quick_start_spin.setValue(float(cfg.get("QUICK_START_S", 0.0)))
        self.quick_index_spin.setValue(int(cfg.get("QUICK_START_INTERVAL_INDEX", 0)))
        self.quick_dur_spin.setValue(float(cfg.get("QUICK_DURATION_S", 20.0)))
        self.quick_n_spin.setValue(int(cfg.get("QUICK_N_INTERVALS", 10)))
        self.loop_images_check.setChecked(bool(cfg.get("LOOP_IMAGES", True)))
        glow_enabled = bool(cfg.get("FX_GLOW_ENABLED", DEFAULTS["FX_GLOW_ENABLED"]))
        self.glow_enable_check.setChecked(glow_enabled)
        color_value = str(cfg.get("FX_GLOW_COLOR", DEFAULTS["FX_GLOW_COLOR"]))
        idx = self.glow_color_combo.findData(color_value)
        if idx >= 0:
            self.glow_color_combo.setCurrentIndex(idx)
        preset_key = str(cfg.get("FX_GLOW_PRESET", DEFAULTS["FX_GLOW_PRESET"]))
        preset_idx = self.glow_preset_combo.findData(preset_key)
        if preset_idx < 0:
            preset_idx = self.glow_preset_combo.findData("legacy")
        if preset_idx >= 0:
            self.glow_preset_combo.setCurrentIndex(preset_idx)
        self.glow_width_spin.setValue(int(cfg.get("FX_GLOW_WIDTH", DEFAULTS["FX_GLOW_WIDTH"])))
        self.glow_intensity_spin.setValue(float(cfg.get("FX_GLOW_INTENSITY", DEFAULTS["FX_GLOW_INTENSITY"])))
        impact_kind = str(cfg.get("IMPACT_KIND", "disabled"))
        idx = self.impact_kind_combo.findData(impact_kind)
        if idx < 0:
            idx = self.impact_kind_combo.findData("disabled")
        if idx >= 0:
            self.impact_kind_combo.setCurrentIndex(idx)
        self.impact_duration_spin.setValue(float(cfg.get("IMPACT_DURATION_S", DEFAULTS["IMPACT_DURATION_S"])))
        self.impact_enable_check.setChecked(bool(cfg.get("IMPACT_ENABLED", DEFAULTS["IMPACT_ENABLED"])))
        wave_enabled = bool(cfg.get("WAVEFORM_ENABLED", DEFAULTS["WAVEFORM_ENABLED"]))
        self.waveform_enable_check.setChecked(wave_enabled)
        wave_style = str(cfg.get("WAVEFORM_STYLE", DEFAULTS.get("WAVEFORM_STYLE", "oscilloscope")))
        idx = self.waveform_style_combo.findData(wave_style)
        if idx >= 0:
            self.waveform_style_combo.setCurrentIndex(idx)
        wave_color = str(cfg.get("WAVEFORM_COLOR", DEFAULTS["WAVEFORM_COLOR"]))
        idx = self.waveform_color_combo.findData(wave_color)
        if idx >= 0:
            self.waveform_color_combo.setCurrentIndex(idx)
        self.waveform_height_spin.setValue(float(cfg.get("WAVEFORM_HEIGHT_PCT", DEFAULTS["WAVEFORM_HEIGHT_PCT"])))
        self.waveform_opacity_spin.setValue(float(cfg.get("WAVEFORM_OPACITY", DEFAULTS["WAVEFORM_OPACITY"])))
        self._update_glow_controls_enabled()
        self._update_impact_controls_enabled()
        self._update_waveform_controls_enabled()
        dirs = cfg.get("IMAGES_DIRS", [])
        if isinstance(dirs, list):
            self.image_dirs = [str(d) for d in dirs]

    # --- Project save/load ----------------------------------------------
    def _serialize_project(self) -> Dict[str, object]:
        return {
            "config": self._collect_config(),
            "segments": [seg.to_dict() for seg in self.segments],
            "marker_frames": self.marker_frames,
            "marker_frames_raw": self.marker_frames_raw,
            "image_dirs": list(self.image_dirs),
            "version": 2,
        }

    def _write_project(self, path: str) -> None:
        data = self._serialize_project()
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.current_project_path = path
        self.project_label.setText(f"Project: {path}")
        self._set_status(f"Saved project → {path}")

    def save_project(self) -> None:
        if self._busy:
            return
        if not self.current_project_path:
            self.save_project_as()
            return
        try:
            self._write_project(self.current_project_path)
        except Exception as exc:  # noqa: broad-except
            QtWidgets.QMessageBox.critical(self, "Save error", str(exc))

    def save_project_as(self) -> None:
        if self._busy:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save project",
            self.current_project_path or DEFAULTS["PROJECT_JSON"],
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not path:
            return
        try:
            self._write_project(path)
        except Exception as exc:  # noqa: broad-except
            QtWidgets.QMessageBox.critical(self, "Save error", str(exc))

    def load_project_dialog(self) -> None:
        if self._busy:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load project",
            self.current_project_path or DEFAULTS["PROJECT_JSON"],
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not path:
            return
        try:
            self.load_project_from_path(path)
        except Exception as exc:  # noqa: broad-except
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))

    def _segments_from_legacy(self, data: Dict[str, object], cfg: Dict[str, object]) -> List[Segment]:
        intervals = [tuple(x) for x in data.get("intervals", [])]
        assignments = data.get("image_assignments", [])
        fps = int(cfg.get("FPS", 25))
        segments: List[Segment] = []
        min_duration = 1.0 / max(1, fps)
        for idx, (a, b) in enumerate(intervals, start=1):
            dur = max(min_duration, (b - a) / fps)
            img = assignments[idx - 1] if idx - 1 < len(assignments) else ""
            kind = "image" if img else "black"
            segments.append(
                Segment(name=f"Segment {idx:03d}", duration=dur, kind=kind, image_path=img)
            )
        return segments

    def load_project_from_path(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        self._apply_config(cfg)
        marker_values = data.get("marker_frames", [])
        self.marker_frames = [max(0, int(x)) for x in marker_values]
        raw_values = data.get("marker_frames_raw")
        if raw_values is not None:
            self.marker_frames_raw = [int(x) for x in raw_values]
        else:
            self.marker_frames_raw = list(self.marker_frames)
        if "segments" in data:
            self.segments = [Segment.from_dict(d) for d in data["segments"]]
        else:
            self.segments = self._segments_from_legacy(data, cfg)
        self._normalize_segment_image_paths()
        self.image_dirs = [str(d) for d in data.get("image_dirs", cfg.get("IMAGES_DIRS", []))]
        self.current_project_path = path
        self.project_label.setText(f"Project: {path}")
        self._update_image_dirs_list()
        self.rescan_images()
        self._refresh_timeline(preserve_selection=False)
        self._set_status(f"Loaded project ← {path}")

    # --- Rendering -------------------------------------------------------
    def ensure_ready_to_render(self) -> bool:
        if not self.segments:
            QtWidgets.QMessageBox.information(self, "Render", "No segments to render.")
            return False
        audio = self.audio_edit.text().strip()
        exists, _ = self._path_exists_runtime(audio)
        if not audio or not exists:
            QtWidgets.QMessageBox.information(
                self, "Render", "Select a valid audio file first."
            )
            return False
        return True

    def _compute_tc_base(self) -> float:
        text = self.tc_base_edit.text().strip()
        if text:
            try:
                return float(text)
            except ValueError:
                pass
        source_frames = self.marker_frames_raw if self.marker_frames_raw else self.marker_frames
        if source_frames:
            return infer_timecode_base_seconds(source_frames, max(1, self.fps_spin.value()))
        return 0.0

    def _compute_quick_window(self) -> Tuple[float, float]:
        mode = self.quick_mode_combo.currentText().lower()
        fps = max(1, self.fps_spin.value())
        if mode == "first_marker":
            if self.marker_frames:
                fm_s = self.marker_frames[0] / fps
            else:
                fm_s = self.segments[0].start if self.segments else 0.0
            start = max(0.0, fm_s - max(0.0, self.preroll_spin.value()))
        elif mode == "absolute":
            start = max(0.0, self.quick_start_spin.value())
        elif mode == "interval_index":
            idx = self.quick_index_spin.value()
            if idx < 0 or idx >= len(self.segments):
                raise ValueError("Interval index out of range")
            start = self.segments[idx].start
        else:
            start = 0.0
        dur = self.quick_dur_spin.value()
        if dur <= 0.0:
            n = max(1, self.quick_n_spin.value())
            dur = sum(seg.duration for seg in self.segments[:n])
        dur = max(0.5, dur)
        return start, dur

    def start_quick_render(self) -> None:
        if self._busy:
            return
        if not self.ensure_ready_to_render():
            return
        out_path_input = self.quick_out_edit.text().strip() or str(Path.home() / "quick_preview.mp4")
        self.quick_out_edit.setText(out_path_input)
        out_path = self._to_runtime_path(out_path_input)
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        audio_input = self.audio_edit.text().strip()
        exists, audio_path = self._path_exists_runtime(audio_input)
        if not exists:
            QtWidgets.QMessageBox.critical(self, "Audio", "Audio file not accessible from WSL.")
            return
        ffmpeg_path_input = self.ffmpeg_path_edit.text().strip() or "ffmpeg"
        ffmpeg_path = self._to_runtime_path(ffmpeg_path_input)
        requested_nvenc = self.use_nvenc_check.isChecked()
        nvenc_enabled = requested_nvenc and self._nvenc_supported(ffmpeg_path)
        if requested_nvenc and not nvenc_enabled:
            self._set_status("NVENC requested for quick render but unavailable; using CPU.")
        fps = max(1, self.fps_spin.value())
        frame_size = (int(self.frame_w_spin.value()), int(self.frame_h_spin.value()))
        try:
            start, dur = self._compute_quick_window()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Quick window", str(exc))
            return
        segments = timeline_to_render_segments(self.segments)
        impact_kind = self._selected_impact_kind()
        impact_duration = float(self.impact_duration_spin.value()) if self.impact_enable_check.isChecked() else 0.0
        glow_opts = self._current_glow_options()
        wave_opts = self._current_waveform_options()
        task_args = (
            out_path,
            ffmpeg_path,
            fps,
            frame_size,
            audio_path,
            float(self.audio_gain_spin.value()),
            segments,
            start,
            dur,
            self._compute_tc_base(),
            float(self.audio_offset_spin.value()),
            self.snap_mode_combo.currentText(),
            str(impact_kind),
            impact_duration,
            glow_opts,
            wave_opts,
        )
        self._start_worker(render_quick_windowed, *task_args)
        self._set_status("Rendering quick preview…")

    def start_full_export(self) -> None:
        if self._busy:
            return
        if not self.ensure_ready_to_render():
            return
        out_path_input = self.full_out_edit.text().strip() or str(Path.home() / "final_export.mp4")
        self.full_out_edit.setText(out_path_input)
        out_path = self._to_runtime_path(out_path_input)
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        audio_input = self.audio_edit.text().strip()
        exists, audio_path = self._path_exists_runtime(audio_input)
        if not exists:
            QtWidgets.QMessageBox.critical(self, "Audio", "Audio file not accessible from WSL.")
            return
        fps = max(1, self.fps_spin.value())
        frame_size = (int(self.frame_w_spin.value()), int(self.frame_h_spin.value()))
        intervals, assignments = timeline_to_intervals(self.segments, fps)
        if self.marker_frames_raw:
            first_marker = self.marker_frames_raw[0]
        else:
            first_marker = self.marker_frames[0] if self.marker_frames else 0
        ffmpeg_path_input = self.ffmpeg_path_edit.text().strip() or "ffmpeg"
        ffprobe_path_input = self.ffprobe_path_edit.text().strip() or "ffprobe"
        ffmpeg_path = self._to_runtime_path(ffmpeg_path_input)
        ffprobe_path = self._to_runtime_path(ffprobe_path_input)
        requested_nvenc = self.use_nvenc_check.isChecked()
        nvenc_enabled = requested_nvenc and self._nvenc_supported(ffmpeg_path)
        if requested_nvenc and not nvenc_enabled:
            self._set_status("NVENC requested but encoder unavailable; falling back to CPU.")
        impact_kind = self._selected_impact_kind()
        impact_duration = float(self.impact_duration_spin.value()) if self.impact_enable_check.isChecked() else 0.0
        glow_opts = self._current_glow_options()
        wave_opts = self._current_waveform_options()
        pre_normalize = self.pre_norm_check.isChecked()
        task_args = (
            ffmpeg_path,
            ffprobe_path,
            out_path,
            fps,
            frame_size,
            audio_path,
            float(self.audio_gain_spin.value()),
            nvenc_enabled,
            self.use_shortest_check.isChecked(),
            pre_normalize,
            self.pre_format_combo.currentText(),
            int(self.pre_quality_spin.value()),
            intervals,
            assignments,
            first_marker,
            self._compute_tc_base(),
            str(impact_kind),
            impact_duration,
            glow_opts,
            wave_opts,
        )
        self._start_worker(ffmpeg_full_export, *task_args)
        self._set_status("Exporting full video…")

    def _start_worker(self, func, *args, **kwargs) -> None:
        if self.worker_thread and self.worker_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Busy", "A render is already running."
            )
            return
        self._set_busy(True)
        self._clear_progress()
        self.worker = RenderWorker(func, *args, **kwargs)
        self.worker_thread = QtCore.QThread(self)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.status.connect(self._on_worker_status)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _cancel_active_render(self, force: bool = False) -> None:
        if not self.worker_thread:
            return
        self._set_status("Stopping render…")
        if self.worker:
            self.worker.cancel()
        if self.worker_thread.isRunning():
            self.worker_thread.requestInterruption()
            self.worker_thread.quit()
            if force and not self.worker_thread.wait(1500):
                self.worker_thread.terminate()
                self.worker_thread.wait()
        self._cleanup_worker()
        self._set_busy(False)

    def _on_worker_progress(self, value: float) -> None:
        self._update_progress(value)

    def _on_worker_status(self, message: str) -> None:
        self._set_status(message)

    def _on_worker_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Render error", message)
        self._set_status("Error during render")
        self._set_busy(False)

    def _on_worker_finished(self) -> None:
        self._set_status("Render finished")
        self._set_busy(False)
        self._update_progress(1.0)

    def _cleanup_worker(self) -> None:
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: D401
        super().showEvent(event)
        if not self._cursor_guard.isActive():
            self._cursor_guard.start()
        self._guard_system_cursor()

    def enterEvent(self, event: QtGui.QEnterEvent) -> None:  # noqa: D401
        super().enterEvent(event)
        self._guard_system_cursor()

    def _guard_system_cursor(self) -> None:
        try:
            cursor = QtGui.QGuiApplication.overrideCursor()
        except RuntimeError:
            cursor = None
        while cursor is not None:
            QtGui.QGuiApplication.restoreOverrideCursor()
            try:
                cursor = QtGui.QGuiApplication.overrideCursor()
            except RuntimeError:
                cursor = None
        if self.cursor().shape() == Qt.BlankCursor:
            self.unsetCursor()
            if CURSOR_DEBUG:
                print("[CURSOR] BlankCursor detected → unsetCursor() invoked")
        elif CURSOR_DEBUG:
            print("[CURSOR] Guard run; platform:", QtGui.QGuiApplication.platformName(), "cursor shape:", self.cursor().shape())

    def closeEvent(self, event):  # noqa: D401 - Qt override
        if self._busy:
            res = QtWidgets.QMessageBox.question(
                self,
                "Render in progress",
                "A render/export is currently running.\nStop it and exit?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if res != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self._cancel_active_render(force=True)
        if self._image_scan_thread and self._image_scan_thread.isRunning():
            self._image_scan_thread.quit()
            self._image_scan_thread.wait()
        self._cleanup_image_scan()
        super().closeEvent(event)


def main() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "xcb"))
    print(f"[Timeline] QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM')}")
    app = QtWidgets.QApplication([])
    try:
        print(f"[Timeline] Qt platform plugin: {QtGui.QGuiApplication.platformName()}")
    except Exception:
        pass
    install_app_style(app)
    window = TimelineApp()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

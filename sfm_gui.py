"""
sfm_gui.py
==========
SFMformer Super-Resolution desktop GUI (PyQt6).

Runs on Raspberry Pi 5 (CPU, FP32) or any desktop with PyTorch.
- Upload a local image and super-resolve it (x2/x3/x4).
- Run standard benchmarks (Set5/Set14/B100/Urban100/Manga109), browse each
  image and see its PSNR-Y / SSIM-Y against the HR ground truth.
- Live device stats: CPU temp (Pi only), CPU %, RAM %.
- Synchronised zoom on the LR/SR comparison (mouse wheel + slider).
- Save the SR result as PNG.

Launch:
    cd ~/sfmformer-sr-deploy
    python sfm_gui.py

Designed for the 7" official Pi screen (1024x600), light theme, English only.
"""
from __future__ import annotations

import sys
import time
import platform
import subprocess
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox, QProgressBar,
    QFileDialog, QSlider, QScrollArea, QSizePolicy, QFrame, QButtonGroup,
    QStackedWidget, QListWidget, QListWidgetItem,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# =============================================================================
# Model config + checkpoint map (mirrors test_pi.py)
# =============================================================================
SFMFORMER_LIGHT_CONFIG = dict(
    in_chans=3, img_size=64, embed_dim=52,
    depths=[2, 4, 6, 6, 6], num_heads=4,
    num_topk=[1024, 1024, 256, 256, 256, 256,
              128, 128, 128, 128, 128, 128,
              64, 64, 64, 64, 64, 64,
              32, 32, 32, 32, 32, 32],
    window_size=32, convffn_kernel_size=7, img_range=1.0,
    mlp_ratio=1, upsampler='pixelshuffledirect',
    resi_connection='1conv', use_checkpoint=False, use_ups=True,
)
CHECKPOINT_MAP = {
    2: '101_SFMformer_SRx2_scratch.pth',
    3: '102_SFMformer_SRx3_finetune.pth',
    4: '103_SFMformer_SRx4_finetune.pth',
}
BENCHMARKS = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']

IS_PI = 'arm' in platform.machine().lower() or 'aarch64' in platform.machine().lower()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Theming: follow the OS light/dark preference so the UI doesn't clash with the
# system colours (e.g. a dark Windows theme makes native popups dark while a
# hard-coded light stylesheet stays light). One Fusion palette + a matching
# stylesheet, chosen from the detected colour scheme. Works the same on the Pi.
# =============================================================================
def is_dark_mode(app) -> bool:
    """True if the OS prefers a dark colour scheme. Falls back to the default
    palette's luminance on Qt versions without colorScheme()."""
    try:
        from PyQt6.QtCore import Qt as _Qt
        scheme = app.styleHints().colorScheme()
        if scheme == _Qt.ColorScheme.Dark:
            return True
        if scheme == _Qt.ColorScheme.Light:
            return False
    except Exception:
        pass
    try:
        from PyQt6.QtGui import QPalette as _QP
        return app.palette().color(_QP.ColorRole.Window).lightness() < 128
    except Exception:
        return False


def theme_colors(dark: bool) -> dict:
    """Palette of colours for the current theme. 'accent' (green) is chosen to
    stay readable on both backgrounds."""
    if dark:
        return dict(
            bg='#202124', panel='#2a2b2e', field='#1c1d20', text='#e6e6e6',
            muted='#9aa0a6', border='#3c3f44', btn='#303134',
            btn_hover='#3a414d', hover_border='#5a7fbf',
            checked_bg='#e8e8e8', checked_fg='#202124',
            disabled_fg='#6b6f76', disabled_bg='#26272a', accent='#37b24d',
        )
    return dict(
        bg='#f5f5f5', panel='#ffffff', field='#ffffff', text='#222222',
        muted='#555555', border='#dddddd', btn='#ffffff',
        btn_hover='#eef4ff', hover_border='#6aa0ff',
        checked_bg='#222222', checked_fg='#ffffff',
        disabled_fg='#aaaaaa', disabled_bg='#f0f0f0', accent='#2a8c4a',
    )


def make_palette(dark: bool):
    from PyQt6.QtGui import QPalette, QColor
    c = theme_colors(dark)
    p = QPalette()
    R, G = QPalette.ColorRole, QPalette.ColorGroup
    p.setColor(R.Window, QColor(c['bg']))
    p.setColor(R.WindowText, QColor(c['text']))
    p.setColor(R.Base, QColor(c['field']))
    p.setColor(R.AlternateBase, QColor(c['panel']))
    p.setColor(R.Text, QColor(c['text']))
    p.setColor(R.Button, QColor(c['btn']))
    p.setColor(R.ButtonText, QColor(c['text']))
    p.setColor(R.ToolTipBase, QColor(c['panel']))
    p.setColor(R.ToolTipText, QColor(c['text']))
    p.setColor(R.Highlight, QColor(c['hover_border']))
    p.setColor(R.HighlightedText, QColor('#ffffff'))
    p.setColor(R.PlaceholderText, QColor(c['muted']))
    for role in (R.Text, R.ButtonText, R.WindowText):
        p.setColor(G.Disabled, role, QColor(c['disabled_fg']))
    return p


def build_stylesheet(c: dict) -> str:
    return f'''
        QMainWindow {{ background: {c['bg']}; }}
        QWidget {{ font-size: 12px; color: {c['text']}; }}
        QGroupBox {{
            border: 1px solid {c['border']}; border-radius: 6px;
            margin-top: 8px; padding: 6px; background: {c['panel']};
            font-weight: 600;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
        QPushButton {{
            background: {c['btn']}; border: 1px solid {c['border']};
            border-radius: 5px; padding: 6px; color: {c['text']};
        }}
        QPushButton:hover {{ background: {c['btn_hover']}; border-color: {c['hover_border']}; }}
        QPushButton:checked {{ background: {c['checked_bg']}; color: {c['checked_fg']}; border-color:{c['checked_bg']}; }}
        QPushButton:disabled {{ color:{c['disabled_fg']}; background:{c['disabled_bg']}; }}
        QProgressBar {{
            border: 1px solid {c['border']}; border-radius: 4px; text-align: center;
            height: 16px; background:{c['field']}; color:{c['text']};
        }}
        QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}
        QScrollArea {{ border: 1px solid {c['border']}; border-radius: 6px; background:{c['field']}; }}
        QComboBox {{ padding: 4px; border:1px solid {c['border']}; border-radius:4px;
                     background:{c['field']}; color:{c['text']}; }}
        QComboBox QAbstractItemView {{ background:{c['panel']}; color:{c['text']};
                     selection-background-color:{c['hover_border']}; selection-color:#ffffff;
                     border:1px solid {c['border']}; }}
        QListWidget {{ background:{c['field']}; color:{c['text']};
                       border:1px solid {c['border']}; border-radius:4px; }}
        QLabel {{ background: transparent; }}
    '''


# =============================================================================
# Image <-> tensor + metrics (ported from test_pi.py)
# =============================================================================
def img2tensor(rgb: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(DEVICE)


def tensor2img(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0).round().astype(np.uint8)


def to_y_channel(rgb: np.ndarray) -> np.ndarray:
    img = rgb.astype(np.float64) / 255.0
    return np.dot(img, [65.481, 128.553, 24.966]) + 16.0


def calculate_psnr(img, img2, crop_border, test_y=False):
    img = img.astype(np.float64); img2 = img2.astype(np.float64)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y:
        img = to_y_channel(img.astype(np.uint8))
        img2 = to_y_channel(img2.astype(np.uint8))
    mse = np.mean((img - img2) ** 2)
    return float('inf') if mse < 1e-10 else 20.0 * np.log10(255.0 / np.sqrt(mse))


def _ssim_1ch(img, img2):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(k, k.transpose())
    img = img.astype(np.float64); img2 = img2.astype(np.float64)
    mu1 = cv2.filter2D(img, -1, win)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, win)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = cv2.filter2D(img ** 2, -1, win)[5:-5, 5:-5] - mu1_sq
    s2 = cv2.filter2D(img2 ** 2, -1, win)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(img * img2, -1, win)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean()


def calculate_ssim(img, img2, crop_border, test_y=False):
    img = img.astype(np.float64); img2 = img2.astype(np.float64)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y:
        return _ssim_1ch(to_y_channel(img.astype(np.uint8)),
                         to_y_channel(img2.astype(np.uint8)))
    return float(np.mean([_ssim_1ch(img[..., c], img2[..., c])
                          for c in range(img.shape[2])]))


@torch.no_grad()
def patchwise_test(model, lq, scale, progress_cb=None):
    _, C, h, w = lq.size()
    sth, stw = h // 256 + 1, w // 256 + 1
    mph = (sth - h % sth) % sth
    mpw = (stw - w % stw) % stw
    img = F.pad(lq, (0, mpw, 0, mph), mode='reflect')
    _, _, H, W = img.size()
    sh, sw = H // sth, W // stw
    shh, shw = sh // 10, sw // 10
    ral, row = H // sh, W // sw

    slices = []
    for i in range(ral):
        for j in range(row):
            if i == 0 and i == ral - 1:
                top = slice(i * sh, (i + 1) * sh)
            elif i == 0:
                top = slice(i * sh, (i + 1) * sh + shh)
            elif i == ral - 1:
                top = slice(i * sh - shh, (i + 1) * sh)
            else:
                top = slice(i * sh - shh, (i + 1) * sh + shh)
            if j == 0 and j == row - 1:
                left = slice(j * sw, (j + 1) * sw)
            elif j == 0:
                left = slice(j * sw, (j + 1) * sw + shw)
            elif j == row - 1:
                left = slice(j * sw - shw, (j + 1) * sw)
            else:
                left = slice(j * sw - shw, (j + 1) * sw + shw)
            slices.append((top, left))

    # Run the tiles one at a time so we can report per-tile progress. This is
    # functionally identical to the old list comprehension but lets the GUI
    # show "Processing tile k/N" on big images (which can take many minutes),
    # so the user knows it is working rather than frozen.
    total_tiles = len(slices)
    outputs = []
    for k, (t, l) in enumerate(slices):
        outputs.append(model(img[..., t, l]))
        if progress_cb is not None:
            progress_cb(k + 1, total_tiles)
    out = torch.zeros(1, C, H * scale, W * scale, dtype=img.dtype, device=img.device)
    for i in range(ral):
        for j in range(row):
            top = slice(i * sh * scale, (i + 1) * sh * scale)
            left = slice(j * sw * scale, (j + 1) * sw * scale)
            _top = slice(0, sh * scale) if i == 0 else slice(shh * scale, (shh + sh) * scale)
            _left = slice(0, sw * scale) if j == 0 else slice(shw * scale, (shw + sw) * scale)
            out[..., top, left] = outputs[i * row + j][..., _top, _left]
    out = out[:, :, : H * scale - mph * scale, : W * scale - mpw * scale]
    # CUDA kernels are async; make sure all GPU work is finished before the
    # caller reads the elapsed time, otherwise latency is under-measured.
    if out.is_cuda:
        torch.cuda.synchronize()
    return out


def load_model(scale: int):
    from basicsr.archs.sfmformer_arch import SFMformer
    model = SFMformer(upscale=scale, **SFMFORMER_LIGHT_CONFIG)
    ckpt = ROOT / 'experiments' / 'pretrained_models' / CHECKPOINT_MAP[scale]
    raw = torch.load(str(ckpt), map_location='cpu', weights_only=False)
    sd = raw
    if isinstance(raw, dict):
        for key in ('params_ema', 'params', 'state_dict', 'model'):
            if key in raw:
                sd = raw[key]
                break
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(DEVICE)


def read_temp() -> str:
    """Pi CPU temperature via vcgencmd; '--' elsewhere."""
    if not IS_PI:
        return '--'
    try:
        out = subprocess.check_output(['vcgencmd', 'measure_temp'], timeout=2).decode()
        return out.strip().replace('temp=', '').replace("'C", '°C')
    except Exception:
        return '--'


# =============================================================================
# Inference worker thread (keeps UI responsive)
# =============================================================================
class InferenceWorker(QThread):
    progress = pyqtSignal(int, int)            # done, total
    tile_progress = pyqtSignal(int, int)       # done_tiles, total_tiles
    one_done = pyqtSignal(str, object, object, object, float, float, float)
    # name, lr_rgb, sr_rgb, hr_rgb_or_None, elapsed_s, psnr_y, ssim_y
    finished_all = pyqtSignal(float)           # avg time
    error = pyqtSignal(str)

    def __init__(self, scale, jobs):
        # jobs: list of (name, lr_path, hr_path_or_None)
        super().__init__()
        self.scale = scale
        self.jobs = jobs
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            torch.set_num_threads(4)
            model = load_model(self.scale)
            times = []
            total = len(self.jobs)
            for i, (name, lr_path, hr_path) in enumerate(self.jobs):
                if self._stop:
                    break
                lr_rgb = np.array(Image.open(lr_path).convert('RGB'))
                lr_t = img2tensor(lr_rgb)
                t0 = time.perf_counter()
                sr_t = patchwise_test(
                    model, lr_t, self.scale,
                    progress_cb=lambda d, tt: self.tile_progress.emit(d, tt))
                dt = time.perf_counter() - t0
                times.append(dt)
                sr_rgb = tensor2img(sr_t)

                hr_rgb, psnr_y, ssim_y = None, -1.0, -1.0
                if hr_path is not None and Path(hr_path).exists():
                    hr_rgb = np.array(Image.open(hr_path).convert('RGB'))
                    hh = min(sr_rgb.shape[0], hr_rgb.shape[0])
                    ww = min(sr_rgb.shape[1], hr_rgb.shape[1])
                    sr_c, hr_c = sr_rgb[:hh, :ww], hr_rgb[:hh, :ww]
                    psnr_y = calculate_psnr(sr_c, hr_c, self.scale, True)
                    ssim_y = calculate_ssim(sr_c, hr_c, self.scale, True)

                self.one_done.emit(name, lr_rgb, sr_rgb, hr_rgb, dt, psnr_y, ssim_y)
                self.progress.emit(i + 1, total)

            avg = float(np.mean(times)) if times else 0.0
            self.finished_all.emit(avg)
        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n{traceback.format_exc()[:500]}')


# =============================================================================
# Crop worker: super-resolve a single user-selected region (from an in-memory
# LR array) rather than a whole image from disk. Used by the Crop & SR view so
# the user can SR just a small region in seconds instead of waiting minutes on
# a full large image.
# =============================================================================
class CropWorker(QThread):
    done = pyqtSignal(object, object, float)   # crop_box, sr_rgb, elapsed_s
    tile_progress = pyqtSignal(int, int)
    error = pyqtSignal(str)

    def __init__(self, scale, lr_full, crop_box):
        super().__init__()
        self.scale = scale
        self.lr_full = lr_full        # full LR rgb (np.ndarray)
        self.crop_box = crop_box      # (x, y, w, h) in LR pixels

    def run(self):
        try:
            torch.set_num_threads(4)
            model = load_model(self.scale)
            x, y, w, h = self.crop_box
            crop = np.ascontiguousarray(self.lr_full[y:y + h, x:x + w])
            lr_t = img2tensor(crop)
            t0 = time.perf_counter()
            sr_t = patchwise_test(
                model, lr_t, self.scale,
                progress_cb=lambda d, tt: self.tile_progress.emit(d, tt))
            dt = time.perf_counter() - t0
            sr_rgb = tensor2img(sr_t)
            self.done.emit(self.crop_box, sr_rgb, dt)
        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n{traceback.format_exc()[:500]}')
# =============================================================================
# Why "physical zoom"? LR and SR have different pixel grids (e.g. 256x256 vs
# 1024x1024), so naively broadcasting the same `scale` between them shows
# misaligned regions: scrolling to display-pixel (100, 100) lands on LR
# pixel (100, 100) but SR pixel (100, 100), which are different physical
# points in the underlying scene.
#
# We model a "physical zoom" z = screen pixels per 1 SR pixel.
#   - SR view's display scale = z          (1:1 at z=1)
#   - LR view's display scale = z * ratio  (ratio = SR/LR, e.g. 4)
# With this, the two views always have identical display dimensions
# (lw*z*ratio == sw*z), so their scroll bars can share the same pixel
# coordinate and end up showing the SAME physical region. Naturally, LR
# pixels are then smoothly stretched (looks like Bicubic), while SR shows
# the model's actual detail at native resolution.
# =============================================================================
class ZoomView(QScrollArea):
    """Scrollable image view linked to a partner so the same physical region
    of the underlying image is shown at the same on-screen size in both
    views. Wheel = zoom (centred on cursor). Drag = pan."""

    # Emits the new zoom percentage when the user changes zoom via wheel.
    # External widgets (e.g. the zoom slider) connect to this to stay in sync.
    zoom_changed = pyqtSignal(int)

    def __init__(self, title: str):
        super().__init__()
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel('—')
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet('color:#888;')
        self.setWidget(self._label)
        self._pix_full: QPixmap | None = None
        self._ratio = 1.0          # how many SR pixels equal 1 of MY image pixels
        self._zoom = 1.0           # physical zoom: screen px per SR px
        self._partners = []
        self._title = title
        self._panning = False
        self._pan_start = QPoint()
        self.setMinimumSize(300, 360)

    # ---- linking / configuration ----
    def link(self, other):
        if other not in self._partners:
            self._partners.append(other)

    def set_ratio(self, ratio: float):
        """Set how many SR pixels equal one of this view's image pixels.
        SR view -> 1.0; LR view -> SR/LR scale factor (e.g. 4)."""
        self._ratio = float(ratio) if ratio and ratio > 0 else 1.0
        self._apply(broadcast=False)

    # ---- image lifecycle ----
    def set_image(self, rgb: np.ndarray):
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pix_full = QPixmap.fromImage(qimg.copy())
        # Don't auto-fit here; the MainWindow calls _fit_zoom() to sync both
        # views together after ratios are set on both sides.
        self._apply(broadcast=False)

    def clear_image(self):
        self._pix_full = None
        self._label.setPixmap(QPixmap())
        self._label.setText('—')
        self._label.adjustSize()

    # ---- display ----
    @property
    def display_scale(self) -> float:
        """Per-image-pixel screen scale = zoom * ratio."""
        return self._zoom * self._ratio

    def _apply(self, broadcast: bool = True):
        if self._pix_full is None:
            return
        s = self.display_scale
        w = max(1, int(round(self._pix_full.width() * s)))
        h = max(1, int(round(self._pix_full.height() * s)))
        scaled = self._pix_full.scaled(
            w, h, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())
        if broadcast:
            hv = self.horizontalScrollBar().value()
            vv = self.verticalScrollBar().value()
            for p in self._partners:
                p.sync(self._zoom, hv, vv)

    def sync(self, zoom: float, hval: int, vval: int):
        """Receive zoom + scroll state from a partner."""
        self._zoom = max(0.05, min(8.0, float(zoom)))
        self._apply(broadcast=False)
        self.horizontalScrollBar().setValue(hval)
        self.verticalScrollBar().setValue(vval)

    # ---- public API kept compatible with the old class ----
    def set_scale_pct(self, pct: int, broadcast: bool = True):
        """Now sets the *physical* zoom (% of native SR scale)."""
        self._zoom = max(0.05, min(8.0, pct / 100.0))
        self._apply(broadcast=broadcast)

    def current_pct(self) -> int:
        return int(round(self._zoom * 100))

    # ---- mouse wheel zoom, centred on cursor ----
    def wheelEvent(self, e: QWheelEvent):
        if self._pix_full is None:
            e.ignore(); return
        factor = 1.15 if e.angleDelta().y() > 0 else 1.0 / 1.15
        old_zoom = self._zoom
        new_zoom = max(0.05, min(8.0, old_zoom * factor))
        if new_zoom == old_zoom:
            e.accept(); return

        # Keep the image point under the cursor pinned during zoom.
        pos = e.position()
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        old_s = max(old_zoom * self._ratio, 1e-6)
        new_s = new_zoom * self._ratio
        img_x = (hbar.value() + pos.x()) / old_s
        img_y = (vbar.value() + pos.y()) / old_s
        self._zoom = new_zoom
        self._apply(broadcast=False)
        hbar.setValue(int(round(img_x * new_s - pos.x())))
        vbar.setValue(int(round(img_y * new_s - pos.y())))
        # Broadcast unified state to partners (they share the same pixel
        # scroll coords because their display sizes match).
        for p in self._partners:
            p.sync(self._zoom, hbar.value(), vbar.value())
        # Notify external listeners (slider) that the user-driven zoom changed.
        self.zoom_changed.emit(int(round(self._zoom * 100)))
        e.accept()

    # ---- drag to pan ----
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_start = e.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if not self._panning:
            return
        delta = e.pos() - self._pan_start
        self._pan_start = e.pos()
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        hbar.setValue(hbar.value() - delta.x())
        vbar.setValue(vbar.value() - delta.y())
        for p in self._partners:
            p.horizontalScrollBar().setValue(hbar.value())
            p.verticalScrollBar().setValue(vbar.value())

    def mouseReleaseEvent(self, e):
        self._panning = False
        self.setCursor(Qt.CursorShape.ArrowCursor)


# =============================================================================
# Magnifier view: shows the SR image as the base; hovering shows a SINGLE
# loupe that magnifies the SR region under the cursor (LR is no longer shown).
# Mouse wheel adjusts the loupe magnification, so you can trade off "how
# zoomed in" against "how much structure fits in the loupe window" -- a lower
# factor packs more SR pixels into the loupe so complete structures stay
# visible instead of a handful of giant pixels.
# =============================================================================
class MagnifierView(QLabel):
    LOUPE = 280              # loupe window size (px on screen)
    ZOOM = 2.5               # fixed magnification (screen px per SR px);
                             # lower than the old 4x so more SR pixels (=more
                             # complete structure) fit inside the loupe window

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setMinimumSize(360, 380)
        self.setStyleSheet('color:#888;')
        self.setText('Run inference, then hover here')
        self._sr = None          # SR rgb (sH,sW,3)
        self._disp_pix = None    # the fitted SR pixmap actually shown
        self._disp_rect = None   # QRect where the pixmap sits inside the label
        self._mouse = None       # current mouse pos (in label coords)

    def set_pair(self, lr_rgb, sr_rgb, scale):
        # Signature kept for backward compatibility with MainWindow; the LR
        # argument is intentionally ignored now (SR-only magnifier).
        self._sr = np.ascontiguousarray(sr_rgb)
        self._mouse = None
        self._rebuild_base()

    def clear_pair(self):
        self._sr = None
        self._disp_pix = None
        self.setText('Run inference, then hover here')
        self.update()

    def resizeEvent(self, e):
        self._rebuild_base()
        super().resizeEvent(e)

    def _rebuild_base(self):
        if self._sr is None:
            return
        h, w, _ = self._sr.shape
        qimg = QImage(self._sr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        aw = max(self.width() - 8, 1)
        ah = max(self.height() - 8, 1)
        self._disp_pix = pix.scaled(
            aw, ah, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        x = (self.width() - self._disp_pix.width()) // 2
        y = (self.height() - self._disp_pix.height()) // 2
        from PyQt6.QtCore import QRect
        self._disp_rect = QRect(x, y, self._disp_pix.width(),
                                self._disp_pix.height())
        self.setText('')
        self.update()

    def mouseMoveEvent(self, e):
        self._mouse = e.pos()
        self.update()

    def leaveEvent(self, e):
        self._mouse = None
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._disp_pix is None or self._disp_rect is None:
            return
        p = QPainter(self)
        p.drawPixmap(self._disp_rect.topLeft(), self._disp_pix)

        if self._mouse is None or not self._disp_rect.contains(self._mouse):
            p.end()
            return

        from PyQt6.QtCore import QRect
        from PyQt6.QtGui import QPen, QColor, QFont

        # mouse -> normalized [0,1] over the displayed SR image
        rx = (self._mouse.x() - self._disp_rect.x()) / self._disp_rect.width()
        ry = (self._mouse.y() - self._disp_rect.y()) / self._disp_rect.height()
        rx = min(max(rx, 0.0), 1.0)
        ry = min(max(ry, 0.0), 1.0)

        sh, sw, _ = self._sr.shape

        # Field of view in SR pixels, centred on the cursor.
        # LOUPE px on screen show fov_sr SR-pixels => magnification = LOUPE/fov_sr.
        pane = self.LOUPE
        fov_sr = max(8, int(pane / self.ZOOM))   # SR pixels visible in loupe
        cx_s, cy_s = rx * sw, ry * sh
        sx0 = int(min(max(cx_s - fov_sr / 2, 0), max(sw - fov_sr, 0)))
        sy0 = int(min(max(cy_s - fov_sr / 2, 0), max(sh - fov_sr, 0)))
        sx1 = min(sx0 + fov_sr, sw)
        sy1 = min(sy0 + fov_sr, sh)
        sr_crop = np.ascontiguousarray(self._sr[sy0:sy1, sx0:sx1])

        hh, ww, _ = sr_crop.shape
        qi = QImage(sr_crop.data, ww, hh, 3 * ww, QImage.Format.Format_RGB888)
        sr_pix = QPixmap.fromImage(qi.copy()).scaled(
            pane, pane, Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

        # Single loupe pane positioned near the cursor, clamped to the widget.
        gx = min(max(self._mouse.x() - pane // 2, 0), self.width() - pane)
        gy = min(max(self._mouse.y() - pane // 2, 0), self.height() - pane)
        p.drawPixmap(gx, gy, sr_pix)

        # outer border
        p.setPen(QPen(QColor('#222222'), 2))
        p.drawRect(QRect(gx, gy, pane, pane))

        # crosshair on the base image at the sampled point
        p.setPen(QPen(QColor(255, 60, 60), 1))
        p.drawLine(self._mouse.x() - 6, self._mouse.y(),
                   self._mouse.x() + 6, self._mouse.y())
        p.drawLine(self._mouse.x(), self._mouse.y() - 6,
                   self._mouse.x(), self._mouse.y() + 6)

        # 'SR' label with a subtle drop-shadow for readability
        f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        tag = 'SR'
        p.setPen(QColor(0, 0, 0)); p.drawText(gx + 7, gy + 18, tag)
        p.setPen(QColor(255, 255, 255)); p.drawText(gx + 6, gy + 17, tag)
        p.end()


# =============================================================================
# Crop & SR view: load an image, drag a rubber-band box to choose a region,
# then super-resolve just that region (the SR run is triggered by the toolbar
# button wired up in MainWindow). After SR the main area shows the region's
# SR output, and a bottom-left thumbnail shows the full source image with the
# selected box marked, so you know where the region came from.
# =============================================================================
class CropSRView(QWidget):
    THUMB_MAX = 110          # context thumbnail max edge (screen px)
    MARGIN = 12
    CAPTION_H = 18           # reserved height for the caption under the thumb
    ZOOM_MAX = 4.0           # max zoom for the SR result (pixel-level look)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(360, 380)
        self.setMouseTracking(True)
        self._lr = None          # full source (LR) rgb
        self._lr_pix = None      # source fitted to widget (select mode)
        self._lr_rect = None
        self._sel_start = None   # rubber-band start (widget coords)
        self._sel_cur = None     # rubber-band current (widget coords)
        self._dragging = False   # rubber-band drag (select mode)
        self._sr = None          # SR of the cropped region
        self._base_pix = None    # SR at native resolution (result mode)
        self._disp_pix = None    # SR scaled by current zoom (result mode)
        self._zoom = 1.0         # display scale (screen px per SR px)
        self._zoom_min = 1.0     # = "fit whole SR" scale (<= 1.0)
        self._offset = QPoint(0, 0)
        self._pan_last = None    # last pos while panning (result mode)
        self._crop_box = None    # (x, y, w, h) in LR pixels
        self._thumb_pix = None   # context thumbnail (full LR)
        self._thumb_rect = None
        self._mode = 'empty'     # 'empty' | 'select' | 'result'

    # ---------------- public API ----------------
    def set_image(self, lr_rgb):
        self._lr = np.ascontiguousarray(lr_rgb)
        self._sr = None; self._base_pix = None; self._disp_pix = None
        self._crop_box = None
        self._sel_start = None; self._sel_cur = None; self._dragging = False
        self._mode = 'select'
        self._rebuild_lr()
        self.update()

    def set_result(self, crop_box, sr_rgb):
        self._crop_box = crop_box
        self._sr = np.ascontiguousarray(sr_rgb)
        self._base_pix = self._np_to_pix(self._sr)
        self._mode = 'result'
        self._recompute_fit()
        # show at native 1:1 first (clamped so a giant SR can still zoom out)
        self._zoom = max(self._zoom_min, min(1.0, self.ZOOM_MAX))
        self._offset = QPoint(0, 0)
        self._rebuild_disp()
        self._clamp_offset()
        self._rebuild_thumb()
        self.update()

    def back_to_select(self):
        if self._lr is not None:
            self._sel_start = None; self._sel_cur = None
            self._mode = 'select'
            self._rebuild_lr()
            self.update()

    def has_image(self):
        return self._lr is not None

    def selection_image_box(self):
        """Current rubber-band as (x, y, w, h) in source-image pixels, or None
        when there is no usable selection."""
        if self._lr is None or self._lr_rect is None:
            return None
        if self._sel_start is None or self._sel_cur is None:
            return None
        from PyQt6.QtCore import QRect
        sel = QRect(self._sel_start, self._sel_cur).normalized()
        sel = sel.intersected(self._lr_rect)
        if sel.width() < 4 or sel.height() < 4:
            return None
        lh, lw = self._lr.shape[:2]
        rx = (sel.x() - self._lr_rect.x()) / self._lr_rect.width()
        ry = (sel.y() - self._lr_rect.y()) / self._lr_rect.height()
        rw = sel.width() / self._lr_rect.width()
        rh = sel.height() / self._lr_rect.height()
        x = max(0, min(int(round(rx * lw)), lw - 1))
        y = max(0, min(int(round(ry * lh)), lh - 1))
        w = max(1, min(int(round(rw * lw)), lw - x))
        h = max(1, min(int(round(rh * lh)), lh - y))
        return (x, y, w, h)

    # ---------------- internals ----------------
    def resizeEvent(self, e):
        if self._mode == 'select':
            self._rebuild_lr()
        elif self._mode == 'result':
            self._recompute_fit(keep_zoom=True)
            self._rebuild_disp()
            self._clamp_offset()
            self._rebuild_thumb()
        super().resizeEvent(e)

    @staticmethod
    def _np_to_pix(arr):
        arr = np.ascontiguousarray(arr)
        h, w, _ = arr.shape
        qi = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qi.copy())

    def _fit_pix(self, arr):
        from PyQt6.QtCore import QRect
        pix = self._np_to_pix(arr)
        aw, ah = max(self.width() - 8, 1), max(self.height() - 8, 1)
        sp = pix.scaled(aw, ah, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
        x = (self.width() - sp.width()) // 2
        y = (self.height() - sp.height()) // 2
        return sp, QRect(x, y, sp.width(), sp.height())

    def _rebuild_lr(self):
        if self._lr is not None:
            self._lr_pix, self._lr_rect = self._fit_pix(self._lr)

    # ----- result-mode zoom/pan (fit-aware native-1:1 viewer) -----
    def _sr_size(self):
        return self._sr.shape[1], self._sr.shape[0]   # (w, h)

    def _recompute_fit(self, keep_zoom=False):
        if self._sr is None:
            return
        sw, sh = self._sr_size()
        W = max(self.width() - 8, 1)
        H = max(self.height() - 8, 1)
        # never need to zoom out past native unless SR is bigger than viewport
        self._zoom_min = min(min(W / sw, H / sh), 1.0)
        if keep_zoom:
            self._zoom = max(self._zoom_min, min(self.ZOOM_MAX, self._zoom))

    def _rebuild_disp(self):
        if self._base_pix is None:
            return
        sw, sh = self._sr_size()
        tw = max(1, int(round(sw * self._zoom)))
        th = max(1, int(round(sh * self._zoom)))
        if abs(self._zoom - 1.0) < 1e-6:
            self._disp_pix = self._base_pix
        else:
            self._disp_pix = self._base_pix.scaled(
                tw, th, Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation)

    def _clamp_offset(self):
        if self._disp_pix is None:
            return
        pw, ph = self._disp_pix.width(), self._disp_pix.height()
        W, H = self.width(), self.height()
        ox = (W - pw) // 2 if pw <= W else max(W - pw, min(0, self._offset.x()))
        oy = (H - ph) // 2 if ph <= H else max(H - ph, min(0, self._offset.y()))
        self._offset = QPoint(ox, oy)

    def _rebuild_thumb(self):
        from PyQt6.QtCore import QRect
        if self._lr is None:
            return
        lh, lw = self._lr.shape[:2]
        s = self.THUMB_MAX / max(lw, lh, 1)
        tw, th = max(1, int(round(lw * s))), max(1, int(round(lh * s)))
        self._thumb_pix = self._np_to_pix(self._lr).scaled(
            tw, th, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        x = self.MARGIN
        # leave room BELOW the thumbnail for the caption line
        y = self.height() - self.MARGIN - self.CAPTION_H - self._thumb_pix.height()
        self._thumb_rect = QRect(x, y, self._thumb_pix.width(),
                                 self._thumb_pix.height())

    # ---------------- mouse: differs by mode ----------------
    def mousePressEvent(self, e):
        if self._mode == 'select':
            if self._lr_rect is not None and self._lr_rect.contains(e.pos()):
                self._sel_start = e.pos()
                self._sel_cur = e.pos()
                self._dragging = True
                self.update()
        elif self._mode == 'result':
            self._pan_last = e.pos()

    def mouseMoveEvent(self, e):
        if self._mode == 'select' and self._dragging and self._lr_rect is not None:
            from PyQt6.QtCore import QPoint as QP
            x = min(max(e.pos().x(), self._lr_rect.x()),
                    self._lr_rect.x() + self._lr_rect.width())
            y = min(max(e.pos().y(), self._lr_rect.y()),
                    self._lr_rect.y() + self._lr_rect.height())
            self._sel_cur = QP(x, y)
            self.update()
        elif self._mode == 'result' and self._pan_last is not None \
                and self._disp_pix is not None:
            d = e.pos() - self._pan_last
            self._offset = self._offset + d
            self._pan_last = e.pos()
            self._clamp_offset()
            self.update()

    def mouseReleaseEvent(self, e):
        self._dragging = False
        self._pan_last = None

    def wheelEvent(self, e):
        if self._mode != 'result' or self._base_pix is None:
            return
        old = self._zoom
        factor = 1.2 if e.angleDelta().y() > 0 else 1.0 / 1.2
        new = max(self._zoom_min, min(self.ZOOM_MAX, old * factor))
        if abs(new - old) < 1e-9:
            e.accept(); return
        cp = e.position()
        cx, cy = cp.x(), cp.y()
        sx = (cx - self._offset.x()) / old
        sy = (cy - self._offset.y()) / old
        self._zoom = new
        self._rebuild_disp()
        self._offset = QPoint(int(round(cx - sx * new)),
                              int(round(cy - sy * new)))
        self._clamp_offset()
        self.update()
        e.accept()

    # ---------------- paint ----------------
    def paintEvent(self, e):
        super().paintEvent(e)
        from PyQt6.QtCore import QRect
        from PyQt6.QtGui import QPen, QColor, QFont
        p = QPainter(self)

        if self._mode == 'empty' or (self._mode == 'select' and self._lr_pix is None):
            f = QFont(); f.setPointSize(11); p.setFont(f)
            p.setPen(QColor('#888888'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       'Load an image, then drag a box to select a region')
            p.end(); return

        if self._mode == 'select':
            p.drawPixmap(self._lr_rect.topLeft(), self._lr_pix)
            if self._sel_start is not None and self._sel_cur is not None:
                sel = QRect(self._sel_start, self._sel_cur).normalized()
                p.setPen(QPen(QColor(255, 60, 60), 2))
                p.drawRect(sel)
            f = QFont(); f.setPointSize(8); f.setBold(True); p.setFont(f)
            hint = 'Drag a box, then press "SR on selection"'
            p.setPen(QColor(0, 0, 0)); p.drawText(self.MARGIN + 1, self.MARGIN + 13, hint)
            p.setPen(QColor(255, 255, 255)); p.drawText(self.MARGIN, self.MARGIN + 12, hint)
            p.end(); return

        # result mode: SR at native 1:1 by default, wheel-zoom + drag-pan
        if self._disp_pix is not None:
            p.drawPixmap(self._offset, self._disp_pix)
        if self._thumb_pix is not None and self._thumb_rect is not None:
            p.drawPixmap(self._thumb_rect.topLeft(), self._thumb_pix)
            p.setPen(QPen(QColor(20, 20, 20), 1)); p.drawRect(self._thumb_rect)
            if self._crop_box is not None and self._lr is not None:
                lh, lw = self._lr.shape[:2]
                cx, cy, cw, ch = self._crop_box
                tr = self._thumb_rect
                bx = tr.x() + int(round(cx / lw * tr.width()))
                by = tr.y() + int(round(cy / lh * tr.height()))
                bw = max(2, int(round(cw / lw * tr.width())))
                bh = max(2, int(round(ch / lh * tr.height())))
                p.setPen(QPen(QColor(255, 60, 60), 2))
                p.drawRect(bx, by, bw, bh)
            f = QFont(); f.setPointSize(8); f.setBold(True); p.setFont(f)
            cap_y = self._thumb_rect.y() + self._thumb_rect.height() + 13
            zpct = int(round(self._zoom * 100))
            cap = f'Source (red = SR region) · wheel zoom · drag pan   ({zpct}%)'
            p.setPen(QColor(0, 0, 0)); p.drawText(self.MARGIN + 1, cap_y + 1, cap)
            p.setPen(QColor(255, 255, 255)); p.drawText(self.MARGIN, cap_y, cap)
        p.end()


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SFMformer SR')
        self.resize(1024, 600)
        # detect the OS light/dark preference once; all widget colours below
        # are pulled from this so the UI matches the system theme.
        self._dark = is_dark_mode(QApplication.instance())
        self._c = theme_colors(self._dark)
        self.scale = 4
        self.worker = None
        self.results = {}          # name -> (lr, sr, hr, dt, psnr, ssim)
        self.last_sr = None        # current SR np.ndarray for saving
        self.last_name = None

        self._build_ui()
        self._apply_style()

        # device monitor timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_stats)
        self.timer.start(2000)
        self._update_stats()

    # ---------------- UI construction ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        # ---- Header ----
        header = QHBoxLayout()
        title = QLabel('SFMformer SR')
        title.setStyleSheet('font-size:18px; font-weight:700;')
        if IS_PI and DEVICE == 'cpu':
            dev = 'Pi 5 · CPU · FP32'
        elif DEVICE == 'cuda':
            try:
                gpu = torch.cuda.get_device_name(0)
            except Exception:
                gpu = 'GPU'
            dev = f'{gpu} · FP32'
        else:
            dev = 'CPU · FP32'
        self.dev_label = QLabel(f'● {dev}')
        self.dev_label.setStyleSheet(f'color:{self._c["accent"]}; font-weight:600;')

        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.dev_label)
        header.addSpacing(12)

        # scale buttons
        header.addWidget(QLabel('Scale:'))
        self.scale_group = QButtonGroup(self)
        for s in (2, 3, 4):
            b = QPushButton(f'×{s}')
            b.setCheckable(True)
            b.setFixedWidth(44)
            if s == self.scale:
                b.setChecked(True)
            b.clicked.connect(lambda _, sc=s: self._set_scale(sc))
            self.scale_group.addButton(b)
            header.addWidget(b)

        header.addSpacing(12)
        header.addWidget(QLabel('View:'))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(['Side-by-Side', 'Magnifier', 'Crop & SR'])
        self.mode_selector.currentIndexChanged.connect(self._on_mode_changed)
        self.mode_selector.setFixedWidth(130)
        header.addWidget(self.mode_selector)
        root.addLayout(header)

        # ---- Body: left images | right panel ----
        body = QHBoxLayout()
        body.setSpacing(8)

        # left: a stacked widget with two view modes
        self.stack = QStackedWidget()

        # ----- Page 0: Side-by-Side (two linked ZoomViews + zoom slider) -----
        page_sbs = QWidget()
        sbs_v = QVBoxLayout(page_sbs)
        sbs_v.setContentsMargins(0, 0, 0, 0)
        img_box = QHBoxLayout()
        img_box.setSpacing(6)
        self.view_lr = ZoomView('LR')
        self.view_sr = ZoomView('SR')
        self.view_lr.link(self.view_sr)
        self.view_sr.link(self.view_lr)
        # When the user wheel-zooms either view, sync the slider position so
        # the UI stays consistent. setValue is wrapped in blockSignals to
        # avoid bouncing the change back into the views.
        self.view_lr.zoom_changed.connect(self._on_view_zoom_change)
        self.view_sr.zoom_changed.connect(self._on_view_zoom_change)

        lr_col = QVBoxLayout()
        lr_tag = QLabel('INPUT · LR')
        lr_tag.setStyleSheet(f'font-weight:600; color:{self._c["muted"]};')
        lr_col.addWidget(lr_tag)
        lr_col.addWidget(self.view_lr, 1)

        sr_col = QVBoxLayout()
        sr_tag = QLabel('OUTPUT · SR')
        sr_tag.setStyleSheet(f'font-weight:600; color:{self._c["muted"]};')
        sr_col.addWidget(sr_tag)
        sr_col.addWidget(self.view_sr, 1)

        img_box.addLayout(lr_col, 1)
        img_box.addLayout(sr_col, 1)
        sbs_v.addLayout(img_box, 1)

        zrow = QHBoxLayout()
        zrow.addWidget(QLabel('Zoom'))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider)
        self.zoom_pct = QLabel('100%')
        self.zoom_pct.setFixedWidth(44)
        zrow.addWidget(self.zoom_slider, 1)
        zrow.addWidget(self.zoom_pct)
        fit_btn = QPushButton('Fit')
        fit_btn.setFixedWidth(50)
        fit_btn.clicked.connect(self._fit_zoom)
        zrow.addWidget(fit_btn)
        sbs_v.addLayout(zrow)

        # ----- Page 1: Magnifier (SR base + hover loupe, SR-only) -----
        page_mag = QWidget()
        mag_v = QVBoxLayout(page_mag)
        mag_v.setContentsMargins(0, 0, 0, 0)
        mag_tag = QLabel('SR  ·  hover to inspect detail')
        mag_tag.setStyleSheet(f'font-weight:600; color:{self._c["muted"]};')
        mag_v.addWidget(mag_tag)
        self.magnifier = MagnifierView()
        mag_v.addWidget(self.magnifier, 1)

        # ----- Page 2: Crop & SR (load image, box a region, SR just that) -----
        page_crop = QWidget()
        crop_v = QVBoxLayout(page_crop)
        crop_v.setContentsMargins(0, 0, 0, 0)
        crop_bar = QHBoxLayout()
        self.btn_crop_load = QPushButton('Load image')
        self.btn_crop_run = QPushButton('▶ SR on selection')
        self.btn_crop_back = QPushButton('↺ New selection')
        self.btn_crop_run.setEnabled(False)
        self.btn_crop_back.setEnabled(False)
        crop_bar.addWidget(self.btn_crop_load)
        crop_bar.addWidget(self.btn_crop_run)
        crop_bar.addWidget(self.btn_crop_back)
        crop_bar.addStretch(1)
        crop_v.addLayout(crop_bar)
        self.crop_view = CropSRView()
        crop_v.addWidget(self.crop_view, 1)
        self.btn_crop_load.clicked.connect(self._crop_load)
        self.btn_crop_run.clicked.connect(self._crop_run)
        self.btn_crop_back.clicked.connect(self._crop_back)

        self.stack.addWidget(page_sbs)    # index 0
        self.stack.addWidget(page_mag)    # index 1
        self.stack.addWidget(page_crop)   # index 2

        # left column = stack on top + dimension bar at the bottom
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(4)
        left_col.addWidget(self.stack, 1)

        self.lbl_dims = QLabel('Input —  →  Output —')
        self.lbl_dims.setStyleSheet(
            f'color:{self._c["muted"]}; font-weight:600; padding:2px 4px;')
        left_col.addWidget(self.lbl_dims)

        left_wrap = QWidget()
        left_wrap.setLayout(left_col)
        body.addWidget(left_wrap, 1)

        # right: control panel (fixed width)
        panel = self._build_panel()
        body.addWidget(panel)

        root.addLayout(body, 1)

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(250)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # Model card
        model_box = QGroupBox('Model')
        mb = QVBoxLayout(model_box)
        mb.addWidget(QLabel('<b>SFMformer</b> (lightweight)'))
        mb.addWidget(QLabel('Params: 0.97 M'))
        mb.addWidget(QLabel('Precision: FP32'))
        backend = 'PyTorch CPU' if DEVICE == 'cpu' else 'PyTorch CUDA'
        mb.addWidget(QLabel(f'Backend: {backend}'))
        v.addWidget(model_box)

        # Inference status
        inf_box = QGroupBox('Inference')
        ib = QVBoxLayout(inf_box)
        self.lbl_status = QLabel('Idle')
        self.lbl_time = QLabel('Time: —')
        self.lbl_metric = QLabel('PSNR-Y: —   SSIM-Y: —')
        self.progress = QProgressBar()
        self.progress.setValue(0)
        ib.addWidget(self.lbl_status)
        ib.addWidget(self.lbl_time)
        ib.addWidget(self.lbl_metric)
        ib.addWidget(self.progress)
        v.addWidget(inf_box)

        # Queue (also the result browser: click a done item to view it)
        q_box = QGroupBox('Queue')
        qb = QVBoxLayout(q_box)
        qbtns = QHBoxLayout()
        self.btn_add_files = QPushButton('+ Files')
        self.btn_add_folder = QPushButton('+ Folder')
        self.btn_clear = QPushButton('Clear')
        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_clear.clicked.connect(self._clear_queue)
        for b in (self.btn_add_files, self.btn_add_folder, self.btn_clear):
            b.setStyleSheet('padding:4px;')
            qbtns.addWidget(b)
        qb.addLayout(qbtns)

        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(120)
        self.queue_list.currentItemChanged.connect(self._on_queue_click)
        qb.addWidget(self.queue_list)

        self.btn_run_all = QPushButton('▶ Run All')
        self.btn_run_all.clicked.connect(self._run_queue)
        qb.addWidget(self.btn_run_all)
        self._queue_box = q_box
        v.addWidget(q_box)

        # Device stats
        dev_box = QGroupBox('Device')
        db = QGridLayout(dev_box)
        self.lbl_temp = QLabel('—')
        self.lbl_cpu = QLabel('—')
        self.lbl_ram = QLabel('—')
        db.addWidget(QLabel('Temp'), 0, 0); db.addWidget(self.lbl_temp, 0, 1)
        db.addWidget(QLabel('CPU'), 1, 0);  db.addWidget(self.lbl_cpu, 1, 1)
        db.addWidget(QLabel('RAM'), 2, 0);  db.addWidget(self.lbl_ram, 2, 1)
        v.addWidget(dev_box)

        # Benchmark (separate path: computes PSNR/SSIM vs HR). Wrapped in a
        # container so it can be hidden as one unit in Crop & SR mode.
        self._bench_box = QWidget()
        bench_v = QVBoxLayout(self._bench_box)
        bench_v.setContentsMargins(0, 0, 0, 0)
        bench_lbl = QLabel('Benchmark (with PSNR/SSIM):')
        bench_lbl.setStyleSheet(f'color:{self._c["muted"]}; font-weight:600;')
        bench_v.addWidget(bench_lbl)
        bench_row = QHBoxLayout()
        self.bench_selector = QComboBox()
        self.bench_selector.addItems(BENCHMARKS)
        bench_row.addWidget(self.bench_selector, 1)
        self.btn_bench = QPushButton('Run')
        self.btn_bench.setFixedWidth(54)
        self.btn_bench.clicked.connect(self._on_run_benchmark)
        bench_row.addWidget(self.btn_bench)
        bench_v.addLayout(bench_row)
        v.addWidget(self._bench_box)

        self.btn_save = QPushButton('Save SR Result')
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save.setEnabled(False)
        v.addWidget(self.btn_save)

        v.addStretch(1)
        return panel

    def _apply_style(self):
        self.setStyleSheet(build_stylesheet(self._c))

    # ---------------- Actions ----------------
    def _set_scale(self, s):
        self.scale = s

    def _set_busy(self, busy: bool):
        for b in (self.btn_add_files, self.btn_add_folder, self.btn_clear,
                  self.btn_run_all, self.btn_bench,
                  self.btn_crop_load, self.btn_crop_run, self.btn_crop_back):
            b.setEnabled(not busy)
        for b in self.scale_group.buttons():
            b.setEnabled(not busy)
        # restore crop buttons to their context-appropriate state when idle
        if not busy:
            self.btn_crop_run.setEnabled(self.crop_view.has_image())
            self.btn_crop_back.setEnabled(self.crop_view._mode == 'result')

    # ---------------- Crop & SR ----------------
    def _crop_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load image to crop', str(ROOT),
            'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not path:
            return
        try:
            lr = np.array(Image.open(path).convert('RGB'))
        except Exception as e:
            self.lbl_status.setText(f'Load failed: {e}')
            return
        self.crop_view.set_image(lr)
        self.btn_crop_run.setEnabled(True)
        self.btn_crop_back.setEnabled(False)
        self._crop_name = Path(path).stem
        h, w = lr.shape[:2]
        self.lbl_dims.setText('Input —  →  Output —')
        self.lbl_status.setText(
            f'Loaded {Path(path).name} ({w}×{h}) — drag a box, then SR on selection')

    def _crop_run(self):
        box = self.crop_view.selection_image_box()
        if box is None:
            self.lbl_status.setText('Draw a selection box on the image first')
            return
        x, y, w, h = box
        self._set_busy(True)
        self.lbl_status.setText(
            f'SR on {w}×{h} region → {w * self.scale}×{h * self.scale}…')
        self.crop_worker = CropWorker(self.scale, self.crop_view._lr, box)
        self.crop_worker.done.connect(self._crop_done)
        self.crop_worker.tile_progress.connect(self._on_tile_progress)
        self.crop_worker.error.connect(self._on_error)
        self.crop_worker.start()

    def _crop_done(self, box, sr_rgb, dt):
        self._set_busy(False)
        self.crop_view.set_result(box, sr_rgb)
        self.btn_crop_back.setEnabled(True)
        x, y, w, h = box
        sh2, sw2 = sr_rgb.shape[0], sr_rgb.shape[1]
        self.last_sr = sr_rgb
        self.last_name = f'{getattr(self, "_crop_name", "crop")}_crop'
        self.btn_save.setEnabled(True)
        # the dimension bar should reflect the cropped region, not the full img
        self.lbl_dims.setText(
            f'Input  {w}×{h}   →   Output  {sw2}×{sh2}   (×{self.scale})')
        self.lbl_status.setText(
            f'Done · {w}×{h} → {sw2}×{sh2}  ({dt:.2f}s)')

    def _crop_back(self):
        self.crop_view.back_to_select()
        self.btn_crop_back.setEnabled(False)
        self.lbl_dims.setText('Input —  →  Output —')
        self.lbl_status.setText('Draw a new box, then SR on selection')

    SUPPORTED = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def _add_to_queue(self, path):
        """Add one image path to the queue (skip duplicates)."""
        path = str(path)
        # avoid duplicates by stored path
        for i in range(self.queue_list.count()):
            if self.queue_list.item(i).data(Qt.ItemDataRole.UserRole) == path:
                return
        name = Path(path).name
        item = QListWidgetItem(f'⏳ {name}')
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.queue_list.addItem(item)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Select image(s)', str(Path.home()),
            'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        for p in paths:
            self._add_to_queue(p)
        self.lbl_status.setText(f'Queue: {self.queue_list.count()} image(s)')

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select a folder', str(Path.home()))
        if not folder:
            return
        added = 0
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in self.SUPPORTED:
                self._add_to_queue(p)
                added += 1
        self.lbl_status.setText(f'Added {added} from folder · queue {self.queue_list.count()}')

    def _clear_queue(self):
        self.queue_list.clear()
        self.results.clear()
        self.magnifier.clear_pair()
        self.view_lr.clear_image()
        self.view_sr.clear_image()
        self.last_sr = None
        self.btn_save.setEnabled(False)
        self.lbl_status.setText('Queue cleared')
        self.lbl_time.setText('Time: —')
        self.lbl_metric.setText('PSNR-Y: —   SSIM-Y: —')
        self.lbl_dims.setText('Input —  →  Output —')

    def _run_queue(self):
        jobs = []
        for i in range(self.queue_list.count()):
            it = self.queue_list.item(i)
            path = it.data(Qt.ItemDataRole.UserRole)
            jobs.append((Path(path).name, path, None))   # custom images: no HR
        if not jobs:
            self.lbl_status.setText('Queue is empty — add files or a folder')
            return
        # reset done-marks to pending
        for i in range(self.queue_list.count()):
            it = self.queue_list.item(i)
            it.setText(f'⏳ {Path(it.data(Qt.ItemDataRole.UserRole)).name}')
        self.results.clear()
        self.lbl_metric.setText('PSNR-Y: —   SSIM-Y: —  (no HR)')
        self._launch(jobs)

    def _on_run_benchmark(self):
        bench = self.bench_selector.currentText()
        hr_dir = ROOT / 'datasets' / 'TestDataSR' / 'HR' / bench / f'x{self.scale}'
        lr_dir = ROOT / 'datasets' / 'TestDataSR' / 'LR' / 'LRBI' / bench / f'x{self.scale}'
        if not lr_dir.exists():
            self.lbl_status.setText(f'LR dir not found:\n{lr_dir}')
            return
        exts = {'.png', '.bmp', '.jpg', '.jpeg'}
        lr_files = sorted([p for p in lr_dir.iterdir() if p.suffix.lower() in exts])
        if not lr_files:
            self.lbl_status.setText('No LR images found')
            return
        # benchmark populates the queue list too (so user can browse results)
        self.queue_list.clear()
        jobs = []
        for p in lr_files:
            hr = hr_dir / p.name
            jobs.append((p.name, str(p), str(hr) if hr.exists() else None))
            item = QListWidgetItem(f'⏳ {p.name}')
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.queue_list.addItem(item)
        self.results.clear()
        self._launch(jobs)

    def _launch(self, jobs):
        self._set_busy(True)
        self.progress.setMaximum(len(jobs))
        self.progress.setValue(0)
        self.lbl_status.setText(f'Running… (0/{len(jobs)})')
        self.worker = InferenceWorker(self.scale, jobs)
        self.worker.one_done.connect(self._on_one_done)
        self.worker.progress.connect(self._on_progress)
        self.worker.tile_progress.connect(self._on_tile_progress)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_one_done(self, name, lr, sr, hr, dt, psnr, ssim):
        self.results[name] = (lr, sr, hr, dt, psnr, ssim)
        # Mark the matching queue item as done. For benchmark items we have
        # HR, so show per-image PSNR/SSIM; for custom uploads we only have
        # the elapsed time.
        for i in range(self.queue_list.count()):
            it = self.queue_list.item(i)
            if Path(it.data(Qt.ItemDataRole.UserRole)).name == name:
                if hr is not None and psnr > 0:
                    it.setText(f'✓ {name}   {psnr:.2f} dB / {ssim:.4f}')
                else:
                    it.setText(f'✓ {name}   ({dt:.1f}s)')
                break
        # Auto-show the first finished result so the user sees something
        # right away. _show_result writes single-image stats into lbl_metric;
        # we override that below with the running average whenever this run
        # has HR ground truth, which is the user's expectation during a
        # benchmark.
        if len(self.results) == 1:
            self._show_result(name)
        # Keep `Time` dynamic — show the time of the image that just
        # finished. Otherwise it would stay frozen at the first image's
        # value (the one auto-shown) for the rest of the run, which is
        # misleading. The status line keeps the running average.
        self.lbl_time.setText(f'Time: {dt:.2f} s   (last: {name})')
        vals = [(p, s) for (_, _, h, _, p, s) in self.results.values()
                if h is not None and p > 0]
        if vals:
            ap = float(np.mean([p for p, _ in vals]))
            asim = float(np.mean([s for _, s in vals]))
            self.lbl_metric.setText(
                f'AVG PSNR-Y: {ap:.4f}   SSIM-Y: {asim:.4f}   '
                f'({len(vals)} img)')

    def _on_progress(self, done, total):
        self.progress.setValue(done)
        self.lbl_status.setText(f'Running… ({done}/{total})')

    def _on_tile_progress(self, done, total):
        # Only surface tile-level progress for genuinely large (heavily tiled)
        # images, where a single inference can take many minutes and the user
        # needs reassurance it isn't frozen. Small benchmark images (<8 tiles)
        # are skipped so the status line doesn't flicker during a batch run.
        if total >= 8:
            self.lbl_status.setText(f'Processing tile {done}/{total}…')

    def _on_finished(self, avg):
        self._set_busy(False)
        n = len(self.results)
        # average PSNR/SSIM over those with HR (benchmark runs)
        vals = [(p, s) for (_, _, hr, _, p, s) in self.results.values()
                if hr is not None and p > 0]
        if vals:
            ap = float(np.mean([p for p, _ in vals]))
            asim = float(np.mean([s for _, s in vals]))
            self.lbl_status.setText(f'Done · {n} img · avg {avg:.2f}s')
            self.lbl_metric.setText(
                f'AVG PSNR-Y: {ap:.4f}   SSIM-Y: {asim:.4f}   '
                f'({len(vals)} img)')
        else:
            self.lbl_status.setText(f'Done · {n} img · {avg:.2f}s')

    def _on_error(self, msg):
        self._set_busy(False)
        self.lbl_status.setText('ERROR (see terminal)')
        print('--- Inference error ---\n', msg)

    def _on_queue_click(self, current, previous):
        if current is None:
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        name = Path(path).name
        if name in self.results:
            self._show_result(name)

    def _show_result(self, name):
        lr, sr, hr, dt, psnr, ssim = self.results[name]
        self.view_lr.set_image(np.ascontiguousarray(lr))
        self.view_sr.set_image(np.ascontiguousarray(sr))
        # SR/LR pixel ratio. Float for the zoom views (handles any scale);
        # rounded int for the magnifier (its sampling uses integer math).
        ratio_f = sr.shape[0] / max(lr.shape[0], 1)
        self.view_lr.set_ratio(ratio_f)
        self.view_sr.set_ratio(1.0)
        self.magnifier.set_pair(lr, sr, max(1, round(ratio_f)))
        self.last_sr = sr
        self.last_name = name
        self.btn_save.setEnabled(True)
        lh, lw = lr.shape[0], lr.shape[1]
        sh2, sw2 = sr.shape[0], sr.shape[1]
        self.lbl_dims.setText(
            f'Input  {lw}×{lh}   →   Output  {sw2}×{sh2}   (×{self.scale})')
        self.lbl_time.setText(f'Time: {dt:.2f} s')
        if hr is not None and psnr > 0:
            self.lbl_metric.setText(f'PSNR-Y: {psnr:.4f}   SSIM-Y: {ssim:.4f}')
        else:
            self.lbl_metric.setText('PSNR-Y: —   SSIM-Y: —  (no HR)')
        # Fit both views to the same physical region and sync the slider.
        self._fit_zoom()

    def _on_save(self):
        if self.last_sr is None:
            return
        default = f'{Path(self.last_name).stem}_SFMformer_SRx{self.scale}.png'
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save SR image', str(Path.home() / default), 'PNG (*.png)')
        if path:
            Image.fromarray(self.last_sr).save(path)
            self.lbl_status.setText(f'Saved: {Path(path).name}')

    # ---------------- View mode ----------------
    def _on_mode_changed(self, idx):
        self.stack.setCurrentIndex(idx)
        # Crop & SR is a self-contained workflow (its own Load / SR / Reset
        # buttons live on the page). Hide the queue + benchmark panels in that
        # mode so the two inference paths can't be confused.
        # Crop & SR is now index 2 (Quadrant removed). It is a self-contained
        # workflow, so hide the queue + benchmark panels to avoid confusion.
        crop_mode = (idx == 2)
        self._queue_box.setVisible(not crop_mode)
        self._bench_box.setVisible(not crop_mode)
        # avoid showing a stale queue/benchmark dimension while in crop mode
        # before the user has run a crop
        if crop_mode and self.crop_view._mode != 'result':
            self.lbl_dims.setText('Input —  →  Output —')

    # ---------------- Zoom ----------------
    def _on_view_zoom_change(self, pct: int):
        """Called when a ZoomView's wheelEvent changes its zoom. Mirror the
        new value onto the slider without re-triggering the slider handler."""
        pct = max(self.zoom_slider.minimum(),
                  min(self.zoom_slider.maximum(), int(pct)))
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(pct)
        self.zoom_pct.setText(f'{pct}%')
        self.zoom_slider.blockSignals(False)

    def _on_zoom_slider(self, val):
        self.zoom_pct.setText(f'{val}%')
        # set both explicitly so they stay in lock-step
        self.view_lr.set_scale_pct(val, broadcast=False)
        self.view_sr.set_scale_pct(val, broadcast=False)

    def _fit_zoom(self):
        if self.view_lr._pix_full is None or self.view_sr._pix_full is None:
            return
        # Both views show the same physical area; fitting the SR (the larger
        # pixmap) automatically fits the LR. We compute the physical zoom
        # (screen px per SR px) that makes the SR fit its viewport.
        sr_pix = self.view_sr._pix_full
        vw = max(self.view_sr.viewport().width(), 1)
        vh = max(self.view_sr.viewport().height(), 1)
        phys = min(vw / sr_pix.width(),
                   vh / sr_pix.height(), 1.0)
        pct = max(10, min(400, int(round(phys * 100))))
        self.view_lr.set_scale_pct(pct, broadcast=False)
        self.view_sr.set_scale_pct(pct, broadcast=False)
        # Reset scroll to top-left so both views start aligned.
        for v in (self.view_lr, self.view_sr):
            v.horizontalScrollBar().setValue(0)
            v.verticalScrollBar().setValue(0)
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(pct)
        self.zoom_pct.setText(f'{pct}%')
        self.zoom_slider.blockSignals(False)

    # ---------------- Stats ----------------
    def _update_stats(self):
        self.lbl_temp.setText(read_temp())
        try:
            import psutil
            self.lbl_cpu.setText(f'{psutil.cpu_percent():.0f}%')
            self.lbl_ram.setText(f'{psutil.virtual_memory().percent:.0f}%')
        except Exception:
            self.lbl_cpu.setText('n/a')
            self.lbl_ram.setText('n/a')


def main():
    app = QApplication(sys.argv)
    # Fusion renders consistently on the Pi and Windows and honours the palette
    # we set below, which is built from the OS light/dark preference.
    app.setStyle('Fusion')
    app.setPalette(make_palette(is_dark_mode(app)))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
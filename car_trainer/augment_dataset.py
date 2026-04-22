"""
augment_dataset.py  —  Interactive dataset augmentation tool for the DAVE-2 robot.

Usage:
    python augment_dataset.py
    python augment_dataset.py --catalog ../data/catalog_0.catalog --image_dir ../data/

Output:
    A NEW catalog file  <original_name>_aug.catalog   (never modifies the original).
    Augmented images written alongside the originals in IMAGE_DIR.

Augmentations available
───────────────────────
  • Horizontal flip          – mirrors the frame and negates steering angle
  • Brightness / contrast    – random gamma correction + contrast jitter
  • Gaussian blur            – simulates slight focus loss
  • Salt & pepper noise      – simulates camera sensor noise
  • Rotation jitter          – slight ±5° rotation with proportional steering adjust
  • JPEG compression         – encode/decode at low quality to match real-camera artifacts

GUI controls
────────────
  Preview pane  : original (left) vs augmented (right) with label overlay
  Sliders       : intensity for each augmentation type
  Checkboxes    : enable/disable each augmentation
  ◀ / ▶        : browse samples
  "Randomize"   : re-roll augmentation for current sample
  "Batch Apply" : apply all enabled augs to every sample → write aug catalog
"""

import argparse
import json
import os
import sys
import copy
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CATALOG = os.path.normpath(os.path.join(_THIS_DIR, "..", "data", "catalog_0.catalog"))
DEFAULT_IMG_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def aug_hflip(img: np.ndarray, steering: float, _strength: float):
    """Horizontally flip the image and negate the steering angle."""
    return cv2.flip(img, 1), -steering


def aug_brightness(img: np.ndarray, steering: float, strength: float):
    """Random gamma correction (dark / bright) + contrast jitter."""
    rng = random.Random()
    gamma = 1.0 + strength * rng.uniform(-0.6, 0.6)
    inv_g = 1.0 / max(gamma, 0.05)
    table = np.array([((i / 255.0) ** inv_g) * 255 for i in range(256)], dtype=np.uint8)
    bright = cv2.LUT(img, table)
    # Contrast jitter
    alpha = 1.0 + strength * rng.uniform(-0.3, 0.3)
    out = cv2.convertScaleAbs(bright, alpha=alpha, beta=0)
    return out, steering


def aug_blur(img: np.ndarray, steering: float, strength: float):
    """Gaussian blur proportional to strength."""
    ksize = max(1, int(strength * 9))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0), steering


def aug_noise(img: np.ndarray, steering: float, strength: float):
    """Salt & pepper noise."""
    out = img.copy()
    n_pixels = int(strength * img.size * 0.05)
    coords_y = np.random.randint(0, img.shape[0], n_pixels)
    coords_x = np.random.randint(0, img.shape[1], n_pixels)
    # Salt
    out[coords_y[:n_pixels//2], coords_x[:n_pixels//2]] = 255
    # Pepper
    out[coords_y[n_pixels//2:], coords_x[n_pixels//2:]] = 0
    return out, steering


def aug_rotation(img: np.ndarray, steering: float, strength: float):
    """Slight random rotation; steering adjusted proportionally."""
    angle_deg = strength * random.uniform(-5.0, 5.0)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    # Proportional steering correction: positive angle → slight right steer
    steer_adjust = -(angle_deg / 45.0)  # normalised [-1,1]
    new_steer = float(np.clip(steering + steer_adjust * strength, -1.0, 1.0))
    return rotated, new_steer


def aug_jpeg(img: np.ndarray, steering: float, strength: float):
    """JPEG encode at reduced quality to simulate camera compression artifacts."""
    quality = max(5, int(100 - strength * 80))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return decoded, steering


# Registry: (label, function)
AUGMENTATIONS = [
    ("Horizontal Flip",       aug_hflip),
    ("Brightness / Contrast", aug_brightness),
    ("Gaussian Blur",         aug_blur),
    ("Salt & Pepper Noise",   aug_noise),
    ("Rotation Jitter",       aug_rotation),
    ("JPEG Compression",      aug_jpeg),
]


def apply_augmentations(img, steering, enabled_flags, strengths):
    """Apply each enabled augmentation in sequence."""
    out_img = img.copy()
    out_steer = float(steering)
    for i, (_, fn) in enumerate(AUGMENTATIONS):
        if enabled_flags[i]:
            out_img, out_steer = fn(out_img, out_steer, strengths[i])
    return out_img, out_steer


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def read_catalog(path: str):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    entries.append(obj)
            except json.JSONDecodeError:
                continue
    return entries


def resolve_image(catalog_path: str, field: str) -> str:
    if not field:
        return ""
    if os.path.isabs(field) and os.path.exists(field):
        return field
    catalog_dir = os.path.dirname(os.path.abspath(catalog_path))
    candidates = [
        os.path.normpath(os.path.join(catalog_dir, field)),
        os.path.normpath(field),
        os.path.normpath(os.path.join(os.getcwd(), field)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

PREVIEW_W, PREVIEW_H = 400, 300   # per-panel preview size
PANEL_W = PREVIEW_W * 2 + 20      # total canvas width

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT    = "#0f3460"
HIGHLIGHT = "#e94560"
TEXT_CLR  = "#e0e0e0"
SLIDER_BG = "#0f3460"


class AugmentationApp(tk.Tk):
    def __init__(self, catalog_path: str, image_dir: str):
        super().__init__()
        self.catalog_path = catalog_path
        self.image_dir    = image_dir
        self.entries      = read_catalog(catalog_path)
        self.pos          = 0

        if not self.entries:
            messagebox.showerror("Empty catalog", f"No valid entries in:\n{catalog_path}")
            self.destroy()
            return

        self.title("DAVE-2 Dataset Augmentation Tool")
        self.configure(bg=DARK_BG)
        self.resizable(False, False)

        # Per-augmentation state
        self.enabled  = [tk.BooleanVar(value=(i == 0)) for i in range(len(AUGMENTATIONS))]
        self.strength = [tk.DoubleVar(value=0.5)        for _ in AUGMENTATIONS]

        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── Title bar ──────────────────────────────────────────────────
        title = tk.Label(self, text="✦  DAVE-2 Augmentation Studio  ✦",
                         bg=DARK_BG, fg=HIGHLIGHT,
                         font=("Segoe UI", 16, "bold"), pady=10)
        title.pack(fill="x")

        # ── Main body ─────────────────────────────────────────────────
        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=12, pady=4)

        # Left: preview canvas
        left = tk.Frame(body, bg=DARK_BG)
        left.pack(side="left", fill="both", expand=True)

        canvas_frame = tk.Frame(left, bg=ACCENT, bd=1, relief="solid")
        canvas_frame.pack(pady=6)
        self.canvas = tk.Canvas(canvas_frame, width=PANEL_W, height=PREVIEW_H + 60,
                                bg=PANEL_BG, highlightthickness=0)
        self.canvas.pack()

        # Label strip below canvas
        info_frame = tk.Frame(left, bg=DARK_BG)
        info_frame.pack(fill="x", pady=2)
        self.info_var = tk.StringVar(value="")
        tk.Label(info_frame, textvariable=self.info_var, bg=DARK_BG, fg=TEXT_CLR,
                 font=("Consolas", 10)).pack()

        # Navigation buttons
        nav = tk.Frame(left, bg=DARK_BG)
        nav.pack(pady=4)
        btn_style = dict(bg=ACCENT, fg=TEXT_CLR, activebackground=HIGHLIGHT,
                         font=("Segoe UI", 11, "bold"), bd=0, padx=14, pady=6,
                         cursor="hand2")
        tk.Button(nav, text="◀  Prev", command=self._prev, **btn_style).pack(side="left", padx=4)
        tk.Button(nav, text="Randomize 🎲", command=self._randomize, **btn_style).pack(side="left", padx=4)
        tk.Button(nav, text="Next  ▶", command=self._next, **btn_style).pack(side="left", padx=4)

        # Right: controls panel
        right = tk.Frame(body, bg=PANEL_BG, bd=1, relief="solid", padx=12, pady=10)
        right.pack(side="left", fill="y", padx=(14, 0))

        tk.Label(right, text="Augmentations", bg=PANEL_BG, fg=HIGHLIGHT,
                 font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=3,
                                                      sticky="w", pady=(0, 8))

        tk.Label(right, text="Enable", bg=PANEL_BG, fg=TEXT_CLR,
                 font=("Segoe UI", 9, "italic")).grid(row=1, column=0, sticky="w")
        tk.Label(right, text="Augmentation", bg=PANEL_BG, fg=TEXT_CLR,
                 font=("Segoe UI", 9, "italic")).grid(row=1, column=1, sticky="w", padx=6)
        tk.Label(right, text="Strength", bg=PANEL_BG, fg=TEXT_CLR,
                 font=("Segoe UI", 9, "italic")).grid(row=1, column=2, sticky="w", padx=6)

        for i, (label, _) in enumerate(AUGMENTATIONS):
            r = i + 2
            cb = tk.Checkbutton(right, variable=self.enabled[i],
                                bg=PANEL_BG, fg=TEXT_CLR, selectcolor=ACCENT,
                                activebackground=PANEL_BG, command=self._refresh)
            cb.grid(row=r, column=0, sticky="w", pady=3)

            tk.Label(right, text=label, bg=PANEL_BG, fg=TEXT_CLR,
                     font=("Segoe UI", 10), width=22, anchor="w").grid(
                         row=r, column=1, sticky="w", padx=6)

            sl = ttk.Scale(right, from_=0.05, to=1.0, orient="horizontal",
                           variable=self.strength[i], length=140,
                           command=lambda _e, _i=i: self._on_slider(_i))
            sl.grid(row=r, column=2, padx=6, sticky="w")

        # Separator
        sep = tk.Frame(right, bg=ACCENT, height=1)
        sep.grid(row=len(AUGMENTATIONS) + 2, column=0, columnspan=3,
                 sticky="ew", pady=12)

        # Batch controls
        batch_row = len(AUGMENTATIONS) + 3
        self.copies_var = tk.IntVar(value=1)
        tk.Label(right, text="Copies per sample:", bg=PANEL_BG, fg=TEXT_CLR,
                 font=("Segoe UI", 10)).grid(row=batch_row, column=0,
                                              columnspan=2, sticky="w")
        tk.Spinbox(right, from_=1, to=10, textvariable=self.copies_var,
                   width=4, bg=ACCENT, fg=TEXT_CLR,
                   font=("Segoe UI", 10)).grid(row=batch_row, column=2,
                                                sticky="w", padx=6)

        batch_btn = tk.Button(right, text="⚡  Batch Apply All",
                              command=self._batch_apply,
                              bg=HIGHLIGHT, fg="white",
                              activebackground="#c73652",
                              font=("Segoe UI", 11, "bold"),
                              bd=0, padx=10, pady=8, cursor="hand2")
        batch_btn.grid(row=batch_row + 1, column=0, columnspan=3,
                       sticky="ew", pady=(10, 2))

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(right, textvariable=self.status_var, bg=PANEL_BG,
                 fg=HIGHLIGHT, font=("Consolas", 9),
                 wraplength=260, justify="left").grid(
                     row=batch_row + 2, column=0, columnspan=3,
                     sticky="w", pady=4)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_slider(self, _idx):
        self._refresh()

    def _prev(self):
        self.pos = (self.pos - 1) % len(self.entries)
        self._refresh()

    def _next(self):
        self.pos = (self.pos + 1) % len(self.entries)
        self._refresh()

    def _randomize(self):
        for i in range(len(AUGMENTATIONS)):
            self.strength[i].set(round(random.uniform(0.1, 1.0), 2))
        self._refresh()

    # ------------------------------------------------------------------
    # Preview rendering
    # ------------------------------------------------------------------
    def _refresh(self):
        entry    = self.entries[self.pos]
        img_path = resolve_image(self.catalog_path, entry.get("cam/image_array", ""))
        orig     = cv2.imread(img_path)

        if orig is None:
            self._draw_error("Image not found:\n" + img_path)
            self.info_var.set(f"[{self.pos+1}/{len(self.entries)}]  MISSING IMAGE")
            return

        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        steer_raw = float(entry.get("angle", 0))
        steer_norm = steer_raw / 50.0   # normalised [-1,1]

        flags    = [v.get() for v in self.enabled]
        strengths = [v.get() for v in self.strength]

        aug_img, aug_steer_norm = apply_augmentations(orig_rgb, steer_norm, flags, strengths)
        aug_steer_raw = aug_steer_norm * 50.0

        # Render panels
        orig_panel = self._make_panel(orig_rgb, steer_raw,  "Original")
        aug_panel  = self._make_panel(aug_img,  aug_steer_raw, "Augmented")
        combined   = np.hstack([orig_panel, aug_panel])

        self._draw_to_canvas(combined)

        throttle = entry.get("throttle", "?")
        mode     = entry.get("user/mode", "?")
        self.info_var.set(
            f"[{self.pos+1}/{len(self.entries)}]  "
            f"steer={steer_raw:.1f}→{aug_steer_raw:.1f}  "
            f"throttle={throttle}  mode={mode}"
        )

    def _make_panel(self, img_rgb: np.ndarray, steer: float, label: str) -> np.ndarray:
        """Resize to preview size and draw overlay."""
        panel = cv2.resize(img_rgb, (PREVIEW_W, PREVIEW_H))
        # Dark header band
        header = np.full((40, PREVIEW_W, 3), (20, 20, 40), dtype=np.uint8)
        cv2.putText(header, label, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (233, 69, 96), 2, cv2.LINE_AA)
        steer_txt = f"steer: {steer:+.1f}"
        cv2.putText(header, steer_txt, (PREVIEW_W - 140, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        # Steering bar
        bar_mid  = PREVIEW_W // 2
        bar_len  = int((steer / 50.0) * (PREVIEW_W // 2 - 10))
        bar_col  = (233, 69, 96) if abs(steer) > 5 else (80, 200, 120)
        cv2.rectangle(header, (bar_mid, 34), (bar_mid + bar_len, 38), bar_col, -1)
        cv2.line(header, (bar_mid, 32), (bar_mid, 40), (180, 180, 180), 1)
        return np.vstack([header, panel])

    def _draw_to_canvas(self, img_rgb: np.ndarray):
        pil_img = Image.fromarray(img_rgb)
        self._tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(10, 10, anchor="nw", image=self._tk_img)

    def _draw_error(self, msg: str):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, PANEL_W, PREVIEW_H + 60,
                                     fill=PANEL_BG, outline="")
        self.canvas.create_text(PANEL_W // 2, (PREVIEW_H + 60) // 2,
                                text=msg, fill=HIGHLIGHT,
                                font=("Consolas", 11), justify="center")

    # ------------------------------------------------------------------
    # Batch apply
    # ------------------------------------------------------------------
    def _batch_apply(self):
        flags     = [v.get() for v in self.enabled]
        strengths = [v.get() for v in self.strength]
        copies    = self.copies_var.get()
        n_enabled = sum(flags)

        if n_enabled == 0:
            messagebox.showwarning("Nothing enabled",
                                   "Enable at least one augmentation first.")
            return

        # Determine output catalog path
        base, ext  = os.path.splitext(self.catalog_path)
        out_catalog = base + "_aug" + ext
        # Determine output image directory
        out_img_dir = os.path.join(os.path.dirname(self.catalog_path), "images_aug")
        os.makedirs(out_img_dir, exist_ok=True)

        # Confirm
        msg = (
            f"This will generate up to {len(self.entries) * copies} new samples.\n\n"
            f"Images  → {out_img_dir}\n"
            f"Catalog → {out_catalog}\n\n"
            "Proceed?"
        )
        if not messagebox.askyesno("Confirm Batch Apply", msg):
            return

        self.status_var.set("Running batch… please wait.")
        self.update_idletasks()

        new_entries = []
        # Compute starting index from the original catalog's max index
        existing_max = max((e.get("index", -1) for e in self.entries), default=-1)
        new_idx = existing_max + 1
        skipped = 0

        for i, entry in enumerate(self.entries):
            img_path = resolve_image(self.catalog_path, entry.get("cam/image_array", ""))
            orig_bgr = cv2.imread(img_path)
            if orig_bgr is None:
                skipped += 1
                continue
            orig_rgb   = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            steer_norm = float(entry.get("angle", 0)) / 50.0

            for c in range(copies):
                aug_rgb, aug_steer_norm = apply_augmentations(
                    orig_rgb, steer_norm, flags, strengths)
                aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

                filename = os.path.join(out_img_dir, f"aug_{new_idx}.jpg")
                cv2.imwrite(filename, aug_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

                new_entry = copy.deepcopy(entry)
                new_entry["index"]          = new_idx
                new_entry["cam/image_array"] = filename
                new_entry["angle"]          = round(aug_steer_norm * 50.0, 4)
                new_entry["augmented"]      = True
                new_entries.append(new_entry)
                new_idx += 1

            if i % 50 == 0:
                self.status_var.set(f"Processing {i+1}/{len(self.entries)}…")
                self.update_idletasks()

        # Write output catalog
        with open(out_catalog, "w", encoding="utf-8") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")

        summary = (
            f"✓ Done!  {len(new_entries)} samples written.\n"
            f"Skipped: {skipped} (missing images)\n"
            f"Catalog: {os.path.basename(out_catalog)}"
        )
        self.status_var.set(summary)
        messagebox.showinfo("Batch Complete", summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DAVE-2 Dataset Augmentation Tool")
    p.add_argument("--catalog",   default=DEFAULT_CATALOG,
                   help="Path to catalog file (.catalog / JSONL)")
    p.add_argument("--image_dir", default=DEFAULT_IMG_DIR,
                   help="Base directory for images")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.catalog):
        print(f"[ERROR] Catalog not found: {args.catalog}")
        sys.exit(1)
    app = AugmentationApp(catalog_path=args.catalog, image_dir=args.image_dir)
    app.mainloop()

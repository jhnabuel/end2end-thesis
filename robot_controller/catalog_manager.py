import argparse
import json
import os
import shutil
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    default_catalog = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "catalog_0.catalog")
    )

    parser = argparse.ArgumentParser(
        description="Preview and clean a Donkey-style catalog file."
    )
    parser.add_argument(
        "--catalog",
        default=default_catalog,
        help="Path to catalog file (json lines).",
    )
    parser.add_argument(
        "--window",
        default="Catalog Preview",
        help="OpenCV window title.",
    )
    return parser.parse_args()


def read_catalog(catalog_path: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with open(catalog_path, "r", encoding="utf-8") as f:
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


def write_catalog(catalog_path: str, entries: list[dict[str, Any]]) -> None:
    with open(catalog_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def resolve_image_path(catalog_path: str, image_field: str) -> str:
    if not image_field:
        return ""

    if os.path.isabs(image_field):
        return image_field

    catalog_dir = os.path.dirname(os.path.abspath(catalog_path))
    candidates = [
        os.path.normpath(os.path.join(catalog_dir, image_field)),
        os.path.normpath(image_field),
        os.path.normpath(os.path.join(os.getcwd(), image_field)),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


def fit_image(img: np.ndarray, max_w: int = 1280, max_h: int = 720) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale >= 1.0:
        return img
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def draw_overlay(
    img: np.ndarray,
    entry: dict[str, Any],
    pos: int,
    total: int,
    image_path: str,
) -> np.ndarray:
    canvas = img.copy()
    panel_h = 130
    panel = np.full((panel_h, canvas.shape[1], 3), (25, 25, 25), dtype=np.uint8)

    idx = entry.get("index", "?")
    throttle = entry.get("throttle", "?")
    angle = entry.get("angle", "?")
    mode = entry.get("user/mode", "?")

    lines = [
        f"Item: {pos + 1}/{total}   catalog index: {idx}",
        f"mode={mode}  throttle={throttle}  angle={angle}",
        f"image: {image_path}",
        "[A/Left]=Prev  [D/Right]=Next  [X or Del]=Delete  [Q or Esc]=Quit",
    ]

    y = 28
    for i, line in enumerate(lines):
        color = (230, 230, 230) if i < 3 else (130, 220, 130)
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 30

    if canvas.shape[1] < panel.shape[1]:
        panel = panel[:, : canvas.shape[1]]
    elif canvas.shape[1] > panel.shape[1]:
        pad = np.full(
            (panel_h, canvas.shape[1] - panel.shape[1], 3),
            (25, 25, 25),
            dtype=np.uint8,
        )
        panel = np.hstack([panel, pad])

    return np.vstack([panel, canvas])


def render_missing_image(text: str, width: int = 1024, height: int = 576) -> np.ndarray:
    frame = np.full((height, width, 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(
        frame,
        "Image not found",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (80, 80, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (40, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    args = parse_args()
    catalog_path = os.path.abspath(args.catalog)

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    entries = read_catalog(catalog_path)
    if not entries:
        print("Catalog has no valid entries.")
        return

    backup_path = catalog_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(catalog_path, backup_path)
        print(f"[BACKUP] Created: {backup_path}")

    print(
        f"Loaded {len(entries)} entries from {catalog_path}\n"
        "Controls: A/Left prev, D/Right next, X/Delete remove current, Q/Esc quit"
    )

    pos = 0
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    while True:
        if not entries:
            blank = np.full((260, 860, 3), (20, 20, 20), dtype=np.uint8)
            cv2.putText(
                blank,
                "Catalog is empty.",
                (35, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                blank,
                "Press Q or Esc to exit.",
                (35, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (170, 220, 170),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(args.window, blank)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            continue

        pos = max(0, min(pos, len(entries) - 1))
        entry = entries[pos]
        raw_path = str(entry.get("cam/image_array", ""))
        image_path = resolve_image_path(catalog_path, raw_path)

        img = cv2.imread(image_path)
        if img is None:
            frame = render_missing_image(image_path)
        else:
            frame = fit_image(img)

        composed = draw_overlay(frame, entry, pos, len(entries), image_path)
        cv2.imshow(args.window, composed)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("a"), 81):
            pos = (pos - 1) % len(entries)
            continue
        if key in (ord("d"), 83):
            pos = (pos + 1) % len(entries)
            continue
        if key in (ord("x"), 8, 127):
            removed = entries.pop(pos)
            removed_img = resolve_image_path(
                catalog_path,
                str(removed.get("cam/image_array", "")),
            )

            if os.path.exists(removed_img):
                try:
                    os.remove(removed_img)
                    print(f"[DELETE] Image removed: {removed_img}")
                except OSError as exc:
                    print(f"[WARN] Could not remove image '{removed_img}': {exc}")

            write_catalog(catalog_path, entries)
            print(f"[DELETE] Removed catalog entry at viewer position {pos + 1}.")

            if pos >= len(entries):
                pos = max(0, len(entries) - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

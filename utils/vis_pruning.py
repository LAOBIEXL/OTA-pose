from pathlib import Path

import cv2
import numpy as np


def _parse_display_grid(display_grid, grid_h: int, grid_w: int):
    if display_grid is None:
        return grid_h, grid_w
    if isinstance(display_grid, str):
        if display_grid.lower() in {"fine", "native", "none"}:
            return grid_h, grid_w
        if "x" not in display_grid.lower():
            raise ValueError("display_grid must look like '12x9' or 'fine'")
        parts = display_grid.lower().split("x")
        if len(parts) != 2:
            raise ValueError("display_grid must look like '12x9' or 'fine'")
        disp_h, disp_w = int(parts[0]), int(parts[1])
    else:
        disp_h, disp_w = display_grid
    if disp_h <= 0 or disp_w <= 0:
        raise ValueError("display_grid dimensions must be positive")
    return int(disp_h), int(disp_w)


def _normalize_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    low, high = np.percentile(score, [2, 98])
    if high <= low:
        low, high = float(score.min()), float(score.max())
    return np.clip((score - low) / (high - low + 1e-8), 0.0, 1.0)


def _indices_to_mask(indices: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= grid_h * grid_w):
        raise ValueError("keep_index contains invalid token indices")
    mask = np.zeros(grid_h * grid_w, dtype=np.float32)
    mask[indices] = 1.0
    return mask.reshape(grid_h, grid_w)


def _aggregate_grid(values: np.ndarray, display_h: int, display_w: int, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    grid_h, grid_w = values.shape
    if (display_h, display_w) == (grid_h, grid_w):
        return values.copy()

    out = np.zeros((display_h, display_w), dtype=np.float32)
    for row in range(display_h):
        y1 = int(np.floor(row * grid_h / display_h))
        y2 = int(np.ceil((row + 1) * grid_h / display_h))
        for col in range(display_w):
            x1 = int(np.floor(col * grid_w / display_w))
            x2 = int(np.ceil((col + 1) * grid_w / display_w))
            patch = values[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            if mode == "max":
                out[row, col] = float(patch.max())
            elif mode == "any":
                out[row, col] = float(patch.max() > 0.0)
            else:
                out[row, col] = float(patch.mean())
    return out


def _draw_grid(image_rgb: np.ndarray, grid_h: int, grid_w: int, color=(242, 246, 245), thickness=1):
    height, width = image_rgb.shape[:2]
    color = tuple(int(v) for v in color)
    for row in range(1, grid_h):
        y = round(row * height / grid_h)
        cv2.line(image_rgb, (0, y), (width, y), color, thickness, lineType=cv2.LINE_AA)
    for col in range(1, grid_w):
        x = round(col * width / grid_w)
        cv2.line(image_rgb, (x, 0), (x, height), color, thickness, lineType=cv2.LINE_AA)


def _importance_colormap(score_map: np.ndarray) -> np.ndarray:
    """Map normalized scores to a continuous purple-to-yellow palette."""
    score_map = np.asarray(score_map, dtype=np.float32)
    score_uint8 = np.uint8(np.clip(score_map, 0.0, 1.0) * 255)
    heatmap_bgr = cv2.applyColorMap(score_uint8, cv2.COLORMAP_PLASMA)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def save_importance_overlay(
    image_rgb: np.ndarray,
    score: np.ndarray,
    grid_h: int,
    grid_w: int,
    output_path: str,
    alpha: float = 0.58,
    keep_index=None,
    display_grid="12x9",
    keep_only: bool = True,
) -> None:
    """Overlay DPC token scores as a coarse paper-style patch heatmap."""
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    score = np.asarray(score).reshape(-1)

    if score.size != grid_h * grid_w:
        raise ValueError(f"score size {score.size} != grid size {grid_h}x{grid_w}")

    score_map = score.reshape(grid_h, grid_w).astype(np.float32)
    if keep_index is not None and keep_only:
        score_map = score_map * _indices_to_mask(keep_index, grid_h, grid_w)

    display_h, display_w = _parse_display_grid(display_grid, grid_h, grid_w)
    score_map = _aggregate_grid(score_map, display_h, display_w, mode="max")
    score_map = _normalize_score(score_map)
    score_map = np.power(score_map, 0.82)
    score_map = cv2.resize(score_map, (width, height), interpolation=cv2.INTER_LINEAR)

    heatmap_rgb = _importance_colormap(score_map).astype(np.float32)
    blue_tint = np.full_like(image_rgb, (39, 37, 94), dtype=np.uint8)
    base = cv2.addWeighted(image_rgb, 0.48, blue_tint, 0.52, 0).astype(np.float32)
    strength = (0.18 + 0.82 * score_map[..., None]) * float(alpha)
    overlay = base * (1.0 - strength) + heatmap_rgb * strength
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    _draw_grid(overlay, display_h, display_w, color=(160, 178, 202), thickness=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def render_pruning_overlay(
    image_rgb: np.ndarray,
    keep_index: np.ndarray,
    grid_h: int,
    grid_w: int,
    prune_alpha: float = 0.82,
    display_grid="12x9",
) -> np.ndarray:
    """Fade pruned regions and leave retained coarse patches visible."""
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    keep_mask = _indices_to_mask(keep_index, grid_h, grid_w)
    display_h, display_w = _parse_display_grid(display_grid, grid_h, grid_w)
    keep_mask = _aggregate_grid(keep_mask, display_h, display_w, mode="any").astype(bool)

    output = image_rgb.astype(np.float32).copy()
    white = np.full_like(output, 255.0)
    output = (1.0 - prune_alpha) * output + prune_alpha * white

    for row in range(display_h):
        for col in range(display_w):
            if not keep_mask[row, col]:
                continue
            y1 = round(row * height / display_h)
            y2 = round((row + 1) * height / display_h)
            x1 = round(col * width / display_w)
            x2 = round((col + 1) * width / display_w)
            output[y1:y2, x1:x2] = image_rgb[y1:y2, x1:x2].astype(np.float32)

    output = np.clip(output, 0, 255).astype(np.uint8)
    _draw_grid(output, display_h, display_w, color=(245, 248, 246), thickness=2)
    return output


def save_pruning_overlay(
    image_rgb: np.ndarray,
    keep_index: np.ndarray,
    grid_h: int,
    grid_w: int,
    output_path: str,
    prune_alpha: float = 0.82,
    display_grid="12x9",
) -> None:
    output = render_pruning_overlay(
        image_rgb=image_rgb,
        keep_index=keep_index,
        grid_h=grid_h,
        grid_w=grid_w,
        prune_alpha=prune_alpha,
        display_grid=display_grid,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

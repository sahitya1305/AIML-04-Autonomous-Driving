import cv2
import numpy as np


def tolerant_iou(pred: np.ndarray, gt: np.ndarray, tol_px: int = 6) -> float:
    """IoU after dilating both thin masks by tol_px (lane lines are 1-5 px wide, strict IoU would be meaningless)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    p, g = cv2.dilate((pred > 0).astype(np.uint8), k) > 0, cv2.dilate((gt > 0).astype(np.uint8), k) > 0
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else 1.0


def bottom_x_error(line, gt_mask: np.ndarray, side: str):
    """Horizontal error (px) of the fitted line at the image bottom vs the GT mask at the bottom-most populated row."""
    if line is None:
        return None
    h, w = gt_mask.shape
    for y in range(h - 1, int(h * 0.6), -1):
        xs = np.flatnonzero(gt_mask[y] > 0)
        xs = xs[xs < w / 2] if side == "left" else xs[xs >= w / 2]
        if xs.size:
            gt_x = xs.mean()
            (xb, yb), (xt, yt) = line
            x_at_y = xb + (xt - xb) * (yb - y) / (yb - yt)
            return float(abs(x_at_y - gt_x))
    return None

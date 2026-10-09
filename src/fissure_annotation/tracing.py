"""Snap coarse waypoints onto the nearest thin bright line: the classical-CV half of the pipeline.

Adapted from the exploratory session's reference_cv.py (Sato ridge filter + minimal-cost path), with the
parts that only suited one near-horizontal fissure on one slice removed:
- the 'horizontal curvature' term is gone (it ruled out oblique fissures); the Sato response is orientation-agnostic
- the per-column sub-pixel refine is gone (assumes one row per column); the path is smoothed along its arc length instead
- the lung mask comes from the lobe segmentation, not 'largest dark component under a threshold chosen by eye'

Everything here works in (row, col) pixels of the full isotropic slice, with `mm_per_px` for the reported distances.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sato
from skimage.graph import route_through_array

SIGMAS = [0.8, 1.2, 1.6]       # px; a fissure is 1-2 px wide at ~0.75 mm pixels
COST_EPS = 0.02                # cost = 1 / (score + eps): a ridge is cheap, empty lung costs ~50x more
LUNG_ERODE_PX = 2              # keep the tracer off the pleural edge, where the lung/chest-wall edge looks like a ridge
STRONG_RIDGE = 0.15            # a point counts as "on a ridge" above this normalised Sato score (to calibrate)
SMOOTH_MM = 5.0                # arc-length Gaussian smoothing; the first agent run's ~1.5 mm smoothing left zigzags the reviewers marked NEAR
SEARCH_MM = 8.0                # each waypoint may move to the strongest ridge within this radius before routing
MIN_RIDGE_TO_MOVE = 0.05       # in empty lung (e.g. a waypoint at the pleura, where the score is zeroed) the waypoint stays put


@dataclass
class Snap:
    path: np.ndarray          # N x 2 (row, col), smoothed, one point per pixel step
    stats: dict


def lung_mask(lobes: np.ndarray) -> np.ndarray:
    "union of the lobes with enclosed holes (vessels, bronchi) filled"
    return ndi.binary_fill_holes(ndi.binary_closing(lobes > 0, structure=np.ones((3, 3)), iterations=2))


def ridge_map(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    "bright thin-line response in 0..1, zero outside the (eroded) lung"
    a = image.astype(float) / 255.0
    score = sato(a, sigmas=SIGMAS, black_ridges=False)
    score[~ndi.binary_erosion(mask, iterations=LUNG_ERODE_PX)] = 0
    return score / max(score.max(), 1e-9)


def snap_waypoint(score: np.ndarray, p, direction: np.ndarray, radius_px: float) -> tuple[int, int]:
    """Move a waypoint along the normal to the line (`direction` = unit row/col vector of the line there) to the best
    ridge within `radius_px`. Searching only across the line keeps it from sliding along the fissure or jumping to a
    vessel that merely lies nearby; a mild distance penalty prefers the closest ridge. In empty lung it stays put."""
    h, w = score.shape
    r0, c0 = clamp(p, h, w)
    normal = np.array([-direction[1], direction[0]])
    steps = np.arange(-int(radius_px), int(radius_px) + 1)
    cand = np.array([clamp((r0 + t * normal[0], c0 + t * normal[1]), h, w) for t in steps])
    vals = score[cand[:, 0], cand[:, 1]] * (1 - 0.5 * np.abs(steps) / max(radius_px, 1))
    if vals.max() < MIN_RIDGE_TO_MOVE:
        return r0, c0
    return int(cand[vals.argmax(), 0]), int(cand[vals.argmax(), 1])


def local_directions(waypoints: list[tuple[float, float]]) -> list[np.ndarray]:
    pts = np.array(waypoints, dtype=float)
    out = []
    for i in range(len(pts)):
        d = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        out.append(d / max(np.hypot(*d), 1e-9))
    return out


def trace(score: np.ndarray, waypoints: list[tuple[int, int]]) -> np.ndarray:
    "minimal-cost path through the ordered (row, col) waypoints"
    cost = 1.0 / (score + COST_EPS)
    h, w = score.shape
    path = []
    for p, q in zip(waypoints[:-1], waypoints[1:]):
        p, q = clamp(p, h, w), clamp(q, h, w)
        seg, _ = route_through_array(cost, p, q, fully_connected=True, geometric=True)
        path.extend(seg if not path else seg[1:])
    return np.array(path, dtype=float)


def clamp(p, h: int, w: int) -> tuple[int, int]:
    return min(max(int(round(p[0])), 0), h - 1), min(max(int(round(p[1])), 0), w - 1)


def smooth(path: np.ndarray, sigma_px: float) -> np.ndarray:
    "Gaussian smoothing of each coordinate along the path; the end points stay put"
    out = ndi.gaussian_filter1d(path, sigma_px, axis=0, mode="nearest")
    out[0], out[-1] = path[0], path[-1]
    return out


def arc_length(path: np.ndarray) -> float:
    return float(np.hypot(*np.diff(path, axis=0).T).sum())


def sample(path: np.ndarray, step: float = 1.0) -> np.ndarray:
    "points along a polyline every `step` px"
    out = []
    for a, b in zip(path[:-1], path[1:]):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        out += [a + (b - a) * t / n for t in range(n)]
    return np.array(out + [path[-1]])


def mean_shift_mm(model_line: np.ndarray, snapped: np.ndarray, mm_per_px: float) -> tuple[float, float]:
    "mean and max distance from the model's polyline to the snapped path, in mm"
    pts = sample(model_line)
    d = np.sqrt(((pts[:, None, :] - snapped[None, :, :]) ** 2).sum(-1)).min(axis=1) * mm_per_px
    return float(d.mean()), float(d.max())


def snap(score: np.ndarray, waypoints: list[tuple[float, float]], mm_per_px: float) -> Snap:
    moved = [snap_waypoint(score, p, d, SEARCH_MM / mm_per_px) for p, d in zip(waypoints, local_directions(waypoints))]
    raw = trace(score, moved)
    path = smooth(raw, SMOOTH_MM / mm_per_px)
    ix = np.clip(np.round(path).astype(int), 0, np.array(score.shape) - 1)
    along = score[ix[:, 0], ix[:, 1]]
    mean_shift, max_shift = mean_shift_mm(np.array(waypoints, dtype=float), path, mm_per_px)
    chord = float(np.hypot(*(path[-1] - path[0])))
    return Snap(path, {
        "length_mm": round(arc_length(path) * mm_per_px, 1),
        "mean_ridge_score": round(float(along.mean()), 3),
        "fraction_on_ridge": round(float((along > STRONG_RIDGE).mean()), 2),
        "mean_shift_mm": round(mean_shift, 1),
        "max_shift_mm": round(max_shift, 1),
        "tortuosity": round(arc_length(path) / max(chord, 1.0), 2),
        "ridge_by_fifth": [round(float(x.mean()), 2) for x in np.array_split(along, 5)],
    })

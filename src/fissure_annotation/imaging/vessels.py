"""The vessel cross-sections of one slice, as one dot per vessel, for the model to see.

Why:
- On a slice, and more so on its ridge map, a vessel lying in the slice looks like a faint fissure.
- A dot on each vessel tells the model "this spot is a vessel", without hiding the pixels around it.

How:
1. The input is a vessel mask cut like the slice PNG (slices.slice_view), so a dot sits on the pixel it marks.
2. The mask is split into 8-connected components, and components under `min_area_px` are dropped.
   - They are mostly noise of the segmentation, and dots on them would clutter the picture.
3. The core of a component is its pixel farthest from the background: the maximum of the distance transform.
   - Unlike the centroid, it always lies inside the component, even for a curved or branching one.
4. A vessel lying in the slice is one long component, so its single core marks only one spot of its length.
   - Worse, the hilar vessels join the big vessels of the mediastinum, so a whole tree can get one dot outside the lung.
   - `vessel_dots_along_skeleton` adds dots along each component's centre line, at most `spacing_px` apart, when that is wanted.
   - `elongation` measures how long a component is, to see how often that happens.
5. Dots are drawn as filled discs of one colour. A grey picture becomes RGB, and an RGB one keeps what is on it, so dots go on top of other guides.
"""

import numpy as np
from jaxtyping import Bool, UInt8
from scipy import ndimage
from skimage.morphology import skeletonize

Point = tuple[int, int]  # (x, y) = (column, row)
Colour = tuple[int, int, int]

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def vessel_components(mask: Bool[np.ndarray, "h w"], min_area_px: int) -> tuple[np.ndarray, list[int]]:
    "the label image and the labels of the components that are big enough"
    labels, count = ndimage.label(mask, structure=EIGHT_CONNECTED)
    areas = ndimage.sum_labels(mask, labels, index=np.arange(1, count + 1))
    kept = [label for label, area in zip(range(1, count + 1), areas) if area >= min_area_px]
    return labels, kept


def vessel_cores(mask: Bool[np.ndarray, "h w"], min_area_px: int) -> list[Point]:
    labels, kept = vessel_components(mask, min_area_px)
    if not kept:
        return []
    depth = ndimage.distance_transform_edt(mask)
    positions = ndimage.maximum_position(depth, labels, index=kept)
    return [(int(column), int(row)) for row, column in positions]


def elongation(component: Bool[np.ndarray, "h w"]) -> float:
    "the long axis over the short axis of the pixels' spread; 1 for a disc, large for a line"
    rows, columns = np.nonzero(component)
    if len(rows) < 3:
        return 1.0
    spread = np.linalg.eigvalsh(np.cov(np.stack([rows, columns]).astype(float)))
    short, long = np.sqrt(np.clip(spread, 0.25, None))  # a 1 px wide line still has some width
    return float(long / short)


def component_elongations(mask: Bool[np.ndarray, "h w"], min_area_px: int) -> list[float]:
    labels, kept = vessel_components(mask, min_area_px)
    return [elongation(labels == label) for label in kept]


def vessel_dots_along_skeleton(mask: Bool[np.ndarray, "h w"], min_area_px: int, spacing_px: float) -> list[Point]:
    """the core of each component, then more dots on its skeleton until every skeleton pixel is within half spacing_px of a dot

    The extra dots are the skeleton pixels farthest from every dot so far (farthest point sampling), so they are evenly spread.
    - A round component keeps its single core.
    - A long vessel or a branching tree gets a dot every half to whole spacing_px along each branch.
    Distances are straight lines, not along the skeleton, so two branches closer than spacing_px can share a dot.
    """
    labels, kept = vessel_components(mask, min_area_px)
    depth = ndimage.distance_transform_edt(mask)
    boxes = ndimage.find_objects(labels)
    dots = []
    for label in kept:
        box = boxes[label - 1]
        component = labels[box] == label
        core = np.unravel_index(np.argmax(np.where(component, depth[box], -1)), component.shape)
        offset = np.array([box[0].start, box[1].start])
        chosen = farthest_point_dots(np.argwhere(skeletonize(component)), np.array(core), spacing_px / 2)
        dots += [(int(column), int(row)) for row, column in chosen + offset]
    return dots


def farthest_point_dots(points: np.ndarray, start: np.ndarray, reach_px: float) -> np.ndarray:
    "start, then the point farthest from every chosen one, while it is more than reach_px away; rows of (row, column)"
    chosen = [start]
    distance = np.hypot(*(points - start).T) if len(points) else np.zeros(0)
    while len(points) and distance.max() > reach_px:
        farthest = points[np.argmax(distance)]
        chosen.append(farthest)
        distance = np.minimum(distance, np.hypot(*(points - farthest).T))
    return np.array(chosen)


def disc_offsets(radius_px: float) -> list[tuple[int, int]]:
    reach = int(np.ceil(radius_px))
    return [
        (d_row, d_column)
        for d_row in range(-reach, reach + 1)
        for d_column in range(-reach, reach + 1)
        if d_row**2 + d_column**2 <= radius_px**2
    ]


def draw_dots(
    picture: UInt8[np.ndarray, "h w *channels"], points: list[Point], colour: Colour, radius_px: float
) -> UInt8[np.ndarray, "h w 3"]:
    rgb = np.stack([picture] * 3, axis=-1) if picture.ndim == 2 else picture.copy()
    height, width = rgb.shape[:2]
    for x, y in points:
        for d_row, d_column in disc_offsets(radius_px):
            row, column = y + d_row, x + d_column
            if 0 <= row < height and 0 <= column < width:
                rgb[row, column] = colour
    return rgb


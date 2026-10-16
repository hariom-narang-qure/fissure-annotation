"""Look at a fissure from the side, as a wall: the CT and a mask flattened onto a smooth surface through the fissure.

A fissure is a thin sheet, so seen along its normal it is a wall.
Axial, coronal and sagittal slices only ever cut it as a line, so they cannot show how a mask is spread over the whole sheet.

Steps, and why:
- Points: the voxels of the mask that belong to this fissure, in mm, one point per BIN_MM cube.
  - Traces are dense along a slice and sparse between slices, binning keeps the fit from favouring the traced direction.
- Frame: PCA of the points gives the best plane (u, v) and its normal w, a fissure is nearly a height field w = f(u, v) over that plane.
- Surface: f is a thin-plate spline with smoothing SMOOTHING, so the surface bridges gaps in the mask and does not chase 1 voxel jitter.
  - The value 30 won blocked cross-validation for all three fissures on the first scan (fusion LOG, step 5).
- Wall image: each (u, v) pixel samples the CT at the surface, averaged over the slab of +-SLAB_HALF_MM along the local normal.
  - A fissure is slightly brighter than parenchyma, so a thin slab shows it as a bright, vessel-poor sheet and a thick one washes it out.
- Mask on the wall: a mask voxel lands on the pixel of its (u, v) if it is within MASK_REACH_MM of the surface along w.
  - Its distance to the surface is kept, so the mask can be coloured by how far it sits from the fitted wall.
- Extent: pixels farther than EXTEND_MM in (u, v) from every point are left blank, so the sheet is not extrapolated across the lung.
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Bool, Float, UInt8
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

BIN_MM = 2.0
SMOOTHING = 30.0
PIXEL_MM = 0.8
EXTEND_MM = 8.0
SLAB_HALF_MM = 1.0
SLAB_STEP_MM = 0.4
MASK_REACH_MM = 3.0
LUNG_WINDOW_HU = (-1100.0, -500.0)

Points = Float[np.ndarray, "n 3"]


class WallImages(NamedTuple):
    ct: UInt8[np.ndarray, "v u"]
    mask_offset_mm: Float[np.ndarray, "v u"]  # NaN where no mask voxel landed, else the signed distance to the surface
    valid: Bool[np.ndarray, "v u"]


def bin_points(points_mm: Points) -> Points:
    keys = np.floor(points_mm / BIN_MM).astype(int)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    sums = np.zeros((inverse.max() + 1, 3))
    np.add.at(sums, inverse, points_mm)
    return sums / np.bincount(inverse)[:, None]


class FissureWall:
    def __init__(self, points_mm: Points):
        binned = bin_points(points_mm)
        self.centre = binned.mean(0)
        _, _, axes = np.linalg.svd(binned - self.centre, full_matrices=False)
        self.axes = axes  # rows are u, v, w, with w the direction of least spread
        local = self.to_frame(binned)
        self.height = RBFInterpolator(
            local[:, :2], local[:, 2], kernel="thin_plate_spline", smoothing=SMOOTHING
        )
        self.near_points = cKDTree(local[:, :2])
        self.u_range = (local[:, 0].min() - EXTEND_MM, local[:, 0].max() + EXTEND_MM)
        self.v_range = (local[:, 1].min() - EXTEND_MM, local[:, 1].max() + EXTEND_MM)

    def to_frame(self, points_mm: Points) -> Points:
        return (points_mm - self.centre) @ self.axes.T

    def from_frame(self, local: Points) -> Points:
        return local @ self.axes + self.centre

    def pixel_grid(self) -> tuple[Float[np.ndarray, "v u"], Float[np.ndarray, "v u"]]:
        u = np.arange(*self.u_range, PIXEL_MM)
        v = np.arange(*self.v_range, PIXEL_MM)
        return np.meshgrid(u, v)

    def surface_normals(
        self, uu: Float[np.ndarray, "v u"], vv: Float[np.ndarray, "v u"]
    ) -> Float[np.ndarray, "v u 3"]:
        "normal of w = f(u, v) is (-df/du, -df/dv, 1), found by central differences of one pixel"
        h = lambda du, dv: self.surface_height(uu + du, vv + dv)
        dfdu = (h(PIXEL_MM, 0) - h(-PIXEL_MM, 0)) / (2 * PIXEL_MM)
        dfdv = (h(0, PIXEL_MM) - h(0, -PIXEL_MM)) / (2 * PIXEL_MM)
        local = np.stack([-dfdu, -dfdv, np.ones_like(dfdu)], axis=-1)
        return (local / np.linalg.norm(local, axis=-1, keepdims=True)) @ self.axes

    def surface_height(self, uu, vv):
        return self.height(np.stack([uu.ravel(), vv.ravel()], axis=-1)).reshape(
            uu.shape
        )

    def pixels_with_support(self, uu, vv) -> Bool[np.ndarray, "v u"]:
        distance, _ = self.near_points.query(np.stack([uu.ravel(), vv.ravel()], axis=-1))
        return (distance <= EXTEND_MM).reshape(uu.shape)

    def flatten(
        self,
        volume_hu: Float[np.ndarray, "x y z"],
        spacing_mm: tuple[float, float, float],
        mask_points_mm: Points,
    ) -> WallImages:
        uu, vv = self.pixel_grid()
        height = self.surface_height(uu, vv)
        surface = self.from_frame(np.stack([uu, vv, height], axis=-1))
        normals = self.surface_normals(uu, vv)
        valid = self.pixels_with_support(uu, vv)
        return WallImages(
            ct=self.slab_image(volume_hu, spacing_mm, surface, normals, valid),
            mask_offset_mm=self.mask_offsets(mask_points_mm, uu, vv, height),
            valid=valid,
        )

    def slab_image(self, volume_hu, spacing_mm, surface, normals, valid):
        offsets = np.arange(-SLAB_HALF_MM, SLAB_HALF_MM + 1e-6, SLAB_STEP_MM)
        samples = [
            ndimage.map_coordinates(
                volume_hu,
                list((surface + d * normals).reshape(-1, 3).T / np.array(spacing_mm)[:, None]),
                order=1,
                mode="nearest",
            ).reshape(valid.shape)
            for d in offsets
        ]
        image = to_lung_window(np.mean(samples, axis=0))
        image[~valid] = 0
        return image

    def mask_offsets(self, mask_points_mm, uu, vv, height) -> Float[np.ndarray, "v u"]:
        local = self.to_frame(mask_points_mm)
        column = np.round((local[:, 0] - self.u_range[0]) / PIXEL_MM).astype(int)
        row = np.round((local[:, 1] - self.v_range[0]) / PIXEL_MM).astype(int)
        inside = (
            (column >= 0) & (column < uu.shape[1]) & (row >= 0) & (row < uu.shape[0])
        )
        column, row, local = column[inside], row[inside], local[inside]
        offset = local[:, 2] - height[row, column]
        close = np.abs(offset) <= MASK_REACH_MM
        out = np.full(uu.shape, np.nan)
        closest_last = np.argsort(-np.abs(offset[close]))  # so the nearest voxel of a pixel is written last and wins
        out[row[close][closest_last], column[close][closest_last]] = offset[close][closest_last]
        return out


def to_lung_window(hu: Float[np.ndarray, "..."]) -> UInt8[np.ndarray, "..."]:
    low, high = LUNG_WINDOW_HU
    return (np.clip((hu - low) / (high - low), 0, 1) * 255).astype(np.uint8)

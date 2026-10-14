"""Turn a ragged fissure mask into one thin sheet by blurring along the sheet only.

Why:
- The model traces each 2D slice on its own, so the stacked 3D mask is a set of lines with holes between them.
  - The holes are largest where the fissure lies almost flat in the slice plane.
- A round blur fills the holes only when it is large, and a large round blur also pushes the sheet sideways.
- A blur that is long along the sheet and short across it fills the holes and keeps the sheet where it was drawn.

How:
1. Sheet normal at every mask voxel by local PCA.
   - Gaussian weighted covariance of the mask voxels around it, at scale `sigma_pca`.
   - The eigenvector of the smallest eigenvalue is the normal.
2. Splat one oriented Gaussian per mask voxel, tilted to its normal.
   - Long along the sheet (`sigma_along`), short across it (`sigma_across`).
   - The short blur across also averages the slice to slice jitter.
   - Each kernel has unit mass, so the density counts mask voxels.
   - The outer product of the normal is splatted with the same weights.
     Averaged over the neighbourhood it has no sign ambiguity, unlike the normal itself.
3. At every voxel with enough density, the normal is the top eigenvector of the splatted tensor.
   - `min_strength` is relative to the density of a fully marked one voxel plane.
   - Thin lines and specks never reach it, which is how most stray marks disappear.
4. Keep only the crest of the density along that normal (non-maximum suppression, as in the Canny edge detector).
   - This leaves a sheet one voxel thick.

Only the mask is used and the scan is never looked at, so the sheet cannot be pulled onto vessels or band edges.
Voxels are treated as isotropic, which is close enough for the usual 0.8 mm CT grid.
Junctions and the ends of fissures are not handled specially, the PCA normal is unreliable there.
"""

import numpy as np
from jaxtyping import Bool, Float
from scipy import ndimage

VoxelMask = Bool[np.ndarray, "x y z"]

CHUNK = 2000  # mask voxels splatted at once, which keeps the kernel by chunk arrays near 10M entries
KERNEL_RADIUS_IN_SIGMAS = 2.5
UPPER_TRIANGLE = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

DEFAULT_SIGMA_PCA_VOXELS = 3.0
DEFAULT_SIGMA_ALONG_VOXELS = 3.0
DEFAULT_SIGMA_ACROSS_VOXELS = 1.0
DEFAULT_MIN_STRENGTH = 0.3


def bounding_box(mask: VoxelMask) -> tuple[slice, ...]:
    return tuple(slice(int(c.min()), int(c.max()) + 1) for c in np.nonzero(mask))


def coordinate_grids(shape: tuple[int, ...]) -> list[np.ndarray]:
    return [
        np.arange(n, dtype=np.float64).reshape(
            [-1 if i == axis else 1 for i in range(3)]
        )
        for axis, n in enumerate(shape)
    ]


def local_covariance(
    mask: VoxelMask, points: tuple[np.ndarray, ...], sigma: float
) -> Float[np.ndarray, "n 3 3"]:
    density = mask.astype(np.float64)
    grids = coordinate_grids(mask.shape)

    def smoothed_at_points(values: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter(values, sigma)[points]

    weight = smoothed_at_points(density)
    mean = [smoothed_at_points(grid * density) / weight for grid in grids]
    covariance = np.empty((len(weight), 3, 3))
    for a, b in UPPER_TRIANGLE:
        second_moment = smoothed_at_points(grids[a] * grids[b] * density) / weight
        covariance[:, a, b] = covariance[:, b, a] = second_moment - mean[a] * mean[b]
    return covariance


def smallest_eigenvector(
    matrices: Float[np.ndarray, "n 3 3"],
) -> Float[np.ndarray, "n 3"]:
    return np.linalg.eigh(matrices)[1][:, :, 0]


def largest_eigenvector(
    matrices: Float[np.ndarray, "n 3 3"],
) -> Float[np.ndarray, "n 3"]:
    return np.linalg.eigh(matrices)[1][:, :, -1]


def kernel_offsets(radius: int) -> Float[np.ndarray, "k 3"]:
    axis = np.arange(-radius, radius + 1)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)


def oriented_weights(
    offsets: np.ndarray,
    normals: Float[np.ndarray, "chunk 3"],
    sigma_along: float,
    sigma_across: float,
) -> Float[np.ndarray, "k chunk"]:
    across = offsets @ normals.T
    along_squared = np.maximum((offsets**2).sum(1)[:, None] - across**2, 0)
    weights = np.exp(
        -0.5 * (along_squared / sigma_along**2 + across**2 / sigma_across**2)
    )
    return weights / weights.sum(0)


def splat_oriented(
    shape: tuple[int, ...],
    points: np.ndarray,
    normals: Float[np.ndarray, "n 3"],
    sigma_along: float,
    sigma_across: float,
) -> tuple[Float[np.ndarray, "x y z"], Float[np.ndarray, "6 x y z"]]:
    "density of the splatted kernels, and the splatted outer product of the normals"
    offsets = kernel_offsets(int(np.ceil(KERNEL_RADIUS_IN_SIGMAS * sigma_along)))
    size = int(np.prod(shape))
    density = np.zeros(size)
    tensor = np.zeros((len(UPPER_TRIANGLE), size))
    for start in range(0, len(points), CHUNK):
        chunk_points = points[start : start + CHUNK]
        chunk_normals = normals[start : start + CHUNK]
        weights = oriented_weights(offsets, chunk_normals, sigma_along, sigma_across)
        targets = chunk_points[None, :, :] + offsets[:, None, :]
        flat = np.ravel_multi_index(targets.reshape(-1, 3).T, shape)
        density += np.bincount(flat, weights.ravel(), minlength=size)
        for row, (a, b) in enumerate(UPPER_TRIANGLE):
            product = chunk_normals[:, a] * chunk_normals[:, b]
            tensor[row] += np.bincount(
                flat, (weights * product).ravel(), minlength=size
            )
    return density.reshape(shape), tensor.reshape((len(UPPER_TRIANGLE), *shape))


def tensors_at(
    tensor: Float[np.ndarray, "6 x y z"], points: tuple[np.ndarray, ...]
) -> Float[np.ndarray, "n 3 3"]:
    matrices = np.empty((len(points[0]), 3, 3))
    for row, (a, b) in enumerate(UPPER_TRIANGLE):
        matrices[:, a, b] = matrices[:, b, a] = tensor[row][points]
    return matrices


def flat_sheet_density(sigma_across: float) -> float:
    "density at the centre of a fully marked one voxel plane after the blur across it"
    return 1.0 / (np.sqrt(2 * np.pi) * sigma_across)


def is_crest(
    density: Float[np.ndarray, "x y z"],
    points: tuple[np.ndarray, ...],
    normals: Float[np.ndarray, "n 3"],
) -> Bool[np.ndarray, " n"]:
    centre = np.stack(points, axis=1).astype(np.float32)
    ahead = ndimage.map_coordinates(density, (centre + normals).T, order=1)
    behind = ndimage.map_coordinates(density, (centre - normals).T, order=1)
    here = density[points]
    # >= on one side and > on the other, so a flat top keeps exactly one voxel
    return (here >= ahead) & (here > behind)


def sheet_in_padded_crop(
    crop: VoxelMask,
    sigma_pca: float,
    sigma_along: float,
    sigma_across: float,
    min_strength: float,
) -> VoxelMask:
    points = np.nonzero(crop)
    normals = smallest_eigenvector(local_covariance(crop, points, sigma_pca))
    density, tensor = splat_oriented(
        crop.shape, np.stack(points, 1), normals, sigma_along, sigma_across
    )
    candidates = np.nonzero(density > min_strength * flat_sheet_density(sigma_across))
    crest_normals = largest_eigenvector(tensors_at(tensor, candidates))
    is_kept = is_crest(
        density.astype(np.float32), candidates, crest_normals.astype(np.float32)
    )
    sheet = np.zeros(crop.shape, dtype=bool)
    sheet[tuple(c[is_kept] for c in candidates)] = True
    return sheet


def oriented_sheet(
    mask: VoxelMask,
    sigma_pca: float = DEFAULT_SIGMA_PCA_VOXELS,
    sigma_along: float = DEFAULT_SIGMA_ALONG_VOXELS,
    sigma_across: float = DEFAULT_SIGMA_ACROSS_VOXELS,
    min_strength: float = DEFAULT_MIN_STRENGTH,
) -> VoxelMask:
    if not mask.any():
        return np.zeros_like(mask)
    # Zero padding keeps every kernel inside the array, also for a mask that touches the volume edge.
    pad = (
        max(int(np.ceil(KERNEL_RADIUS_IN_SIGMAS * sigma_along)), int(3 * sigma_pca)) + 2
    )
    box = bounding_box(mask)
    padded_crop = np.pad(mask[box], pad)
    sheet_crop = sheet_in_padded_crop(
        padded_crop, sigma_pca, sigma_along, sigma_across, min_strength
    )
    sheet = np.zeros_like(mask)
    sheet[box] = sheet_crop[pad:-pad, pad:-pad, pad:-pad]
    return sheet

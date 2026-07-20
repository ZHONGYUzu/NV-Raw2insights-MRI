"""Lightweight fastMRI k-space preprocessing without MONAI imports."""

from __future__ import annotations

import numpy as np
from scipy.fft import fftn, fftshift, ifftn, ifftshift


def crop_kspace_via_image_domain(array: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
    """Remove readout oversampling using the fastMRI image-domain crop convention."""
    if array.ndim < 2:
        raise ValueError(f"Expected at least two spatial dimensions, got {array.shape}")
    if any(target > source for source, target in zip(array.shape[-2:], spatial_shape)):
        raise ValueError(f"Cannot image-domain crop k-space shape {array.shape[-2:]} to {spatial_shape}")

    spatial_axes = (-2, -1)
    coil_images = fftshift(
        ifftn(ifftshift(array, axes=spatial_axes), axes=spatial_axes, norm="ortho"),
        axes=spatial_axes,
    )
    crop_slices = []
    for source_size, target_size in zip(coil_images.shape[-2:], spatial_shape):
        start = (source_size - target_size) // 2
        crop_slices.append(slice(start, start + target_size))
    coil_images = coil_images[..., crop_slices[0], crop_slices[1]]
    cropped_kspace = fftshift(
        fftn(ifftshift(coil_images, axes=spatial_axes), axes=spatial_axes, norm="ortho"),
        axes=spatial_axes,
    )
    return cropped_kspace.astype(array.dtype, copy=False)

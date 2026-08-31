# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Sequence

import numpy as np
import scipy
from monai.config import PathLike
from monai.data.image_reader import ImageReader
from monai.data.utils import is_supported_format
from monai.utils import StrEnum, optional_import, require_pkg
from numpy import ndarray
from scipy.fft import fftn, fftshift, ifftn, ifftshift

from fastmri_preprocessing import crop_kspace_via_image_domain

h5py, has_h5py = optional_import("h5py")

__all__ = ["FastMRIReader", "CestMRIReader", "CMRxReconReader"]


class FastMRIKeys(StrEnum):
    """
    The keys to be used for extracting data from the fastMRI dataset
    """

    KSPACE = "kspace"
    MASK = "mask"
    FILENAME = "filename"
    RECON = "reconstruction_rss"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"


class CestMRIKeys(StrEnum):
    """
    The keys to be used for extracting data from the CestMRI dataset
    """

    KSPACE = "kspace"
    MASK = "mask"
    FILENAME = "filename"
    CSM = "sensitivity_maps"
    RECON_RSS = "reconstruction_rss"
    RECON_SENSE = "reconstruction_sense"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"
    SHAPE = "shape"


class CMRxReconKeys(StrEnum):
    """
    The keys to be used for extracting data from the CMRxRecon dataset
    """

    KSPACE = "kspace_full"
    MASK = "mask"
    MASK_TYPE = "mask_type"
    FILENAME = "filename"
    RECON = "reconstruction_rss"
    RECON_RAW = "reconstruction"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"
    NUM_SLICES = "num_slices"
    NUM_COILS = "num_coils"
    NUM_FRAMES = "num_frames"
    SHAPE = "shape"
    SMAP = "sensitivity_maps"


@require_pkg(pkg_name="h5py")
class FastMRIReader(ImageReader):
    """
    Load fastMRI files with '.h5' suffix. fastMRI files, when loaded with "h5py",
    are HDF5 dictionary-like datasets. The keys are:

    - kspace: contains the fully-sampled kspace
    - reconstruction_rss: contains the root sum of squares of ifft of kspace. This
        is the ground-truth image.

    It also has several attributes with the following keys:

    - acquisition (str): acquisition mode of the data (e.g., AXT2 denotes T2 brain MRI scans)
    - max (float): dynamic range of the data
    - norm (float): norm of the kspace
    - patient_id (str): the patient's id whose measurements were recorded
    """

    def __init__(self, reconstruction_size: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.reconstruction_size = reconstruction_size

    @staticmethod
    def crop_kspace_via_image_domain(array: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
        return crop_kspace_via_image_domain(array, spatial_shape)

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
         Verify whether the specified file format is supported by h5py reader.

        Args:
             filename: file name
        """
        suffixes: Sequence[str] = [".h5"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike) -> dict:  # type: ignore
        """
        Read data from specified h5 file.
        Note that the returned object is a dictionary.

        Args:
            data: file name to read.
        """
        if isinstance(data, (tuple, list)):
            data = data[0]

        with h5py.File(data, "r") as f:
            # extract everything from the ht5 file
            dat = dict(
                [(key, f[key][()]) for key in f]
                + [(key, f.attrs[key]) for key in f.attrs]
                + [(FastMRIKeys.FILENAME, os.path.basename(data))]  # type: ignore
            )
        f.close()

        return dat

    def get_data(self, dat: dict) -> tuple[ndarray, dict]:
        """
        Extract data array and metadata from the loaded data and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            dat: a dictionary loaded from an h5 file
        """
        header = self._get_meta_dict(dat)
        kspace = np.asarray(dat[FastMRIKeys.KSPACE])
        header["source_shape"] = np.asarray(kspace.shape)
        if self.reconstruction_size is not None:
            reconstruction_size = self.reconstruction_size
        elif FastMRIKeys.RECON in dat:
            reconstruction_size = tuple(int(size) for size in np.asarray(dat[FastMRIKeys.RECON]).shape[-2:])
        else:
            raise ValueError(
                "FastMRIReader requires reconstruction_size when reconstruction_rss is unavailable."
            )
        kspace = self.crop_kspace_via_image_domain(kspace, reconstruction_size)
        header["reconstruction_size"] = np.asarray(reconstruction_size)
        data: ndarray = kspace[np.newaxis, ...]
        data_shape = data.shape
        header[CMRxReconKeys.NUM_FRAMES] = data_shape[0]
        header[CMRxReconKeys.NUM_SLICES] = data_shape[1]
        header[CMRxReconKeys.NUM_COILS] = data_shape[2]
        header[CMRxReconKeys.SHAPE] = np.array(data_shape)
        mask = np.ones([1] * data.ndim)
        header[CMRxReconKeys.MASK] = mask.astype(np.float32)
        acquisition = header[FastMRIKeys.ACQUISITION]
        if isinstance(acquisition, (bytes, np.bytes_)):
            acquisition = acquisition.decode()
        acquisition = str(acquisition)
        header["acquisition_raw"] = acquisition
        acquisition_mapping = {
            "AXT1": "T1w",
            "AXT1PRE": "T1w",
            "AXT1POST": "T1w",
            "AXT2": "T2w",
            "AXFLAIR": "T2w",
        }
        if acquisition not in acquisition_mapping:
            raise ValueError(
                f"Unsupported fastMRI acquisition {acquisition!r}; "
                f"supported brain acquisitions are {sorted(acquisition_mapping)}"
            )
        header[FastMRIKeys.ACQUISITION] = acquisition_mapping[acquisition]
        return data, header

    def _get_meta_dict(self, dat: dict) -> dict:
        """
        Get all the metadata of the loaded dict and return the meta dict.

        Args:
            dat: a dictionary object loaded from an h5 file.
        """
        return {k.value: dat[k.value] for k in FastMRIKeys if k.value in dat}


class CestMRIReader(ImageReader):
    """
    Load fastMRI files with '.h5' suffix. fastMRI files, when loaded with "h5py",
    are HDF5 dictionary-like datasets. The keys are:

    - kspace: contains the fully-sampled kspace
    - reconstruction_rss: contains the root sum of squares of ifft of kspace. This
        is the ground-truth image.

    It also has several attributes with the following keys:

    - acquisition (str): acquisition mode of the data (e.g., AXT2 denotes T2 brain MRI scans)
    - max (float): dynamic range of the data
    - norm (float): norm of the kspace
    - patient_id (str): the patient's id whose measurements were recorded
    """

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
         Verify whether the specified file format is supported by h5py reader.

        Args:
             filename: file name
        """
        suffixes: Sequence[str] = [".h5"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike) -> dict:  # type: ignore
        """
        Read data from specified h5 file.
        Note that the returned object is a dictionary.

        Args:
            data: file name to read.
        """
        if isinstance(data, (tuple, list)):
            data = data[0]

        with h5py.File(data, "r") as f:
            # extract everything from the ht5 file
            dat = dict(
                [(key, f[key][()]) for key in f]
                + [(key, f.attrs[key]) for key in f.attrs]
                + [(CestMRIKeys.FILENAME, os.path.basename(data))]  # type: ignore
            )
        f.close()

        return dat

    def get_data(self, dat: dict) -> tuple[ndarray, dict]:
        """
        Extract data array and metadata from the loaded data and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            dat: a dictionary loaded from an h5 file
        """
        header = self._get_meta_dict(dat)
        data: ndarray = np.array(dat[CestMRIKeys.KSPACE])
        data = fftshift(
            ifftn(ifftshift(data, axes=[-3, -2, -1]), axes=[-3, -2, -1], norm="ortho"),
            axes=[-3, -2, -1],
        ).transpose(0, 4, 1, 2, 3)
        data = fftshift(
            fftn(ifftshift(data, axes=[-2, -1]), axes=[-2, -1], norm="ortho"),
            axes=[-2, -1],
        )
        header[CestMRIKeys.MASK] = (
            np.array(dat[CestMRIKeys.MASK]) if CestMRIKeys.MASK in dat.keys() else np.zeros(data.shape)
        )
        header[CestMRIKeys.SHAPE] = np.array(data.shape)
        return data, header

    def _get_meta_dict(self, dat: dict) -> dict:
        """
        Get all the metadata of the loaded dict and return the meta dict.

        Args:
            dat: a dictionary object loaded from an h5 file.
        """
        return {k.value: dat[k.value] for k in CestMRIKeys if k.value in dat}


class CMRxReconReader(ImageReader):
    def __init__(self, fixed_mask_types=None):
        super().__init__()
        self.fixed_mask_types = fixed_mask_types if isinstance(fixed_mask_types, list) else [fixed_mask_types]

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
         Verify whether the specified file format is supported by h5py reader.

        Args:
             filename: file name
        """
        suffixes: Sequence[str] = [".json"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read_mat(self, mat_file: Sequence[PathLike], include_keys: Sequence[str] | None = None) -> list:
        try:
            with h5py.File(mat_file, "r", swmr=True) as f:
                keys = list(include_keys) if include_keys else list(f)
                data_kv = []
                for key in keys:
                    if key not in f:
                        continue
                    if key in ("kSpace", "dMap") and np.issubdtype(f[key].dtype, np.complexfloating):
                        data_kv.append((key, f[key].astype(np.complex64)[()]))
                    else:
                        data_kv.append((key, f[key][()]))
        except BaseException:
            data = scipy.io.loadmat(mat_file)
            keys = include_keys if include_keys else data.keys()
            data_kv = [(key, data[key]) for key in keys if key in data]

        return data_kv

    def read_first_existing(self, paths) -> list:
        if not paths:
            return []
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        for path in paths:
            if path:
                return self.read_mat(path)
        return []

    def filter_masks_by_types(self, masks, fixed_mask_types):
        result = []
        if not all(fixed_mask_types):
            return masks
        for mask in masks:
            if any(mask_type in mask for mask_type in fixed_mask_types):
                result.append(mask)

        return result

    def read(self, data: Sequence[PathLike] | PathLike) -> dict:  # type: ignore
        """
        Read data from specified json file.
        Note that the returned object is a dictionary.

        Args:
            data: file name to read.
        """
        if isinstance(data, (tuple, list)):
            data = data[0]

        with open(data, "r") as f:
            json_data = json.load(f)
            kspace = json_data["kspace"]
            masks = self.filter_masks_by_types(json_data["mask"], self.fixed_mask_types)
            if json_data["mask"] and not masks:
                raise ValueError(
                    f"No masks matched fixed_mask_types={self.fixed_mask_types}; "
                    "update --fixed-mask-types/config or disable filtering with --fixed-mask-types none"
                )
            mask = random.choice(masks) if json_data["mask"] else ""
            mask_type = mask.split("_mask_")[-1][:-4]
            acquisition_match = re.search(r"(?:^|[/\\])MultiCoil[/\\]([^/\\]+)", kspace, flags=re.I)
            acquisition_type = json_data.get("acquisition") or (acquisition_match.group(1) if acquisition_match else None)
            if acquisition_type is None:
                raise ValueError(
                    f"Cannot infer acquisition from {kspace}; add an acquisition field to the descriptor"
                )
            smap = (
                json_data.get("sensitivity_maps")
                or json_data.get("smap")
                or json_data.get("dMap")
            )
            kspace_key = json_data.get("kspace_key") or json_data.get("raw_h5_kspace_key")

        kspace_kv = self.read_mat(kspace, include_keys=[kspace_key] if kspace_key else None)
        mask_kv = self.read_mat(mask) if mask else [(None, None)]
        smap_kv = self.read_first_existing(smap)

        dat = dict(
            kspace_kv
            + mask_kv
            + smap_kv
            + [
                (CMRxReconKeys.FILENAME, os.path.basename(data)),
                (CMRxReconKeys.MASK_TYPE, mask_type),
                (CMRxReconKeys.ACQUISITION, acquisition_type),
            ]
        )
        return dat

    def get_data(self, dat: dict) -> tuple[ndarray, dict]:
        """
        Extract data array and metadata from the loaded data and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            dat: a dictionary loaded from a mat file
        """
        header = self._get_meta_dict(dat)
        if "kus" in dat:
            kspace_key = "kus"
        elif CMRxReconKeys.KSPACE in dat:
            kspace_key = CMRxReconKeys.KSPACE
        elif "kspace" in dat:
            kspace_key = "kspace"
        elif "kSpace" in dat:
            kspace_key = "kSpace"
        else:
            raise ValueError("Expected one of kus, kspace_full, kspace, or raw H5 kSpace")

        if kspace_key == "kSpace":
            if not np.issubdtype(dat[kspace_key].dtype, np.complexfloating):
                raise ValueError(f"raw H5 {kspace_key} must be complex data, got dtype={dat[kspace_key].dtype}")
            if dat[kspace_key].ndim != 5:
                raise ValueError(
                    f"raw H5 {kspace_key} must have shape (slice, coil, time, PE, FE), got {dat[kspace_key].shape}"
                )
            data: ndarray = np.transpose(dat[kspace_key], (2, 0, 1, 3, 4))
            data_shape = data.shape
        elif np.issubdtype(
            dat[kspace_key].dtype, np.complexfloating
        ):  # return from scipy.io.loadmat is complex ndarray with transposed shape
            data_shape = dat[kspace_key].shape[::-1]
            data_shape = (1,) * (5 - len(data_shape)) + data_shape  # [t, z, c, y, x] or [1, z, c, y, x]
            data = dat[kspace_key].transpose()  # .reshape(-1, data_shape[-3], data_shape[-2], data_shape[-1])
        else:
            data_shape = dat[kspace_key]["real"].shape
            data_shape = (1,) * (5 - len(data_shape)) + data_shape  # [t, z, c, y, x] or [1, z, c, y, x]
            data = np.array(dat[kspace_key]["real"] + 1j * dat[kspace_key]["imag"])
        data = data.reshape(data_shape)

        header[CMRxReconKeys.PID] = os.path.splitext(dat[CMRxReconKeys.FILENAME])[0].split("_")[0]
        header[CMRxReconKeys.NUM_FRAMES] = data_shape[0]
        header[CMRxReconKeys.NUM_SLICES] = data_shape[1]
        header[CMRxReconKeys.NUM_COILS] = data_shape[2]
        header[CMRxReconKeys.SHAPE] = np.array(data_shape)
        if CMRxReconKeys.MASK in dat.keys():
            mask = np.array(dat[CMRxReconKeys.MASK])
            if mask.ndim == 2:  # 2D sampling
                mask = np.expand_dims(mask, axis=(0, 1))
            elif mask.ndim == 3:  # 3D k-t sampling
                mask = np.expand_dims(mask, axis=(1, 2))
        else:
            mask = np.ones([1] * data.ndim)
        header[CMRxReconKeys.MASK] = mask.astype(np.float32)
        for smap_key in (CMRxReconKeys.SMAP, "smap", "dMap"):
            if smap_key in dat:
                if np.issubdtype(dat[smap_key].dtype, np.complexfloating):
                    smap_shape = dat[smap_key].shape[::-1]
                    smap_shape = (1,) * (5 - len(smap_shape)) + smap_shape
                    smap = dat[smap_key].transpose()
                else:
                    smap_shape = dat[smap_key]["real"].shape
                    smap_shape = (1,) * (5 - len(smap_shape)) + smap_shape
                    smap = np.array(dat[smap_key]["real"] + 1j * dat[smap_key]["imag"])
                header[CMRxReconKeys.SMAP] = smap.reshape(smap_shape)
                break
        return data, header

    def _get_meta_dict(self, dat: dict) -> dict:
        """
        Get all the metadata of the loaded dict and return the meta dict.

        Args:
            dat: a dictionary object loaded from a mat file.
        """
        return {k.value: dat[k.value] for k in CMRxReconKeys if k.value in dat and k != CMRxReconKeys.KSPACE}

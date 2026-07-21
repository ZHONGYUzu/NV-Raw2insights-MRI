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

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from models.latent_recon import create_mri_recon_model
from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex
from monai.apps.reconstruction.transforms.dictionary import ExtractDataKeyFromMetaKeyd
from monai.data import Dataset, partition_dataset
from monai.data.fft_utils import fftn_centered, ifftn_centered
from monai.transforms import Compose, EnsureTyped, Identityd, Lambdad, LoadImaged, ResizeWithPadOrCropd
from monai.utils import set_determinism
from mri_data.data_utils import crop_k_space, get_reader, postprocess_mri_recon, rearrange_mri_data
from torch.amp import autocast
from torch.distributed.elastic.multiprocessing.errors import record
from transforms import *
from utils import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

warnings.filterwarnings("ignore")


def parse_csv_values(value, cast=str):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).split(",")
    return [cast(item.strip()) for item in items if str(item).strip()]


def inference_input_suffix(dataset: str) -> str:
    return ".h5" if dataset.lower() == "fastmri" else ".json"


def reconstruction_filename(input_name: str) -> str:
    return f"{Path(input_name).stem}.mat"


@record
def infer(args):
    if args.ddp:
        # initialize the distributed training process, every GPU runs in a process
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Set device before initializing process group
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        # Initialize process group with timeout and device mapping
        timeout = timedelta(seconds=1800)  # 30 minutes timeout

        # Initialize process group with explicit device mapping
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )

        # Ensure all processes are synchronized after initialization
        dist.barrier(device_ids=[local_rank])
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank = 0
        world_size = 1

    if args.seed is not None:
        set_determinism(seed=args.seed)
    if args.debug is True:
        set_determinism(seed=0)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    Path(args.output_path).mkdir(parents=True, exist_ok=True)  # create output directory to store model checkpoints

    # Add barrier to ensure rank 1 finishes wandb init before rank 0 starts
    if args.ddp:
        dist.barrier()

    input_suffix = inference_input_suffix(args.dataset)
    test_files = sorted(file for file in Path(args.data_path_test).iterdir() if file.suffix.lower() == input_suffix)
    print(f"#Total test files before filtering: {len(test_files)}")
    # filter out already processed files
    test_files = [
        f
        for f in test_files
        if not (Path(args.output_path) / "val_img4ranking" / reconstruction_filename(f.name)).exists()
    ]
    print(f"#Total test files after filtering: {len(test_files)}")
    test_files = [dict([("kspace", test_files[i])]) for i in range(len(test_files))]
    print(f"#Test files: {len(test_files)}")
    test_files = partition_dataset(data=test_files, num_partitions=world_size, shuffle=False)[rank]

    # debug
    if args.debug:
        test_files = test_files[:1]

    # create the model
    model = create_mri_recon_model(args).to(device)
    try:
        args.is_multi_coil = model.use_csm or model.use_latent_csm
    except BaseException:
        args.is_multi_coil = True

    # Auto resume
    # Load the model, optimizer, and scheduler
    (
        model,
        optimizer_state_dict,
        scheduler_state_dict,
        scaler_state_dict,
        start_epoch,
        start_global_step,
        best_metric,
        best_metric_epoch,
        wandb_run_id,
    ) = load_net(
        model,
        args.model_ckpt,
        device,
        is_ddp=args.ddp,
        resume_rng_state=args.resume_rng_state,
    )
    model = torch.compile(model) if args.uniform_input_kspace else model
    print(f"#model_params: {np.sum([len(p.flatten()) for p in model.parameters()]) * 1.0e-6:.2f}M")

    fastmri_model_axis_adapter = (
        args.dataset.lower() == "fastmri" and getattr(args, "fastmri_model_axis_adapter", False)
    )

    test_transforms = Compose(
        [
            LoadImaged(
                keys=["kspace"],
                reader=get_reader(args, is_testing=True),
                image_only=False,
                dtype=np.complex64,
            ),
            # user can also add other random transforms but remember to disable randomness for val_transforms
            ExtractDataKeyFromMetaKeyd(keys=["mask", "acquisition"], meta_key="kspace_meta_dict"),
            ExtractOptionalDataKeyFromMetaKeyd(keys=["sensitivity_maps"], meta_key="kspace_meta_dict"),
            KspaceMaskd(
                keys=["kspace"],
                mask_types=(["fixed"] if not hasattr(args, "val_mask_types") else args.val_mask_types),
                center_fractions=([1.0] if not hasattr(args, "center_fractions") else args.center_fractions),
                accelerations=([0.0] if not hasattr(args, "accelerations") else args.accelerations),
                spatial_dims=2,
                is_complex=True,
                equispaced_offset=getattr(args, "fastmri_equispaced_offset", None),
            ),
            Lambdad(keys=["kspace"], func=lambda x: convert_to_tensor_complex(x)),
            Lambdad(
                keys=["sensitivity_maps"],
                func=lambda x: convert_to_tensor_complex(x),
                allow_missing_keys=True,
            ),
            (
                Lambdad(
                    keys=["kspace", "kspace_masked", "mask", "sensitivity_maps"],
                    func=lambda x: x.transpose(-3, -2),
                    allow_missing_keys=True,
                )
                if fastmri_model_axis_adapter
                else Identityd(keys=["kspace"])
            ),
            (
                ResizeWithPadOrCropd(
                    keys=["kspace", "mask", "kspace_masked", "sensitivity_maps"],
                    spatial_size=[
                        -1,
                        -1,
                        args.uniform_input_kspace[0],
                        args.uniform_input_kspace[1],
                        2,
                    ],
                    allow_missing_keys=True,
                )
                if args.uniform_input_kspace
                else Identityd(keys=["kspace"])
            ),
            EnsureTyped(keys=["kspace", "kspace_masked", "mask", "sensitivity_maps"], allow_missing_keys=True),
            Lambdad(
                keys=["kspace", "kspace_masked"],
                overwrite=["kspace_ifft", "kspace_masked_ifft"],
                func=lambda x: ifftn_centered(x, spatial_dims=2, is_complex=True),
            ),
            RearrangeAndNormalizeMRI(keys=["kspace_masked_ifft", "kspace_ifft", "mask"], args=args),
            PrepareSensitivityMapd(keys=["sensitivity_maps"], args=args),
        ]
    )

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = MultiEpochsDataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    args.model_structure = str(model).split("\n")
    save_args_to_file_json(args, os.path.join(args.output_path, "config.json"))

    # Test
    model.eval()
    with torch.no_grad():
        tic_val = time.time()
        previous_case_end = time.perf_counter()
        for test_data in tqdm.tqdm(test_loader):
            case_start = time.perf_counter()
            timings = {"data_load": case_start - previous_case_end}
            (
                input,
                mask,
                mask_type,
                acc_factor,
                acq_type,
                mean,
                std,
                temporal_shuffle,
                file_name,
                final_shape,
            ) = (
                test_data["kspace_masked_ifft"][0],
                test_data["mask"][0],
                test_data["mask_type"][0],
                test_data["acc_factor"][0],
                test_data["acquisition"][0],
                test_data["mean"][0],
                test_data["std"][0],
                test_data["temporal_shuffle"][0],
                test_data["kspace_meta_dict"]["filename"][0],
                test_data["kspace_meta_dict"]["shape"][0],
            )
            sensitivity_maps = test_data["sensitivity_maps"][0] if "sensitivity_maps" in test_data else None
            if args.debug:
                print(file_name, mask_type, acc_factor, acq_type)
                if fastmri_model_axis_adapter:
                    mask_bool = mask[..., 0].bool()
                    constant_along_model_frequency = torch.equal(
                        mask_bool,
                        mask_bool[..., :1].expand_as(mask_bool),
                    )
                    varies_along_model_phase = not torch.equal(
                        mask_bool,
                        mask_bool[..., :1, :].expand_as(mask_bool),
                    )
                    if not (constant_along_model_frequency and varies_along_model_phase):
                        raise RuntimeError(
                            "fastMRI model-axis adapter produced an unexpected mask orientation; "
                            "expected variation along model PE/H and constancy along model frequency/W."
                        )
                    print(
                        "Verified adapted mask orientation: varies along model PE/H, "
                        "constant along model frequency/W.",
                        flush=True,
                    )
            prepare_start = time.perf_counter()
            final_shape = [int(s) for s in final_shape]
            model_spatial_shape = (
                [final_shape[-1], final_shape[-2]] if fastmri_model_axis_adapter else final_shape[-2:]
            )
            if args.debug and fastmri_model_axis_adapter:
                print(
                    "Using fastMRI model-axis adapter: source W becomes model PE/H; output is transposed back.",
                    flush=True,
                )
            input = (
                fftn_centered(input, spatial_dims=2, is_complex=True)
                if args.model_type.lower() in ["varnet", "kspace_mar"]
                else input
            )

            # iterate through all samples:
            num_samples = input.shape[0]
            outputs = []
            slice_window_single_frame = (
                args.dataset.lower() == "fastmri"
                and final_shape[-5] == 1
                and getattr(args, "fastmri_adjacent_slice_window", False)
            )
            if args.debug and slice_window_single_frame:
                print(
                    f"Using adjacent-slice {args.num_frames}-view windows for single-frame fastMRI input.",
                    flush=True,
                )
            timings["prepare"] = time.perf_counter() - prepare_start
            timings["window_and_transfer"] = 0.0
            timings["model"] = 0.0
            timings["output"] = 0.0
            num_forwards = 0
            for micro_b, _ in mini_dataloader(
                list(range(num_samples)),
                args.batch_size,
                shuffle=False,
                drop_last=False,
                pad_last=False,
            ):
                # forward pass
                stage_start = time.perf_counter()
                inp, window_idx = windowed_input(
                    input,
                    micro_b,
                    final_shape,
                    num_frames=args.num_frames,
                    slice_window_single_frame=slice_window_single_frame,
                )
                mas = torch.Tensor(mask[window_idx])
                smap = None
                if sensitivity_maps is not None:
                    smap, _ = windowed_input(
                        sensitivity_maps,
                        micro_b,
                        final_shape,
                        num_frames=args.num_frames,
                        slice_window_single_frame=slice_window_single_frame,
                    )
                    smap = smap.to(device)
                inp, mas, mean, std = (
                    inp.to(device),
                    mas.to(device),
                    mean.to(device),
                    std.to(device),
                )
                if args.profile_timing and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                timings["window_and_transfer"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                with autocast("cuda", torch.bfloat16, enabled=args.amp):
                    output = model(inp, mas.bool(), mask_type, acc_factor, acq_type, sensitivity_maps=smap)
                if args.profile_timing and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                timings["model"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                output = output[:, args.num_frames // 2]
                if getattr(args, "hard_data_consistency", False):
                    reference = inp[:, args.num_frames // 2]
                    acquired = mas[:, args.num_frames // 2].bool()
                    output_kspace = fftn_centered(output.float(), spatial_dims=2, is_complex=True)
                    reference_kspace = fftn_centered(reference.float(), spatial_dims=2, is_complex=True)
                    output = ifftn_centered(
                        torch.where(acquired, reference_kspace, output_kspace),
                        spatial_dims=2,
                        is_complex=True,
                    )
                output = output * std[micro_b] + mean[micro_b]  # [1, c/1, 320, 320, 2]
                output = complex_abs(crop_k_space(output, model_spatial_shape))  # [b, c/1, h, w]

                outputs.append(output.data.cpu().numpy())
                timings["output"] += time.perf_counter() - stage_start
                num_forwards += 1
                if args.profile_timing and num_forwards % args.profile_interval == 0:
                    elapsed = time.perf_counter() - case_start
                    eta = elapsed / num_forwards * (num_samples - num_forwards)
                    print(
                        f"\n[timing] {Path(file_name).name}: {num_forwards}/{num_samples} forwards, "
                        f"elapsed={elapsed:.1f}s, estimated_remaining={eta:.1f}s",
                        flush=True,
                    )

            stage_start = time.perf_counter()
            outputs = rearrange_mri_data(
                [np.vstack(outputs)],
                args,
                is_complex=False,
                reverse=True,
                num_slices=final_shape[-4],
                num_coils=final_shape[-3],
                temporal_shuffle=temporal_shuffle,
            )  # (time), slice, coil, h, w
            timings["rearrange"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            outputs_rss = np.sqrt(np.sum(outputs[0] ** 2, axis=-3))  # RSS: (time), slice, h, w
            if fastmri_model_axis_adapter:
                outputs_rss = outputs_rss.swapaxes(-2, -1)
            timings["rss"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            outputs_pp = postprocess_mri_recon(
                outputs_rss,
                args,
                file_name,
                is_training=False,
                pp_z_score_norm=args.pp_z_score_norm,
            )  # w, h, slice, (time)
            timings["postprocess"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            save_img4ranking(
                outputs_pp,
                os.path.join(args.output_path, "val_img4ranking"),
                reconstruction_filename(file_name),
            )
            timings["save_mat"] = time.perf_counter() - stage_start

            case_end = time.perf_counter()
            timings["total"] = case_end - case_start
            if args.profile_timing:
                timing_text = ", ".join(f"{name}={seconds:.2f}s" for name, seconds in timings.items())
                per_forward = timings["model"] / max(num_forwards, 1)
                print(
                    f"\n[timing] {Path(file_name).name}: samples={num_samples}, forwards={num_forwards}, "
                    f"model_per_forward={per_forward:.3f}s, {timing_text}",
                    flush=True,
                )
            previous_case_end = case_end

        if args.ddp:
            # wait for all processes to finish
            dist.barrier()
    torch.cuda.empty_cache()

    print(f"inference completed! test elapsed time: {(time.time() - tic_val) / 60:.2f} mins")

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=Path,
        required=True,
        help="Path to the config file",
    )
    parser.add_argument(
        "-m",
        "--model_ckpt",
        default=None,  # Optional, will be auto-downloaded from Hugging Face if not provided
        type=Path,
        required=False,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default=Path("dataset/CustomCINEDataR1/json_input"),
        type=Path,
        required=False,
        help="Path to the input folder (default: dataset/CustomCINEDataR1/json_input)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=Path("output/CustomCINEOutputR1"),
        type=Path,
        required=False,
        help="Path to the output folder (default: output/CustomCINEOutputR1)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode",
    )
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        default=False,
        help="Print synchronized per-case inference timing broken down by processing stage",
    )
    parser.add_argument(
        "--profile-interval",
        type=int,
        default=25,
        help="Forward-pass interval for live timing updates (default: 25)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the config DataLoader worker count; use 0 to minimize host-memory usage.",
    )
    parser.add_argument(
        "--fixed-mask-types",
        default=None,
        help="Comma-separated fixed mask filters, for example mask_ktRadial4,mask_ktRadial8. Use 'none' to disable filtering.",
    )
    parser.add_argument(
        "--accelerations",
        default=None,
        help="Comma-separated acceleration classes for inference/model conditioning, for example 2,3,4,8,16,24.",
    )
    parser.add_argument(
        "--disable-acs-region",
        action="store_true",
        default=False,
        help="Disable ACS-region extraction for sensitivity-map estimation; useful when masks do not contain a filled ACS center.",
    )
    parser.add_argument(
        "--hard-data-consistency",
        action="store_true",
        default=False,
        help="Replace predicted k-space at acquired locations with the measured values before saving.",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    config.ddp = is_ddp_enabled()
    model_variant = config.model_variant
    config.model_ckpt = resolve_checkpoint_path(model_variant, args.model_ckpt)
    config.output_path = args.output_path
    config.data_path_test = args.input_path
    config.debug = args.debug
    config.profile_timing = args.profile_timing
    config.profile_interval = max(args.profile_interval, 1)
    config.hard_data_consistency = args.hard_data_consistency
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error("--num-workers must be non-negative")
        config.num_workers = args.num_workers
    if args.fixed_mask_types is not None:
        config.fixed_mask_types = None if args.fixed_mask_types.lower() == "none" else parse_csv_values(args.fixed_mask_types)
    if args.accelerations is not None:
        config.accelerations = parse_csv_values(args.accelerations, float)
        if len(getattr(config, "center_fractions", [])) != len(config.accelerations):
            config.center_fractions = [0.0] * len(config.accelerations)
    if args.disable_acs_region:
        config.use_acs_region = False
    if config.ddp and ("MASTER_PORT" not in os.environ.keys()):
        port = str(find_free_network_port())
        print(f"using port {port}")
        os.environ["MASTER_PORT"] = port  # str(port)
    infer(config)

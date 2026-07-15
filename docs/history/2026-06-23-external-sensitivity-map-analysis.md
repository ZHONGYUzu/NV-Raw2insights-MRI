# Historical Analysis: External Sensitivity Map Support

> Status: superseded by the implementation completed after this analysis. The
> current inference reader accepts `sensitivity_maps`, `smap`, or `dMap` from a
> case descriptor; the transform pipeline prepares the maps; and
> `scripts/inference.py` passes them to the model. See
> [`2026-07-14-h5-dmap-noacs-acc8-inference.md`](2026-07-14-h5-dmap-noacs-acc8-inference.md)
> for a completed 10-case server run. The remainder of this file is retained as
> a dated record of the pre-implementation code review.

Date: 2026-06-23

## Question

Confirm whether the current NV-Raw2Insights-MRI code can directly use existing sensitivity maps from the custom CINE H5 data.

Specific questions:

1. Does the current code support external `smap` input?
2. If not, is it fundamentally impossible, or is the interface simply missing?
3. Where are sensitivity maps generated, saved, and passed into the model?

## Executive Conclusion

The current CMRxRecon inference and training path does not directly support external sensitivity map input.

This is not a fundamental model limitation. The model code already has an internal `sensitivity_maps` argument and uses sensitivity maps for coil combination and expansion. However, the data reader, transform pipeline, and top-level train/inference calls do not load or pass external maps from JSON/MAT/H5 files.

In the current base configuration, sensitivity maps are generated inside the model by `CoilSensitivityModel_DCAE`, typically from the ACS region. They are temporary tensors used during forward passes and are not saved to disk.

Important extra caution: even though the cascade `forward` function accepts `sensitivity_maps`, the current default behavior can overwrite externally provided maps unless the surrounding CSM logic is adjusted. In particular, the condition `if sensitivity_maps is None or not self.use_single_csm:` means that with the default `use_single_csm = False`, a provided map would still be recomputed.

## Current Status

| Item | Status |
| --- | --- |
| External `dMap` / `smap` path in JSON | Not supported |
| CMRxRecon reader loads external smap | Not supported |
| Transform pipeline carries smap | Not supported |
| Train/inference passes smap to model | Not supported |
| Model has internal `sensitivity_maps` parameter | Supported |
| Model internally estimates sensitivity maps | Supported |
| Sensitivity maps saved to disk during inference | Not supported |

## Evidence 1: The CMRxRecon Reader Only Reads K-Space And Mask

The JSON reader reads only `kspace` and `mask`. There is no `smap`, `sensitivity_maps`, or `dMap` field in the CMRxRecon JSON interface.

```text
┌─ scripts/readers.py:302-321 ────────────────────────────────────────────────┐
│ 302         with open(data, "r") as f:                                      │
│ 303             json_data = json.load(f)                                    │
│ 304             kspace = json_data["kspace"]                                │
│ 305             masks = self.filter_masks_by_types(json_data["mask"],       │
│ 306                                               self.fixed_mask_types)    │
│ 307             mask = random.choice(masks) if json_data["mask"] else ""    │
│ 308             mask_type = mask.split("_mask_")[-1][:-4]                   │
│ 309             acquisition_type = re.search(                                │
│ 310                 r"(?:^|[/\\])MultiCoil[/\\]([^/\\]+)",                  │
│ 311                 kspace,                                                  │
│ 312                 flags=re.I,                                             │
│ 313             ).group(1)                                                   │
│ 314                                                                       │
│ 315         kspace_kv = self.read_mat(kspace)                               │
│ 316         mask_kv = self.read_mat(mask) if mask else [(None, None)]       │
│ 317                                                                       │
│ 318         dat = dict(                                                     │
│ 319             kspace_kv                                                   │
│ 320             + mask_kv                                                   │
│ 321             + [                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

The loaded data is then interpreted as k-space, and metadata is built from the k-space shape. The mask is placed into metadata, but no sensitivity map is extracted.

```text
┌─ scripts/readers.py:332-363 ────────────────────────────────────────────────┐
│ 332         header = self._get_meta_dict(dat)                               │
│ 333         if "kus" in dat:                                                │
│ 334             kspace_key = "kus"                                          │
│ 335         else:                                                           │
│ 336             kspace_key = CMRxReconKeys.KSPACE if CMRxReconKeys.KSPACE  │
│ 337                          in dat else "kspace"                          │
│ ...                                                                       │
│ 349         header[CMRxReconKeys.PID] = os.path.splitext(                  │
│ 350             dat[CMRxReconKeys.FILENAME]                                 │
│ 351         )[0].split("_")[0]                                               │
│ 352         header[CMRxReconKeys.NUM_FRAMES] = data_shape[0]                │
│ 353         header[CMRxReconKeys.NUM_SLICES] = data_shape[1]                │
│ 354         header[CMRxReconKeys.NUM_COILS] = data_shape[2]                 │
│ 355         header[CMRxReconKeys.SHAPE] = np.array(data_shape)              │
│ 356         if CMRxReconKeys.MASK in dat.keys():                            │
│ 357             mask = np.array(dat[CMRxReconKeys.MASK])                    │
│ 358             if mask.ndim == 2:                                          │
│ 359                 mask = np.expand_dims(mask, axis=(0, 1))               │
│ 360             elif mask.ndim == 3:                                        │
│ 361                 mask = np.expand_dims(mask, axis=(1, 2))               │
│ 362         header[CMRxReconKeys.MASK] = mask.astype(np.float32)            │
│ 363         return data, header                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Evidence 2: Inference Transform Pipeline Does Not Carry Smap

The inference pipeline loads only `kspace`, extracts only `mask` and `acquisition` from metadata, then builds masked k-space and normalized image tensors.

```text
┌─ scripts/inference.py:137-177 ──────────────────────────────────────────────┐
│ 137     test_transforms = Compose(                                          │
│ 138         [                                                              │
│ 139             LoadImaged(                                                │
│ 140                 keys=["kspace"],                                       │
│ 141                 reader=get_reader(args, is_testing=True),              │
│ 142                 image_only=False,                                      │
│ 143                 dtype=np.complex64,                                    │
│ 144             ),                                                         │
│ 145             ExtractDataKeyFromMetaKeyd(                                │
│ 146                 keys=["mask", "acquisition"],                          │
│ 147                 meta_key="kspace_meta_dict"                            │
│ 148             ),                                                         │
│ 149             KspaceMaskd(                                               │
│ 150                 keys=["kspace"],                                       │
│ ...                                                                       │
│ 170             EnsureTyped(keys=["kspace", "kspace_masked", "mask"]),     │
│ 171             Lambdad(                                                   │
│ 172                 keys=["kspace", "kspace_masked"],                     │
│ 173                 overwrite=["kspace_ifft", "kspace_masked_ifft"],      │
│ 174                 func=lambda x: ifftn_centered(                         │
│ 175                     x, spatial_dims=2, is_complex=True                 │
│ 176                 ),                                                     │
│ 177             ),                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

The final rearrangement and normalization transform also expects exactly three keys: input, target, and mask. There is no fourth key for sensitivity maps.

```text
┌─ scripts/transforms.py:1097-1124 ───────────────────────────────────────────┐
│ 1097 class RearrangeAndNormalizeMRI(MapTransform):                         │
│ 1098     def __init__(self, keys: KeysCollection, args,                    │
│ 1099                  allow_missing_keys: bool = False) -> None:           │
│ 1100         MapTransform.__init__(self, keys, allow_missing_keys)         │
│ 1101         self.keys_list = keys                                         │
│ 1102         self.args = args                                              │
│ 1103         assert len(keys) == 3, "Keys should have 3 keys..."           │
│ ...                                                                       │
│ 1112         input, target, mask = rearrange_mri_data(                     │
│ 1113             [d[k][None, ...] for k in self.keys_list],                │
│ 1114             self.args,                                                │
│ 1115             temporal_shuffle=temporal_shuffle                         │
│ 1116         )                                                            │
│ 1117         input, mean, std = complex_zscore(input, dim=[1, 2, 3])       │
│ 1118         d[self.keys_list[0]] = input.contiguous()                     │
│ 1119         d[self.keys_list[1]] = target.contiguous()                    │
│ 1120         d[self.keys_list[2]] = mask.contiguous()                      │
│ 1121         d["mean"] = mean.contiguous()                                 │
│ 1122         d["std"] = std.contiguous()                                   │
│ 1123         d["temporal_shuffle"] = temporal_shuffle                      │
│ 1124         return d                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Evidence 3: Train And Inference Do Not Pass Smap To The Model

During inference, the model is called with only five runtime arguments after `self`: input image, mask, mask type, acceleration factor, and acquisition type.

```text
┌─ scripts/inference.py:242-259 ──────────────────────────────────────────────┐
│ 242             # forward pass                                             │
│ 243             stage_start = time.perf_counter()                          │
│ 244             inp, window_idx = windowed_input(                          │
│ 245                 input, micro_b, final_shape, num_frames=args.num_frames│
│ 246             )                                                          │
│ 247             mas = torch.Tensor(mask[window_idx])                       │
│ ...                                                                       │
│ 256             stage_start = time.perf_counter()                          │
│ 257             with autocast("cuda", torch.bfloat16, enabled=args.amp):   │
│ 258                 output = model(inp, mas.bool(), mask_type,             │
│ 259                                acc_factor, acq_type)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

Training follows the same call pattern.

```text
┌─ scripts/train.py:336-349 ─────────────────────────────────────────────────┐
│ 336             # forward pass                                             │
│ 337             inp, window_idx = windowed_input(                          │
│ 338                 input, micro_b, final_shape, num_frames=args.num_frames│
│ 339             )                                                          │
│ 340             tar = torch.Tensor(target[window_idx])                     │
│ 341             mas = torch.Tensor(mask[window_idx])                       │
│ ...                                                                       │
│ 347             with autocast("cuda", torch.bfloat16, enabled=args.amp):   │
│ 348                 output = model(inp, mas.bool(), mask_type,             │
│ 349                                acc_factor, acq_type)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Evidence 4: The Model Does Have A Sensitivity Map Argument

The cascade-level model has an explicit `sensitivity_maps` argument.

```text
┌─ scripts/models/restormer/restormer.py:1238-1249 ───────────────────────────┐
│ 1238     def forward(                                                       │
│ 1239         self,                                                          │
│ 1240         x: torch.Tensor,                                               │
│ 1241         ref_image: torch.Tensor,                                       │
│ 1242         mask: torch.Tensor = None,                                     │
│ 1243         cas_skips: torch.Tensor = None,                                │
│ 1244         mask_type: str = None,                                         │
│ 1245         acc_factor: int = None,                                        │
│ 1246         acq_type: str = None,                                          │
│ 1247         sensitivity_maps: torch.Tensor = None,                         │
│ 1248         timestep: int = None,                                          │
│ 1249     ) -> tuple[Tensor | Any, Any]:                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

The cascaded wrapper passes `sensitivity_maps` from one cascade to the next.

```text
┌─ scripts/models/latent_recon.py:107-141 ───────────────────────────────────┐
│ 107         ref_image = x.clone() if ref_image is None else ref_image       │
│ 108         for i, recon in enumerate(self.recon_models):                  │
│ ...                                                                       │
│ 127                 x, cas_skips, sensitivity_maps = recon(                │
│ 128                     x,                                                  │
│ 129                     ref_image,                                          │
│ 130                     mask,                                               │
│ 131                     cas_skips,                                          │
│ 132                     mask_type,                                          │
│ 133                     acc_factor,                                         │
│ 134                     acq_type,                                           │
│ 135                     sensitivity_maps,                                   │
│ 136                     actual_timestep,                                    │
│ 137                 )                                                       │
│ ...                                                                       │
│ 141             return x                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Evidence 5: Current Smap Generation Is Internal

The base configuration enables internal coil sensitivity map usage.

```text
┌─ configs/nv_raw2insights_mri_base.json:89-91 ───────────────────────────────┐
│ 89     "use_csm": true,                                                     │
│ 90     "use_tau_csm": false,                                                │
│ 91     "use_acs_region": true,                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

When `use_csm` is enabled, the Restormer MRI module creates a `CoilSensitivityModel_DCAE`.

```text
┌─ scripts/models/restormer/restormer.py:1064-1073 ───────────────────────────┐
│ 1064         self.pad_factor = 2**2                                         │
│ 1065         if self.use_csm:                                               │
│ 1066             if (self.first_cascade and self.use_single_csm) or         │
│ 1067                not self.use_single_csm:                                │
│ 1068                 self.coil_sensitivity_model = CoilSensitivityModel_DCAE│
│ 1069                     spatial_dims=2,                                    │
│ 1070                     features=(12, 24, 48, 96, 192),                   │
│ 1071                     pad_factor=self.pad_factor,                        │
│ 1072                 )                                                      │
│ 1073             if self.args.pretrained_csm is not None:                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

During forward, if no usable sensitivity map is present, the model estimates it internally. With `use_acs_region = true`, it estimates maps from ACS image data.

```text
┌─ scripts/models/restormer/restormer.py:1260-1273 ───────────────────────────┐
│ 1260         if self.use_csm:                                               │
│ 1261             if sensitivity_maps is None or not self.use_single_csm:    │
│ 1262                 if self.use_acs_region and mask is not None:           │
│ 1263                     x_acs = get_acs_image(x, mask)                    │
│ 1264                     if self.use_single_csm:                            │
│ 1265                         sensitivity_maps = self.coil_sensitivity_model│
│ 1266                             (x_acs)                                    │
│ 1267                     else:                                              │
│ 1268                         sensitivity_maps = checkpoint(                 │
│ 1269                             self.coil_sensitivity_model, x_acs,        │
│ 1270                             use_reentrant=False                       │
│ 1271                         )                                             │
│ 1272                 else:                                                  │
│ 1273                     ...                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

The generated maps are then used to reduce multi-coil data to the model's reduced coil representation and expand it back after reconstruction.

```text
┌─ scripts/models/restormer/restormer.py:1273-1311 ───────────────────────────┐
│ 1273             x = sensitivity_map_reduce(                               │
│ 1274                 x, sensitivity_maps, k=self.num_reduced_coils          │
│ 1275             )                                                          │
│ ...                                                                       │
│ 1310         if self.use_csm:                                               │
│ 1311             x = sensitivity_map_expand(x, sensitivity_maps)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

The actual sensitivity map math is implemented in utility helpers.

```text
┌─ scripts/utils.py:371-383 ─────────────────────────────────────────────────┐
│ 371 def sensitivity_map_reduce(img: torch.Tensor,                         │
│ 372                            sens_maps: torch.Tensor,                   │
│ 373                            k: int = 1, mode: str = "rand") -> Tensor: │
│ ...                                                                       │
│ 379         return complex_mul_t(img, complex_conj_t(sens_maps)).sum(      │
│ 380             dim=-4, keepdim=True                                      │
│ 381         )                                                            │
│ 382 def sensitivity_map_expand(img: torch.Tensor,                         │
│ 383                            sens_maps: torch.Tensor) -> torch.Tensor:  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Direct Answer To The Task Questions

### 1. Does the current code support external input smap?

No. The current CMRxRecon path does not provide a usable external smap interface.

The model accepts a `sensitivity_maps` tensor internally, but the application pipeline never loads such a tensor from the custom CINE data and never passes it into `model(...)`.

### 2. If not, is it impossible or just missing an interface?

It is not impossible. It is mostly an interface and data-shaping gap.

To support existing CINE `dMap`, the code needs:

1. JSON schema support, for example:

```json
{
  "kspace": "dataset/CustomCINEDataR1/MultiCoil/Cine/UnderSample_TaskR1/Sub0001_kspace_full.mat",
  "mask": [
    "dataset/CustomCINEDataR1/MultiCoil/Cine/Mask_TaskR1/Sub0001_mask_ktRadial8.mat"
  ],
  "smap": "dataset/CustomCINEDataR1/MultiCoil/Cine/SensitivityMap/Sub0001_smap.mat"
}
```

2. Reader support in `CMRxReconReader.read()` and `CMRxReconReader.get_data()`.
3. Transform support to carry, crop/pad, normalize if needed, and rearrange smap consistently with k-space windows.
4. Train/inference support to slice the correct smap window and call the model with it.
5. Model logic adjustment so externally provided maps are not overwritten when `use_single_csm = False`.

### 3. Where are smaps generated, saved, and passed into the model?

Generated:

- Internally in `restormer_mri.forward()`.
- The source image is either ACS-region image data from `get_acs_image(x, mask)` or the current image tensor `x`, depending on `use_acs_region`.
- The estimator is `CoilSensitivityModel_DCAE`.

Saved:

- They are not saved to disk by the current inference or training path.
- They exist only as temporary tensors during forward.

Passed:

- They are passed internally between cascades through the `sensitivity_maps` variable.
- They are not passed from the top-level data pipeline.

## CINE `dMap` Compatibility Notes

The documented custom CINE H5 `dMap` shape is:

```text
(nSlice, nCha, nPha, nPE, nFE)
example: (12, 15, 1, 132, 176)
```

This is time-averaged sensitivity map data, with `nPha = 1`. The current inference input k-space logical order is:

```text
(time, slice, coil, phase, frequency)
example: (25, 12, 15, 132, 176)
```

Therefore, using existing `dMap` would require at least:

```python
# H5 dMap: (slice, coil, 1, PE, FE)
smap = np.transpose(dmap_h5, (2, 0, 1, 3, 4))  # (1, slice, coil, PE, FE)
smap = np.repeat(smap, repeats=n_time, axis=0) # (time, slice, coil, PE, FE)
```

Then the tensor must be converted into the same complex representation and per-window layout used by the model, likely matching the model's `(B, T, C, H, W, 2)` expectation before cascade flattening.

## Recommended Implementation Direction

The cleanest implementation path is to add an optional external sensitivity map path without disturbing the existing default behavior.

Suggested flags:

```text
--use-external-smap
--external-smap-key sensitivity_maps
```

Suggested behavior:

- If `--use-external-smap` is false, keep the current internal ACS-estimated CSM behavior.
- If true, require each JSON descriptor to contain `smap`.
- Load the map from MAT/H5 using a known key such as `sensitivity_maps`, `smap`, or `dMap`.
- Validate shape against k-space: same slice, coil, PE, FE; either one time frame or matching number of time frames.
- Broadcast time-averaged maps from `T=1` to `T=nPha` when needed.
- Pass the external tensor into `model(...)`.
- Update the Restormer CSM condition so provided maps are respected.

The key model-side change should avoid recomputing sensitivity maps when an external tensor is supplied. Conceptually:

```python
if self.use_csm:
    if sensitivity_maps is None:
        sensitivity_maps = self.estimate_sensitivity_maps(x, mask)
    x = sensitivity_map_reduce(x, sensitivity_maps, k=self.num_reduced_coils)
```

This preserves the current behavior when no external smap is supplied while allowing existing CINE maps to be used deliberately.

## Final Assessment

External CINE sensitivity maps cannot be used directly today.

The path is feasible with moderate code changes. The model is already architecturally close to supporting this, but the repo currently treats sensitivity maps as internally estimated model state, not as input data.

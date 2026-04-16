# Multi-Organ Segmentation of Abdominal CT Scans
### Deep Learning for Medical Image Analysis — Group 9

## 1. Introduction

Manual organ delineation by radiologists is time-consuming, labour-intensive, prone to inter-observer variability, and impractical at clinical scale. Automated volumetric segmentation via deep learning has substantially closed the gap between machine and expert-level performance.

This project implements a full end-to-end pipeline for voxel-wise segmentation of **9 abdominal organs** (Aorta, Gallbladder, Spleen, Left Kidney, Right Kidney, Liver, Stomach, Pancreas, and IVC) from CT volumes. The pipeline spans dataset acquisition, preprocessing, 3D U-Net training, quantitative evaluation (DSC, IoU), and qualitative visualisation.

The primary model is a **3D U-Net** (MONAI framework) trained on 100 CT volumes drawn from the AbdomenAtlas1.0Mini dataset. Training was conducted on a Kaggle T4 GPU (16 GB VRAM) over 100 epochs using a DiceCELoss objective, AdamW optimiser, and cosine-annealing learning-rate schedule.

## 2. Data

### Dataset: AbdomenAtlas1.0Mini

| Property | Detail |
|---|---|
| Source | Hugging Face — `AbdomenAtlas/AbdomenAtlas1.0Mini` |
| Full dataset size | 5,195 annotated abdominal CT volumes |
| Subset used | **100 complete CT + mask pairs** |
| Working set size | ~8 GB (post-preprocessing) |
| Annotation type | Voxel-wise multi-class masks (`combined_labels.nii.gz`) |
| Number of classes | 10 (9 organs + background) |

### Label Mapping

| Index | Organ | Index | Organ |
|---|---|---|---|
| 0 | Background | 5 | Right Kidney |
| 1 | Aorta | 6 | Liver |
| 2 | Gallbladder | 7 | Stomach |
| 3 | Spleen | 8 | Pancreas |
| 4 | Left Kidney | 9 | IVC |

### Dataset Split

A stratified 70 / 15 / 15 split (random seed = 42) was applied to the 100 downloaded volumes:

- **Train:** 70 volumes
- **Validation:** 15 volumes
- **Test:** 15 volumes

### Preprocessing

All splits share a base preprocessing pipeline:
- **Isotropic resampling** to 1.5 × 1.5 × 1.5 mm (trilinear for CT, nearest-neighbour for masks)
- **HU clipping** to [−150, 250], linearly rescaled to [0.0, 1.0]
- **Foreground cropping** to remove zero-padded air regions

Training volumes additionally receive:
- Random patch sampling — 4 patches of 96³ voxels per volume (pos:neg = 1:1)
- Random flipping (p = 0.5 per axis)
- Random 90° axial rotation (p = 0.1)
- Random intensity shift ±0.10 (p = 0.5)



## 3. Questions & Answers



### Q1 — How was the dataset downloaded and prepared from HuggingFace?

**Answer:**  
The repository file listing was fetched via `list_repo_files`. Files were grouped by patient folder and filtered to keep only patients with both `ct.nii.gz` and `combined_labels.nii.gz`. The first 100 complete pairs were downloaded with a 6 GB cap, skipping files already on disk.

```python
from huggingface_hub import list_repo_files, hf_hub_download
from collections import defaultdict
from pathlib import Path

REPO_ID   = "AbdomenAtlas/AbdomenAtlas1.0Mini"
REPO_TYPE = "dataset"
N_PATIENTS = 100
GB_CAP     = 6.0

all_repo_files = list(list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE))

patient_files = defaultdict(dict)
for filepath in all_repo_files:
    parts = Path(filepath).parts
    if len(parts) < 2:
        continue
    patient_id = parts[0]
    filename   = parts[-1]
    if filename == "ct.nii.gz":
        patient_files[patient_id]["ct"] = filepath
    elif filename == "combined_labels.nii.gz":
        patient_files[patient_id]["mask"] = filepath

complete = {
    pid: v for pid, v in patient_files.items()
    if "ct" in v and "mask" in v
}

targets     = sorted(complete.keys())[:N_PATIENTS]
total_bytes = 0

for patient_id in targets:
    if total_bytes / 1e9 >= GB_CAP:
        break
    hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        filename=complete[patient_id]["ct"],
        local_dir=str(DATA_DIR), local_dir_use_symlinks=False,
    )
    hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        filename=complete[patient_id]["mask"],
        local_dir=str(DATA_DIR), local_dir_use_symlinks=False,
    )
```



### Q2 — How was the 70/15/15 train-validation-test split constructed?

**Answer:**  
All complete CT + mask pairs on disk were enumerated into a data list. `train_test_split` from `scikit-learn` was applied twice (seed = 42) — first to carve off 30% as a temporary holdout, then to split that holdout equally into validation and test sets. The split indices were saved as `data_splits.json` for reproducibility.

```python
from sklearn.model_selection import train_test_split
import numpy as np, json

indices = np.arange(len(data_list))
train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42)
val_idx,   test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

train_files = [data_list[i] for i in train_idx]
val_files   = [data_list[i] for i in val_idx]
test_files  = [data_list[i] for i in test_idx]

with open(RESULTS_DIR / "data_splits.json", "w") as fh:
    json.dump({
        "train": [d["image"] for d in train_files],
        "val"  : [d["image"] for d in val_files],
        "test" : [d["image"] for d in test_files],
    }, fh, indent=2)
```



### Q3 — How were the MONAI transform pipelines defined for training and validation?

**Answer:**  
A shared base pipeline (load → ensure channel first → isotropic resampling → HU normalisation → foreground crop) was defined, then extended for training with random patch sampling and spatial/intensity augmentations. Validation used only the deterministic base pipeline.

```python
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandShiftIntensityd, EnsureTyped,
)

_base_transforms = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5),
             mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"],
                         a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
]

train_transforms = Compose(
    _base_transforms + [
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=(96,96,96), pos=1, neg=1,
                               num_samples=4, image_key="image", image_threshold=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        EnsureTyped(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    _base_transforms + [EnsureTyped(keys=["image", "label"])]
)
```



### Q4 — How was the 3D U-Net model configured, and what loss/optimiser were used?

**Answer:**  
MONAI's `UNet` was configured for 3D volumetric segmentation with an encoder depth of 32→64→128→256→512 channels, stride-2 downsampling, 2 residual units per stage, batch normalisation, and 0.1 dropout (~31.2 M parameters). Training used `DiceCELoss` (background excluded), `AdamW`, cosine-annealing LR schedule, and automatic mixed precision (AMP).

```python
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=10,           # 9 organs + background
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="BATCH",
    dropout=0.1,
).to(DEVICE)

loss_fn   = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)
scaler = torch.cuda.amp.GradScaler()   # AMP — ~50% VRAM reduction
```



### Q5 — How was the model trained and the best checkpoint selected?

**Answer:**  
The training loop ran for 100 epochs. Each epoch computed the DiceCELoss with AMP. Validation occurred every 2 epochs using sliding-window inference (96³ window, 50% overlap). The best mean DSC on validation triggered a checkpoint save. Training loss dropped from ~3.2 to ~1.0 in the first 20 epochs; best checkpoint was saved at **Epoch 98** (mean DSC ≈ 0.509 on test set).

```python
best_mean_dice = -1.0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    for batch in train_loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = model(images)
            loss  = loss_fn(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()

    scheduler.step()

    if epoch % 2 == 0:         # validate every 2 epochs
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                val_out = sliding_window_inference(
                    val_batch["image"].to(DEVICE),
                    roi_size=(96,96,96), sw_batch_size=2,
                    predictor=model, overlap=0.5,
                )
                # ... post-process and compute dice_metric
        mean_dsc = dice_metric.aggregate().mean().item()
        if mean_dsc > best_mean_dice:
            best_mean_dice = mean_dsc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
```



### Q6 — How was the test-set evaluation performed, and what were the quantitative results?

**Answer:**  
The best checkpoint was loaded and evaluated on the 15-volume test set. DSC and IoU were computed per class directly from integer argmax masks on CPU (avoiding GPU one-hot tensors and eliminating OOM risk). Background was excluded from all metrics.

```python
def dice_and_iou_from_int_masks(pred_int, lbl_int, num_classes):
    dsc_vals, iou_vals = [], []
    for cls in range(1, num_classes):
        p = (pred_int == cls)
        g = (lbl_int  == cls)
        inter = (p & g).sum().float()
        dsc   = (2 * inter + 1e-6) / (p.sum().float() + g.sum().float() + 1e-6)
        union = (p | g).sum().float()
        iou   = (inter + 1e-6) / (union + 1e-6)
        dsc_vals.append(dsc); iou_vals.append(iou)
    return torch.stack(dsc_vals), torch.stack(iou_vals)
```

**Test-set results (DSC / IoU per organ):**

| Organ | DSC Mean | IoU Mean |
|---|---|---|
| Right Kidney | 0.883 | 0.797 |
| Pancreas | 0.721 | 0.611 |
| Left Kidney | 0.712 | 0.600 |
| Spleen | 0.676 | 0.552 |
| Gallbladder | 0.439 | 0.314 |
| Aorta | 0.402 | 0.286 |
| Liver | 0.365 | 0.229 |
| IVC | 0.353 | 0.241 |
| Stomach | 0.029 | 0.015 |
| **Overall Mean** | **0.509** | .405 |


### Q7 — How were the results visualised?

**Answer:**  
A three-panel figure was generated showing (1) training loss per epoch, (2) validation mean DSC every 2 epochs with a dashed best-epoch marker, and (3) per-organ DSC bar chart with ±std error bars. Additionally, the top-3 foreground-dense axial slices of a test patient were displayed as CT | Ground Truth | Prediction triplets with colour-coded organ overlays.

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1 — Training loss
axes[0].plot(train_loss_history, color="#2563eb", linewidth=2)

# Panel 2 — Validation DSC
ep_v, dsc_v = zip(*val_dice_history)
axes[1].plot(ep_v, dsc_v, color="#16a34a", marker="o")
axes[1].axvline(ep_v[np.argmax(dsc_v)], color="gray", linestyle="--")

# Panel 3 — Per-organ bar chart
axes[2].bar(range(len(organ_names)), df["DSC Mean"],
            yerr=df["DSC Std"], capsize=3)
axes[2].axhline(overall_dsc, color="red", linestyle="--")

plt.savefig(RESULTS_DIR / "training_results.png", dpi=150, bbox_inches="tight")
```


## 4. References

Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning dense volumetric segmentation from sparse annotation. In *Proceedings of MICCAI 2016*, 424–432. https://doi.org/10.1007/978-3-319-46723-8_49

Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The PASCAL Visual Object Classes (VOC) Challenge. *International Journal of Computer Vision*, 88(2), 303–338. https://doi.org/10.1007/s11263-009-0275-4

Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H. R., & Xu, D. (2022). UNETR: Transformers for 3D medical image segmentation. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision* (pp. 574–584).

Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., van der Laak, J., van Ginneken, B., & Sánchez, C. I. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60–88. https://doi.org/10.1016/j.media.2017.07.005

Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. In *Proceedings of ICLR 2017*.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *Proceedings of ICLR 2019*.

Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In *Proceedings of 3DV 2016*, 565–571. https://doi.org/10.1109/3DV.2016.79

NVIDIA. (2020). NVIDIA A100 tensor core GPU architecture. NVIDIA Corporation. https://www.nvidia.com/en-us/data-center/a100/

Qu, C., Zhang, T., Qiao, H., Tang, Y., Wang, A., Yuille, A., & Zhou, Z. (2023). AbdomenAtlas-8k: Annotating 8,000 CT volumes for multi-organ segmentation in three weeks. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Proceedings of MICCAI 2015*, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

Taha, A. A., & Hanbury, A. (2015). Metrics for evaluating 3D medical image segmentation: Analysis, selection, and tool. *BMC Medical Imaging*, 15(1), 29. https://doi.org/10.1186/s12880-015-0068-x

Taghanaki, S. A., Abhishek, K., Cohen, J. P., Cohen-Adad, J., & Hamarneh, G. (2021). Deep semantic segmentation of natural and medical images: A review. *Artificial Intelligence Review*, 54(1), 137–195. https://doi.org/10.1007/s10462-020-09854-1
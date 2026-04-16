# Multiorgan Segmentation of Abdominal CT scans Using

# 3D U-Net and the AbdomenAtlas1.0Mini Dataset

## Overview

This repository presents a complete pipeline for multi-organ abdominal CT segmentation using a 3D U-Net architecture implemented in MONAI. The system is trained on a subset of the AbdomenAtlas1.0Mini dataset and is designed to operate within constrained computational environments such as Kaggle and Google Colab.

The workflow encompasses dataset acquisition, preprocessing, patch-based training, inference via sliding-window strategy, and quantitative evaluation using standard segmentation metrics.

The implementation is provided in `main.ipynb`, with outputs and evaluation artifacts stored in the `results` directory. The full pipeline can be referenced in .

---

## Execution Environment

The pipeline is designed for GPU-enabled environments with limited memory resources. It is recommended to execute the notebook in one of the following:

* Google Colab (GPU runtime)
* Kaggle Notebooks (NVIDIA T4 GPU)

Execution on local systems without sufficient GPU memory may lead to failure during training or inference due to the high memory demands of volumetric data.

---

## Dataset and Sampling Configuration

The dataset is retrieved programmatically from the Hugging Face repository `AbdomenAtlas/AbdomenAtlas1.0Mini`. Due to storage and computational constraints, only a subset of patient volumes is used.

### Key Parameters

```python
N_PATIENTS = 100
GB_CAP = 6.0
```

### Implications

* Limits the diversity of anatomical variation
* May introduce class imbalance or absence of certain organs in the sampled subset
* Directly impacts generalization and per-organ performance

### Calibration Recommendations

* Increase `N_PATIENTS` if storage permits
* Replace lexicographic selection with randomized sampling to reduce bias
* Perform class distribution analysis prior to training

---

## Input Representation and Patch-Based Training

Training is conducted using fixed-size 3D patches extracted from volumetric CT scans.

### Key Parameters

```python
PATCH_SIZE = (96, 96, 96)
NUM_SAMPLES = 4
BATCH_SIZE = 4
```

### Implications

* Patch size determines the spatial context available to the model
* Smaller patches may fail to capture full organ structures
* Limited sampling reduces exposure to rare classes

### Calibration Recommendations

* Increase `PATCH_SIZE` to incorporate more anatomical context, subject to memory limits
* Increase `NUM_SAMPLES` to improve representation of small or sparse organs
* Adjust `BATCH_SIZE` in accordance with available GPU memory

---

## Preprocessing Pipeline

The preprocessing stage standardizes spatial resolution and intensity distribution across samples.

### Key Components

* Resampling to uniform voxel spacing:

  ```python
  pixdim = (1.5, 1.5, 1.5)
  ```
* Intensity normalization:

  ```python
  a_min = -150
  a_max = 250
  ```
* Foreground cropping to remove non-informative regions

### Implications

* Ensures consistency across scans from different sources
* Reduces computational overhead
* Focuses learning on anatomically relevant regions

### Calibration Recommendations

* Modify intensity window (`a_min`, `a_max`) to emphasize different tissue types
* Adjust voxel spacing for higher resolution at increased computational cost
* Evaluate the impact of disabling or modifying cropping for large organs

---

## Data Augmentation

Augmentation is applied during training to improve generalization.

### Techniques Used

* Random flipping along spatial axes
* Random rotation (90-degree increments)
* Intensity shifting

### Implications

* Enhances robustness to orientation and scanner variability
* Reduces overfitting on small datasets

### Calibration Recommendations

* Increase augmentation probabilities for improved generalization
* Introduce additional transformations such as elastic deformation for advanced experimentation

---

## Model Configuration

The segmentation backbone is a 3D U-Net with residual units and batch normalization.

### Key Parameters

```python
channels = (32, 64, 128, 256, 512)
strides = (2, 2, 2, 2)
num_res_units = 2
dropout = 0.1
```

### Implications

* Determines model capacity and representational power
* Larger configurations improve accuracy but increase memory usage and overfitting risk

### Calibration Recommendations

* Reduce channel depth for faster experimentation
* Increase capacity when training on larger datasets
* Adjust dropout to control overfitting

---

## Loss Function

The model is optimized using a compound loss function:

```python
DiceCELoss(include_background=False)
```

### Implications

* Combines overlap-based and classification-based supervision
* Excludes background to prevent dominance of non-informative voxels

### Calibration Recommendations

* Introduce class weighting to address imbalance
* Experiment with alternative losses such as focal or generalized Dice loss for improved performance on small organs

---

## Optimizer and Learning Rate Schedule

### Configuration

```python
LR = 1e-4
WEIGHT_DECAY = 1e-5
Scheduler = CosineAnnealingLR
```

### Implications

* Controls convergence speed and stability
* Improper tuning may result in underfitting or divergence

### Calibration Recommendations

* Increase learning rate for faster convergence in early experiments
* Decrease learning rate for fine-tuning
* Replace scheduler with adaptive alternatives if validation performance stagnates

---

## Inference Strategy

Inference is performed using sliding-window evaluation over full volumes.

### Configuration

```python
roi_size = PATCH_SIZE
overlap = 0.5
```

### Implications

* Enables full-volume prediction under memory constraints
* Overlap reduces boundary artifacts through averaging

### Calibration Recommendations

* Increase overlap for smoother predictions
* Adjust window size based on patch configuration

---

## Evaluation Metrics

Performance is assessed using Dice Similarity Coefficient (DSC), Intersection over Union (IoU), precision, and recall, computed per organ and averaged.

### Implications

* DSC and IoU evaluate region overlap
* Precision and recall diagnose over- and under-segmentation
* Background exclusion ensures meaningful evaluation

---

## Observed Limitations

Empirical results indicate the following common issues:

* Reduced performance on small or low-contrast organs (e.g., pancreas, stomach)
* Presence of false positives in regions with similar intensity profiles
* Variability in performance across organ classes due to imbalance
* Sensitivity to patch size and sampling strategy

These limitations arise primarily from dataset constraints, class imbalance, and limited spatial context.

---

## Recommendations for Improvement

* Increase dataset size and diversity
* Employ class-aware sampling strategies
* Introduce weighted or adaptive loss functions
* Increase patch size to incorporate global anatomical context
* Apply post-processing techniques to remove spurious predictions

---

## Outputs

The pipeline generates the following artifacts within the `results` directory:

* Quantitative evaluation metrics per organ
* Training and validation curves
* Visual comparisons of CT slices, ground truth, and predictions
* Summary statistics in structured format

---

## Conclusion

The presented pipeline demonstrates a modular and reproducible framework for 3D medical image segmentation under constrained computational settings. Performance is influenced significantly by data distribution, preprocessing choices, and training configuration. Careful calibration of these components is essential for achieving robust and clinically meaningful segmentation results.

---

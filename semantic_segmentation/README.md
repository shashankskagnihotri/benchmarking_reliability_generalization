# Semantic Segmentation Benchmark

This repository provides tools and models for benchmarking the robustness of semantic segmentation.

---

## Installation

This package requires **Python 3.10.x** (tested with 3.10.17). Please ensure you have a compatible Python version installed.

### Step 1: Run the installation script

This script installs PyTorch with CUDA support, OpenMMLab dependencies (via `mim`), and handles editable installs for submodules like `mmsegmentation` and its ops.

```bash
bash install.sh
```

### Step 2: Install this package in editable mode

After running the install script, install the core `semsegbench` package:

```bash
pip install -e .
```

---

## Preparation

### Datasets

#### ADE20K

Download the dataset from: https://ade20k.csail.mit.edu/

<details>
<summary>Expected directory structure</summary>

```
data/ade/
└── ADEChallengeData2016/
    ├── annotations/
    │   ├── training/
    │   │   ├── ADE_train_00000001.png
    │   │   ├── ADE_train_00000002.png
    │   │   └── ...
    │   └── validation/
    │       ├── ADE_val_00000001.png
    │       ├── ADE_val_00000002.png
    │       └── ...
    └── images/
        ├── training/
        │   ├── ADE_train_00000001.png
        │   ├── ADE_train_00000002.png
        │   └── ...
        └── validation/
            ├── ADE_val_00000001.png
            ├── ADE_val_00000002.png
            └── ...
```

</details>

#### Cityscapes

Download the dataset from: https://www.cityscapes-dataset.com/


<details>
<summary>Expected directory structure</summary>

```
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   │   ├── aachen_000001_000019_leftImg8bit.png
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       ├── frankfurt/
│       │   ├── frankfurt_000000_000294_leftImg8bit.png
│       │   ├── frankfurt_000000_000576_leftImg8bit.png
│       │   └── ...
│       └── ...
└── gtFine/
    ├── train/
    │   ├── aachen/
    │   │   ├── aachen_000000_000019_gtFine_labelTrainIds.png
    │   │   ├── aachen_000001_000019_gtFine_labelTrainIds.png
    │   │   └── ...
    │   └── ...
    └── val/
        ├── frankfurt/
        │   ├── frankfurt_000000_000294_gtFine_labelTrainIds.png
        │   ├── frankfurt_000000_000576_gtFine_labelTrainIds.png
        │   └── ...
        └── ...
```

</details>

#### VOC

Download the dataset from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

This repository provides example split `.txt` files in `examples/voc2012/` for use with the dataset.  
Each line in these files contains a single image ID (without extension or path), for example:

```
2007_000033
2007_000042
2007_000061
```

To use these splits with MMSegmentation or similar frameworks, place the `.txt` files under:

```
data/VOCdevkit/VOC2012/ImageSets/Segmentation/
```

<details>
<summary>Expected directory structure</summary>

```
data/VOCdevkit/
└── VOC2012/
    ├── ImageSets/
    │   └── Segmentation/
    │       ├── aug.txt
    │       ├── train.txt
    │       └── val.txt
    ├── JPEGImages/
    │   ├── 2007_000027.jpg
    │   ├── 2007_000032.jpg
    │   └── ...
    ├── SegmentationClass/
    │   ├── 2007_000032.png
    │   ├── 2007_000033.png
    │   └── ...
    └── SegmentationClassAug/
        ├── 2007_000032.png
        ├── 2007_000033.png
        └── ...
```

</details>

#### ACDC

Download the dataset from: https://acdc.vision.ee.ethz.ch/

<details>
<summary>Expected directory structure (per weather condition)</summary>

```
data/
├── acdc_fog/
│   ├── gt/
│   │   ├── train/
│   │   │   └── GOPR0475/
│   │   │       ├── GOPR0475_frame_000041_gt_labelTrainIds.png
│   │   │       ├── GOPR0475_frame_000049_gt_labelTrainIds.png
│   │   │       └── ...
│   │   └── test/
│   │       └── GOPR0475/
│   │           ├── GOPR0475_frame_000247_gt_labelTrainIds.png
│   │           ├── GOPR0475_frame_000252_gt_labelTrainIds.png
│   │           └── ...
│   └── rgb_anno/
│       ├── train/
│       │   └── GOPR0475/
│       │       ├── GOPR0475_frame_000041_rgb_anon.png
│       │       ├── GOPR0475_frame_000049_rgb_anon.png
│       │       └── ...
│       └── test/
│           └── GOPR0475/
│               ├── GOPR0475_frame_000247_rgb_anon.png
│               ├── GOPR0475_frame_000252_rgb_anon.png
│               └── ...
├── acdc_night/
│   └── ...
├── acdc_rain/
│   └── ...
└── acdc_snow/
    └── ...
```

</details>

---

## How to Use

### Model Zoo

You can load a pre-trained model by specifying the model name, backbone, and the dataset it was trained on:

```python
from semsegbench.evals import load_model

model = load_model(
    model_name='DeepLabV3', 
    backbone='ResNet50', 
    dataset='ADE20K',
)
```

#### Supported Models

The checkpoint list, with crop size, is [`SUPPORTED_MODELS.md`](./SUPPORTED_MODELS.md). The same architecture and backbone on one dataset can be two checkpoints when the crop differs. Those rows stay. `model_id` includes the crop.

`load_model` opens the config recorded for that checkpoint in [`../results/model_provenance.csv`](../results/model_provenance.csv). Weights are not in git. A local file, if you have one, goes in `checkpoints/<architecture>/`. ResNet filenames are matched as `r50`, `r101`, or `r18`. Other backbones are matched as written (`mit-b0`, `swin-b`), and the name must also contain the dataset. `evaluate(..., retrieve_existing=True)` returns a cached row only when architecture, backbone, dataset, and crop identify one checkpoint. Training settings are in [`../docs/model_provenance.md`](../docs/model_provenance.md). Scores are in [`../docs/results.md`](../docs/results.md).

### Evaluation

#### Adversarial Attacks

```python
from semsegbench.evals import evaluate

model, results = evaluate(
    model_name='DeepLabV3', 
    backbone='ResNet50', 
    dataset='ADE20K',
    retrieve_existing=False,
    threat_config='configs/threat/adv_attacks.yml',
)
```

- `retrieve_existing`: if set to `True` and a matching evaluation exists in the benchmark, the cached result will be returned. Otherwise, the evaluation will be run.
- `adv_attacks.yml` should include:
  - `threat model`: name of the adversarial attack (supported: `'PGD'`, `'SegPGD'`, `'CosPGD'`)
  - `iterations`: number of attack iterations
  - `epsilon`: allowed perturbation budget (ε)
  - `alpha`: attack step size (ϑ)
  - `lp norm`: norm used to constrain perturbations (`'Linf'` or `'L2'`)

#### 2D Common Corruptions

```python
from semsegbench.evals import evaluate

model, results = evaluate(
    model_name='DeepLabV3', 
    backbone='ResNet50', 
    dataset='ADE20K',
    retrieve_existing=False,
    threat_config='configs/threat/2d_corruptions.yml',
)
```

- `retrieve_existing`: behaves as described above.
- `2d_corruptions.yml` should include:
  - `threat model`: must be `'2DCommonCorruption'`; returns the mean performance across 15 corruption types
  - `severity`: integer from 1 to 5 indicating corruption severity

#### ACDC

```python
from semsegbench.evals import evaluate

model, results = evaluate(
    model_name='DeepLabV3', 
    backbone='ResNet50', 
    dataset='ADE20K',
    retrieve_existing=False,
    threat_config='configs/threat/acdc.yml',
)
```

- `retrieve_existing`: behaves as described above.
- `acdc.yml` should include:
  - `threat model`: must be `'ACDC'`
  - `condition`: one of the supported adverse conditions: `'Fog'`, `'Night'`, `'Rain'`, or `'Snow'`
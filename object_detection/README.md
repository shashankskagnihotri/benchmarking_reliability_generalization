
# DetecBench

DetecBench is a comprehensive benchmarking toolkit for evaluating the robustness and reliability of object detection models against various adversarial attacks and image corruptions. This framework enables researchers and practitioners to systematically test and compare how different detection models perform under challenging conditions.

## Overview

DetecBench supports evaluation of object detection models with:
- Adversarial attacks (PGD, FGSM, BIM)
- Common image corruptions (blur, noise, weather effects, etc.)
- 3D common corruptions (realistic distortions)
- Standard IID (Independent and Identically Distributed) evaluation

The toolkit is built on top of MMDetection and provides a standardized way to benchmark models across different datasets, attack parameters, and corruption types.

## Setup

We use uv for installing dependencies, which provides faster and more reliable dependency resolution. The installation instructions for UV can be found at https://docs.astral.sh/uv/.

If not already done, download this repo. `--recursive` checks out `3DCommonCorruptions`. `mmdetection` is named in `.gitmodules` but this repository does not record a commit for it, so clone it into this directory:

```bash
git clone --recursive https://github.com/shashankskagnihotri/benchmarking_reliability_generalization.git
cd benchmarking_reliability_generalization/object_detection
git clone https://github.com/jjakubassa/mmdetection.git mmdetection
```

Then setup a virtual environment with all dependencies:

```bash
uv venv --seed
source .venv/bin/activate
uv sync --no-build-isolation
uv run mim install mmcv
```

## Download data

COCO and PASCAL VOC are loaded through the MMDetection checkout above. After `mmdetection/` exists, its `tools/misc/download_dataset.py` can fetch `coco2017`, `voc2012`, and `voc2017`. That script is not in this repository until that clone is in place.

## Generate corrupted data

DetecBench can evaluate models on corrupted images that simulate real-world conditions. To generate these corrupted datasets:
- Follow instructions at [Common Corruptions](https://github.com/bethgelab/imagecorruptions)
- Follow instructions at [3D Common Corruptions](https://github.com/EPFL-VILAB/3DCommonCorruptions/tree/main)

The DetecBench expects the following folder structure:

```
data/
├── coco/
│   └── cc/                           # Common corruptions for COCO
│       ├── brightness/               # Corruption type
│       │   ├── severity_1/          # Severity level
│       │   │   ├── 000000000139.jpg # Corrupted images
│       │   │   └── ...
│       │   ├── severity_2/
│       │   ├── severity_3/
│       │   ├── severity_4/
│       │   └── severity_5/
│       ├── frost/
│       ├── gaussian_noise/
│       └── ...
├── VOCdevkit/
│   └── VOC2007/
│       └── cc/                       # Common corruptions for VOC
│           ├── brightness/           # Corruption type
│           │   ├── severity_1/      # Severity level
│           │   │   ├── 000001.jpg   # Corrupted images
│           │   │   └── ...
│           │   ├── severity_2/
│           │   ├── severity_3/
│           │   ├── severity_4/
│           │   └── severity_5/
│           ├── frost/
│           ├── gaussian_noise/
│           └── ...
```

For each dataset (COCO and VOC), corrupted images should be organized in a hierarchical structure:
1. First level: `cc/` directory for common corruptions
2. Second level: Corruption type (e.g., `brightness/`, `frost/`, `gaussian_noise/`)
3. Third level: Severity level from 1-5 (e.g., `severity_1/`, `severity_2/`, etc.)
4. Inside each severity folder: Corrupted images with the same filenames as the original dataset


## Checkpoints

Public segmentation checkpoints are the 45 weights whose URLs are in [`../results/model_provenance.csv`](../results/model_provenance.csv). From this directory:

```bash
python download_models.py --list
python download_models.py
```

`--list` prints the destination and does not download. The files go to `../semantic_segmentation/checkpoints/<architecture>/`, which is where SemSegBench looks. Author-trained segmentation weights are not released, and this repository has no detector checkpoint URLs, so those rows are skipped.

To evaluate a detector, create `models/<name>/` with one `.py` and one `.pth`. `evaluate` uses the first of each (`src/detecbench/detecbench.py`).

Segmentation scores are in [`../results/semseg_evaluations.csv`](../results/semseg_evaluations.csv) and [`../results/semseg_model_summary.csv`](../results/semseg_model_summary.csv). The detector names from the paper are in [`../results/detection_models_manuscript.csv`](../results/detection_models_manuscript.csv). Per-detector detection scores are not in this repository. See [`../docs/results.md`](../docs/results.md).

## Evaluation

DetecBench supports multiple evaluation scenarios. Here are some examples:

### Standard IID evaluation (clean data)

```bash
uv run detecbench task:iid
```

```bash
uv run detecbench --model-folder models/DINO_Swin-L --log-dir ./logs task:iid
```

### PGD attack evaluation

Running PGD (Projected Gradient Descent) attacks via command line:

```bash
uv run detecbench task:pgd
```

Run a more customized PGD attack with specific parameters on the DINO model:
```bash
uv run detecbench --model-folder models/DINO_Swin-L --log-dir ./logs task:pgd --task.epsilon 16 --task.alpha 2.0 --task.steps 30 --task.random-start --task.target 0 --task.norm two
```

You can also run evaluations directly from Python:
```python
from detecbench import attacks, corruptions, evaluate
evaluate(task=attacks.PGD())
```

For more fine-grained control over attack parameters:
```python
from detecbench import attacks, corruptions, evaluate
pgd = attacks.PGD(
    epsilon = 8,
    alpha = 2.55,
    steps = 20,
    norm = "two",
    target = False,
    random_start = False,
)
evaluate(task=pgd, model_folder="./models/RetinaNet_R-101-FPN", log_dir = "./logs", wandb_project = None, wandb_entity = None)
```

### FGSM attack evaluation

```bash
uv run detecbench task:fgsm
```

Configure FGSM (Fast Gradient Sign Method) attack with specific parameters:
```bash
uv run detecbench --model-folder models/DINO_Swin-L --log-dir ./logs task:fgsm --task.epsilon 8 --task.alpha 1.0 --task.norm "inf"
```

You can also run FGSM attacks from Python:
```python
from detecbench import attacks, evaluate
fgsm = attacks.FGSM(
    epsilon = 8,
    alpha = 1.0,
    norm = "inf",
    target = False,
)
evaluate(task=fgsm, model_folder="./models/DINO_Swin-L", log_dir = "./logs")
```

### BIM attack evaluation

```bash
uv run detecbench task:bim
```

Configure BIM (Basic Iterative Method) attack with specific parameters:
```bash
uv run detecbench --model-folder models/DINO_Swin-L --log-dir ./logs task:bim --task.epsilon 8 --task.alpha 1.0 --task.steps 10 --task.norm "two"
```

You can also run BIM attacks from Python:
```python
from detecbench import attacks, evaluate
bim = attacks.BIM(
    epsilon = 8,
    alpha = 1.0,
    steps = 10,
    norm = "inf",
    target = 0,
)
evaluate(task=bim, model_folder="./models/DINO_Swin-L", log_dir = "./logs")
```

### Common corruption evaluation

```bash
# 2D
uv run detecbench task:common-corruption --task.name=gaussian_noise --task.severity 3
# 3D
uv run detecbench task:common-corruption3d --task.name=near_focus --task.severity 3
```

or from Python

```python
from detecbench import attacks, corruptions, evaluate

cc_contrast = corruptions.CommonCorruption(name="contrast", severity=3)
evaluate(task=cc_contrast, model_folder="./models/RetinaNet_R-101-FPN", log_dir = "./logs", wandb_project = None, wandb_entity = None)

cc3d_near_focus = corruptions.CommonCorruption3d(name="near_focus", severity=3)
evaluate(task=cc3d_near_focus, model_folder="./models/RetinaNet_R-101-FPN", log_dir = "./logs", wandb_project = None, wandb_entity = None)
```

### Logging with Weights & Biases

```bash
uv run detecbench task:pgd --task.wandb_project your_project --task.wandb_entity your_username
```

## Adding custom models

To add your own models for evaluation:
1. Create a new folder in the `models` directory
2. Add your model configuration (`.py`) and checkpoint (`.pth`) files
3. Run the evaluation specifying your model folder

## Contributing

Contributions to DetecBench are welcome! Please feel free to submit pull requests or open issues to improve the toolkit.

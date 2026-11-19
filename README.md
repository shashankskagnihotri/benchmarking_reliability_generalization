# SemSegBench and DetecBench

Benchmarking reliability and generalization for semantic segmentation and object detection.

Paper: [arXiv:2505.18015](https://arxiv.org/abs/2505.18015)

The release records 76 semantic segmentation checkpoints, 63 object detectors, and 5,780 evaluations: 1,734 segmentation and 4,046 object detection.

```bibtex
@article{agnihotri2025semsegbench,
  title={SEMSEGBENCH \& DETECBENCH: Benchmarking Reliability and Generalization Beyond Classification},
  author={Agnihotri, Shashank and Schader, David and Jakubassa, Jonas and Sharei, Nico and Kral, Simon and Weber, Ruben and Keuper, Margret and others},
  journal={arXiv preprint arXiv:2505.18015},
  year={2025}
}
```

```bash
git clone --recursive https://github.com/shashankskagnihotri/benchmarking_reliability_generalization.git
cd benchmarking_reliability_generalization
```

`--recursive` checks out `object_detection/3DCommonCorruptions`. DetecBench also uses MMDetection:

```bash
git clone https://github.com/jjakubassa/mmdetection.git object_detection/mmdetection
```

## Map

| | |
| --- | --- |
| [SemSegBench](semantic_segmentation/README.md) | Install, data layout, `evaluate` |
| [DetecBench](object_detection/README.md) | Install, attacks, corruptions |
| [Models](semantic_segmentation/SUPPORTED_MODELS.md) | 76 segmentation checkpoints, including crop size |
| [Provenance](docs/model_provenance.md) | 76 segmentation checkpoints and 63 detectors |
| [Models trained for this study](docs/model_provenance.md#models-trained-for-this-study) | The 15 segmentation runs with a training script and config |
| [Author-trained detectors](docs/model_provenance.md#object-detection) | The 10 training configs |
| [Results](docs/results.md) | 5,780 evaluations, and what a row means |
| [Evaluations](results/evaluation_manifest.csv) | 5,780 evaluations |
| [Segmentation scores](results/semseg_evaluations.csv) | 1,734 segmentation evaluations |
| [Detection scores](docs/results.md#result-files) | 4,046 object-detection evaluations |
| [Model summary](results/semseg_model_summary.csv) | Clean mIoU, ReM, GAM, parameter count |
| [RGB-text extension](docs/vision_language_semseg.md) | Six open-vocabulary segmentors, 258 evaluations |
| [Detection summary](results/detection_model_summary.csv) | Clean mAP, ReM, GAM, parameter count |

Public segmentation checkpoints:

```bash
python object_detection/download_models.py --list
python object_detection/download_models.py
```

`--list` prints the public segmentation checkpoints. The second command downloads them.

## Reproduce the tables

```bash
python analysis/reproduce_results.py
```

Reads the stored segmentation and detection tables and rewrites the manifest, summaries, and figures under `results/`. No GPU and no extra packages. Details are in [docs/results.md](docs/results.md).

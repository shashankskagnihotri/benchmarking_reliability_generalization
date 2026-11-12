# SemSegBench and DetecBench

Benchmarking reliability and generalization for semantic segmentation and object detection.

Paper: [arXiv:2505.18015](https://arxiv.org/abs/2505.18015)

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

`--recursive` checks out `object_detection/3DCommonCorruptions`. `.gitmodules` also names `object_detection/mmdetection` (`https://github.com/jjakubassa/mmdetection`), but this repository does not record a commit for it. Clone that repository into `object_detection/mmdetection` yourself. DetecBench imports it from there.

## Map

| | |
| --- | --- |
| [SemSegBench](semantic_segmentation/README.md) | Install, data layout, `evaluate` |
| [DetecBench](object_detection/README.md) | Install, attacks, corruptions |
| [Models](semantic_segmentation/SUPPORTED_MODELS.md) | 76 segmentation checkpoints, including crop size |
| [Provenance](docs/model_provenance.md) | Public vs author-trained, configs, checkpoint URLs |
| [Models trained for this study](docs/model_provenance.md#models-trained-for-this-study) | The 15 runs with a training script and config |
| [Results](docs/results.md) | Where the numbers are, and what a row means |
| [Evaluation manifest](results/semseg_evaluations.csv) | Recorded segmentation evaluations |
| [Detection manifest](results/detection_evaluations.csv) | Recorded MS-COCO detection evaluations |
| [Model summary](results/semseg_model_summary.csv) | Clean mIoU, ReM, GAM, parameter count |
| [Detection summary](results/detection_model_summary.csv) | Clean mAP, ReM, GAM, parameter count |

Public segmentation checkpoints:

```bash
python object_detection/download_models.py --list
python object_detection/download_models.py
```

`--list` does not download. Author-trained weights and detector checkpoints are not in the table of URLs, so the script skips them.

## Reproduce the tables

```bash
python analysis/reproduce_results.py
```

Reads `semantic_segmentation/semsegbench/Config-Paths.csv` and rewrites the tables and figures under `results/`. No GPU and no extra packages. Details are in [docs/results.md](docs/results.md).

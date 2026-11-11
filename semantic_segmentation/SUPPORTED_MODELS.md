# Supported models

Each row is one evaluated checkpoint. The same architecture, backbone, and dataset can appear twice when the input crop differs. `model_id` is the unique name.

Checkpoint files are not in git. `load_model` reads the config path in `results/model_provenance.csv` for that `model_id`. A local `.pth` is optional: `get_checkpoint_path` looks in `checkpoints/<architecture>/`. ResNet filenames are matched as `r50`, `r101`, or `r18`. Other backbones are matched as written (`mit-b0`, `swin-b`). The filename must also contain the dataset, and the crop when more than one file matches. If no local file is found, loading falls back to the MMSegmentation config name, which downloads a published checkpoint when one is registered.

Provenance, training settings, and checkpoint URLs: [docs/model_provenance.md](../docs/model_provenance.md).

| model_id | architecture | backbone | dataset | crop |
| --- | --- | --- | --- | --- |
| `unet_custom_cityscapes_512x1024` | unet | custom | cityscapes | 512x1024 |
| `pspnet_resnet50_cityscapes_512x1024` | pspnet | resnet50 | cityscapes | 512x1024 |
| `pspnet_resnet101_cityscapes_512x1024` | pspnet | resnet101 | cityscapes | 512x1024 |
| `pspnet_resnet18_cityscapes_512x1024` | pspnet | resnet18 | cityscapes | 512x1024 |
| `deeplabv3_resnet50_cityscapes_512x1024` | deeplabv3 | resnet50 | cityscapes | 512x1024 |
| `deeplabv3_resnet101_cityscapes_512x1024` | deeplabv3 | resnet101 | cityscapes | 512x1024 |
| `deeplabv3_resnet18_cityscapes_512x1024` | deeplabv3 | resnet18 | cityscapes | 512x1024 |
| `deeplabv3plus_resnet50_cityscapes_512x1024` | deeplabv3plus | resnet50 | cityscapes | 512x1024 |
| `deeplabv3plus_resnet101_cityscapes_512x1024` | deeplabv3plus | resnet101 | cityscapes | 512x1024 |
| `deeplabv3plus_resnet18_cityscapes_512x1024` | deeplabv3plus | resnet18 | cityscapes | 512x1024 |
| `segformer_mit-b0_cityscapes_512x1024` | segformer | mit-b0 | cityscapes | 512x1024 |
| `segformer_mit-b1_cityscapes_512x1024` | segformer | mit-b1 | cityscapes | 512x1024 |
| `segformer_mit-b2_cityscapes_512x1024` | segformer | mit-b2 | cityscapes | 512x1024 |
| `segformer_mit-b3_cityscapes_512x1024` | segformer | mit-b3 | cityscapes | 512x1024 |
| `segformer_mit-b4_cityscapes_512x1024` | segformer | mit-b4 | cityscapes | 512x1024 |
| `segformer_mit-b5_cityscapes_512x1024` | segformer | mit-b5 | cityscapes | 512x1024 |
| `segformer_mit-b0_cityscapes_1024x1024` | segformer | mit-b0 | cityscapes | 1024x1024 |
| `segformer_mit-b1_cityscapes_1024x1024` | segformer | mit-b1 | cityscapes | 1024x1024 |
| `segformer_mit-b2_cityscapes_1024x1024` | segformer | mit-b2 | cityscapes | 1024x1024 |
| `segformer_mit-b3_cityscapes_1024x1024` | segformer | mit-b3 | cityscapes | 1024x1024 |
| `segformer_mit-b4_cityscapes_1024x1024` | segformer | mit-b4 | cityscapes | 1024x1024 |
| `segformer_mit-b5_cityscapes_1024x1024` | segformer | mit-b5 | cityscapes | 1024x1024 |
| `upernet_internimage-t_cityscapes_512x1024` | upernet | internimage-t | cityscapes | 512x1024 |
| `upernet_internimage-s_cityscapes_512x1024` | upernet | internimage-s | cityscapes | 512x1024 |
| `upernet_internimage-b_cityscapes_512x1024` | upernet | internimage-b | cityscapes | 512x1024 |
| `upernet_internimage-l_cityscapes_512x1024` | upernet | internimage-l | cityscapes | 512x1024 |
| `upernet_internimage-xl_cityscapes_512x1024` | upernet | internimage-xl | cityscapes | 512x1024 |
| `mask2former_resnet50_cityscapes_512x1024` | mask2former | resnet50 | cityscapes | 512x1024 |
| `mask2former_resnet101_cityscapes_512x1024` | mask2former | resnet101 | cityscapes | 512x1024 |
| `mask2former_swin-t_cityscapes_512x1024` | mask2former | swin-t | cityscapes | 512x1024 |
| `mask2former_swin-s_cityscapes_512x1024` | mask2former | swin-s | cityscapes | 512x1024 |
| `mask2former_swin-b_cityscapes_512x1024` | mask2former | swin-b | cityscapes | 512x1024 |
| `upernet_beit-b_ade20k_640x640` | upernet | beit-b | ade20k | 640x640 |
| `upernet_beit-l_ade20k_640x640` | upernet | beit-l | ade20k | 640x640 |
| `pspnet_resnet50_ade20k_512x512` | pspnet | resnet50 | ade20k | 512x512 |
| `pspnet_resnet101_ade20k_512x512` | pspnet | resnet101 | ade20k | 512x512 |
| `deeplabv3_resnet50_ade20k_512x512` | deeplabv3 | resnet50 | ade20k | 512x512 |
| `deeplabv3_resnet101_ade20k_512x512` | deeplabv3 | resnet101 | ade20k | 512x512 |
| `deeplabv3plus_resnet50_ade20k_512x512` | deeplabv3plus | resnet50 | ade20k | 512x512 |
| `deeplabv3plus_resnet101_ade20k_512x512` | deeplabv3plus | resnet101 | ade20k | 512x512 |
| `segformer_mit-b0_ade20k_512x512` | segformer | mit-b0 | ade20k | 512x512 |
| `segformer_mit-b1_ade20k_512x512` | segformer | mit-b1 | ade20k | 512x512 |
| `segformer_mit-b2_ade20k_512x512` | segformer | mit-b2 | ade20k | 512x512 |
| `segformer_mit-b3_ade20k_512x512` | segformer | mit-b3 | ade20k | 512x512 |
| `segformer_mit-b4_ade20k_512x512` | segformer | mit-b4 | ade20k | 512x512 |
| `segformer_mit-b5_ade20k_512x512` | segformer | mit-b5 | ade20k | 512x512 |
| `upernet_internimage-t_ade20k_512x512` | upernet | internimage-t | ade20k | 512x512 |
| `upernet_internimage-s_ade20k_512x512` | upernet | internimage-s | ade20k | 512x512 |
| `upernet_internimage-b_ade20k_512x512` | upernet | internimage-b | ade20k | 512x512 |
| `upernet_internimage-l_ade20k_640x640` | upernet | internimage-l | ade20k | 640x640 |
| `upernet_internimage-xl_ade20k_640x640` | upernet | internimage-xl | ade20k | 640x640 |
| `upernet_internimage-h_ade20k_896x896` | upernet | internimage-h | ade20k | 896x896 |
| `mask2former_resnet50_ade20k_512x512` | mask2former | resnet50 | ade20k | 512x512 |
| `mask2former_resnet101_ade20k_512x512` | mask2former | resnet101 | ade20k | 512x512 |
| `mask2former_swin-t_ade20k_512x512` | mask2former | swin-t | ade20k | 512x512 |
| `mask2former_swin-s_ade20k_512x512` | mask2former | swin-s | ade20k | 512x512 |
| `mask2former_swin-b_ade20k_640x640` | mask2former | swin-b | ade20k | 640x640 |
| `mask2former_swin-b_ade20k_512x512` | mask2former | swin-b | ade20k | 512x512 |
| `pspnet_resnet50_voc_512x512` | pspnet | resnet50 | voc | 512x512 |
| `pspnet_resnet101_voc_512x512` | pspnet | resnet101 | voc | 512x512 |
| `deeplabv3_resnet50_voc_512x512` | deeplabv3 | resnet50 | voc | 512x512 |
| `deeplabv3_resnet101_voc_512x512` | deeplabv3 | resnet101 | voc | 512x512 |
| `deeplabv3plus_resnet50_voc_512x512` | deeplabv3plus | resnet50 | voc | 512x512 |
| `deeplabv3plus_resnet101_voc_512x512` | deeplabv3plus | resnet101 | voc | 512x512 |
| `segformer_mit-b0_voc_512x512` | segformer | mit-b0 | voc | 512x512 |
| `segformer_mit-b1_voc_512x512` | segformer | mit-b1 | voc | 512x512 |
| `segformer_mit-b2_voc_512x512` | segformer | mit-b2 | voc | 512x512 |
| `segformer_mit-b3_voc_512x512` | segformer | mit-b3 | voc | 512x512 |
| `segformer_mit-b4_voc_512x512` | segformer | mit-b4 | voc | 512x512 |
| `segformer_mit-b5_voc_512x512` | segformer | mit-b5 | voc | 512x512 |
| `upernet_internimage-t_voc_512x512` | upernet | internimage-t | voc | 512x512 |
| `upernet_internimage-s_voc_512x512` | upernet | internimage-s | voc | 512x512 |
| `upernet_internimage-b_voc_512x512` | upernet | internimage-b | voc | 512x512 |
| `mask2former_swin-t_voc_512x512` | mask2former | swin-t | voc | 512x512 |
| `mask2former_swin-s_voc_512x512` | mask2former | swin-s | voc | 512x512 |
| `mask2former_swin-b_voc_512x512` | mask2former | swin-b | voc | 512x512 |

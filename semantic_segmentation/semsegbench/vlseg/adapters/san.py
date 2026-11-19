"""SAN ViT-L/14 adapter.

The checkpoint is the OpenMMLab COCO-Stuff model. Class names are replaced
at inference with the benchmark vocabulary. The ViLD templates and the
learned background embedding stay as published. ``predict_by_feat`` drops
the background channel, so the returned scores have one channel per dataset
class. Scores are resized back to the attacked image.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ..prompts import prompt_nouns

ROOT = Path(__file__).resolve().parents[4]
MMSEG_ROOT = ROOT / "semantic_segmentation" / "mmsegmentation"
CONFIG = MMSEG_ROOT / "configs" / "san" / "san-vit-l14_coco-stuff164k-640x640.py"
CHECKPOINT = ROOT / "semantic_segmentation" / "checkpoints" / "vlseg" / "san-vit-l14_20230907-a11e098f.pth"
MEAN = (122.7709, 116.7460, 104.0937)
STD = (68.5005, 66.6322, 70.3232)


def _short_edge(height, width, scale=640, max_size=2560):
    size = float(scale)
    factor = size / min(height, width)
    if height < width:
        new_h, new_w = size, factor * width
    else:
        new_h, new_w = factor * height, size
    if max(new_h, new_w) > max_size:
        capped = max_size / max(new_h, new_w)
        new_h *= capped
        new_w *= capped
    return int(new_w + 0.5), int(new_h + 0.5)


class SANSegmentor:
    def __init__(self, spec, device):
        import sys
        sys.path.insert(0, str(MMSEG_ROOT))
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.runner.checkpoint import load_checkpoint
        from mmseg.registry import MODELS

        init_default_scope("mmseg")
        cfg = Config.fromfile(str(CONFIG))
        nouns = list(prompt_nouns(spec["dataset"]))
        cfg.model.text_encoder.dataset_name = None
        cfg.model.text_encoder.vocabulary = nouns
        cfg.model.pretrained = None
        model = MODELS.build(cfg.model)
        load_checkpoint(model, str(CHECKPOINT), map_location="cpu", strict=True)
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.num_classes = len(nouns)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def freeze(self):
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def scores(self, image):
        _, _, height, width = image.shape
        new_w, new_h = _short_edge(height, width)
        resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        normalized = (resized - self.mean) / self.std
        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32
        padded = F.pad(normalized, (0, pad_w, 0, pad_h))
        meta = [dict(img_shape=(new_h + pad_h, new_w + pad_w), ori_shape=(height, width), pad_shape=(new_h + pad_h, new_w + pad_w), padding_size=[0, pad_w, 0, pad_h])]
        logits = self.model.encode_decode(padded, meta)
        logits = logits[:, :, :new_h, :new_w]
        if logits.shape[1] != self.num_classes:
            raise RuntimeError(f"SAN returned {logits.shape[1]} classes, expected {self.num_classes}")
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return logits

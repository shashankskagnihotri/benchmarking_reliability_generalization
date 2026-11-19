"""Model adapters. Each one returns differentiable dataset-class scores."""

from __future__ import annotations


class Segmentor:
    def scores(self, image):
        raise NotImplementedError

    def freeze(self):
        raise NotImplementedError


def load_segmentor(spec: dict, device: str) -> Segmentor:
    key = spec["model_key"]
    if key == "san":
        from .san import SANSegmentor
        return SANSegmentor(spec, device)
    if key == "catseg":
        from .catseg import CATSegSegmentor
        return CATSegSegmentor(spec, device)
    if key == "sed":
        from .sed import SEDSegmentor
        return SEDSegmentor(spec, device)
    if key == "fcclip":
        from .fcclip import FCCLIPSegmentor
        return FCCLIPSegmentor(spec, device)
    if key == "maftp":
        from .maftp import MAFTSegmentor
        return MAFTSegmentor(spec, device)
    if key == "sam3":
        from .sam3 import SAM3Segmentor
        return SAM3Segmentor(spec, device)
    raise KeyError(key)

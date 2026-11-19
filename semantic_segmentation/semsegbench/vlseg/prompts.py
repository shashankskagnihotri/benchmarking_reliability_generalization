"""Fixed text prompts. Cityscapes and ACDC share one prompt list per model."""

from __future__ import annotations

from .classes import ACDC_CLASSES, ADE20K_CLASSES, CITYSCAPES_CLASSES

SAN_VILD = (
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
)

# CAT-Seg configs/vitl_336.yaml sets PROMPT_ENSEMBLE_TYPE: single.
CATSEG_SINGLE = ("A photo of a {} in the scene",)

# SED configs/convnextL_768.yaml sets PROMPT_ENSEMBLE_TYPE: imagenet.
# The active IMAGENET_TEMPLATES list in the official file is the CLIP set.
SED_IMAGENET = (
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
)

PROMPT_PROTOCOL = {
    "san": {
        "template_name": "mmseg_vild",
        "templates": SAN_VILD,
        "background": "cat_bg embedding is appended and dropped by SideAdapterCLIPHead.predict_by_feat",
    },
    "catseg": {
        "template_name": "catseg_single",
        "templates": CATSEG_SINGLE,
        "background": "no extra background class; argmax over the dataset classes",
    },
    "sed": {
        "template_name": "sed_imagenet",
        "templates": SED_IMAGENET,
        "background": "no extra background class; argmax over the dataset classes",
    },
    "fcclip": {
        "template_name": "fcclip_default_templates",
        "templates": (),  # filled from the official FC-CLIP template list at adapter load
        "background": "official FC-CLIP semantic inference, including its background channel handling",
    },
    "maftp": {
        "template_name": "maft_vild",
        "templates": SAN_VILD,
        "background": "official MAFT+ semantic head",
    },
    "sam3": {
        "template_name": "class_name",
        "templates": ("{}",),
        "background": "no background prompt; absent classes get score 0",
    },
}


def class_names(dataset: str) -> tuple[str, ...]:
    if dataset == "ade20k":
        names = ADE20K_CLASSES
    elif dataset in {"cityscapes", "acdc"}:
        names = CITYSCAPES_CLASSES if dataset == "cityscapes" else ACDC_CLASSES
    else:
        raise ValueError(dataset)
    if dataset == "ade20k" and len(names) != 150:
        raise RuntimeError(len(names))
    if dataset != "ade20k" and len(names) != 19:
        raise RuntimeError(len(names))
    return names


def prompt_nouns(dataset: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in class_names(dataset))


def prompts_for(model: str, dataset: str) -> list[str]:
    nouns = prompt_nouns(dataset)
    templates = PROMPT_PROTOCOL[model]["templates"]
    if not templates:
        raise RuntimeError(f"{model} templates are loaded by its adapter")
    rendered = []
    for noun in nouns:
        rendered.append([template.format(noun) for template in templates])
    return rendered

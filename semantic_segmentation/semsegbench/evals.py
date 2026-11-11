import os
import shutil
import torch
import pandas as pd
import yaml
import json
import math

from pathlib import Path
from mmseg.apis import MMSegInferencer
from mmengine.config import Config
from torch.hub import _get_torch_home

from .semseg_wrapper import SemSegWrapper
from .utils import get_args_parser, get_config_path, get_checkpoint_path, get_results, _norm_crop
from mmsegmentation.tools.test import main


common_corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
] 
    
attacks = ['pgd', 'segpgd', 'cospgd']


def _cached_filters(model_name, backbone, dataset, crop_size):
    filters = {"architecture": model_name, "backbone": backbone, "dataset": dataset}
    crop = _norm_crop(crop_size)
    if crop:
        filters["crop_size"] = crop
    return filters


def _one_cached_row(results_df, model_name, backbone, dataset):
    if results_df is None or results_df.empty:
        return None
    if len(results_df) > 1:
        raise ValueError(
            f"More than one checkpoint matches {model_name}, {backbone}, {dataset}. "
            "Pass crop_size, for example 512x1024 or 1024x1024."
        )
    return results_df.iloc[0]


def load_model(model_name: str, backbone: str, dataset: str, crop_size: str = None, load_without_weights = False):
    """
    Load a pre-configured segmentation model with or without pretrained weights.

    - model_name (str): Name of the segmentation model architecture (e.g., 'DeepLabV3', 'SegFormer').
    - backbone (str): Backbone network used in the model (e.g., 'ResNet50', 'MiT-B5').
    - dataset (str): Name of the dataset the model was trained on (e.g., 'ADE20K', 'Cityscapes').
    - (optional) crop_size (str): Crop size variant if the model config varies by input size (e.g., '512x512').
    - load_without_weights (bool) default False: If True, only the model architecture is loaded without pretrained weights.
    """
    model_name = model_name.lower()
    backbone = backbone.lower()
    dataset = dataset.lower()

    config_path = get_config_path(model_name, backbone, dataset, crop_size)

    if load_without_weights:
        inferencer = MMSegInferencer(model=str(config_path), weights=None)
    else:
        try:
            checkpoint_path = get_checkpoint_path(model_name, backbone, dataset, crop_size)
            inferencer = MMSegInferencer(model=str(config_path), weights=str(checkpoint_path))
        except (FileNotFoundError, RuntimeError) as e:
            inferencer = MMSegInferencer(model=config_path.stem) # automatically download the checkpoint
            torch_home = _get_torch_home()
            torch_ckpt_dir = os.path.join(torch_home, "hub", "checkpoints")

            for file in os.listdir(torch_ckpt_dir):
                if file.startswith(model_name) and file.endswith(".pth"):
                    # move the checkpoint to checkpoints/model_name/
                    ckpt_path = os.path.join(torch_ckpt_dir, file)
                    target_dir = os.path.join("checkpoints", model_name)
                    os.makedirs(target_dir, exist_ok=True)
                    new_ckpt_path = os.path.join(target_dir, file)
                    shutil.move(ckpt_path, new_ckpt_path)

    return inferencer.model


def evaluate(
    model_name: str, backbone: str, dataset: str, retrieve_existing: bool, threat_config: str, crop_size: str = None,
):
    """
    Evaluate a segmentation model under a specified robustness threat scenario. 
    
    This function loads a model and evaluates it according to a defined threat model, such as
    adversarial attacks, common corruptions, or adverse conditions (e.g., ACDC). The evaluation
    is either run from scratch or retrieved from existing evaluation in the benchmark.
    
    - model_name (str): Name of the segmentation model architecture (e.g., 'DeepLabV3', 'SegFormer').
    - backbone (str): Backbone network used in the model (e.g., 'ResNet50', 'MiT-B5').
    - dataset (str): Name of the dataset the model was trained on (e.g., 'ADE20K', 'Cityscapes').
    - retrieve_existing (bool): If True, returns cached results if a matching evaluation exists. Otherwise, reruns evaluation.
    - threat_config (str): Path to a YAML file defining the threat model and parameters for evaluation.
    - (optional) crop_size (str): Crop size variant if the model config varies by input size (e.g., '512x512').
    """
    model_name = model_name.lower()
    backbone = backbone.lower()
    dataset = dataset.lower()
    model = load_model(model_name, backbone, dataset, crop_size)

    if not os.path.exists(threat_config):
        raise ValueError(f"Config {threat_config} does not exist")

    with open(threat_config, 'r') as f:
        config = yaml.safe_load(f)
    threat_model = config['threat_model'].lower()
    
    if threat_model == '2dcommoncorruption':
        severity = int(config['severity'])
        if severity < 1 or severity > 5:
            raise ValueError("Severity must be an integer between 1 and 5.")

        if retrieve_existing and severity == 3:
            wrapper = SemSegWrapper("semsegbench/Config-Paths.csv")
            
            results_df = wrapper.get_result(
                filters=_cached_filters(model_name, backbone, dataset, crop_size),
                return_metric=["aAcc", "mIoU", "mAcc"],
                test_type='2d',
            )

            results = {}
            row = _one_cached_row(results_df, model_name, backbone, dataset)
            if row is not None:
                for corruption in common_corruptions:
                    metrics = ['aAcc', 'mIoU', 'mAcc']
                    metric_values = {metric: row[f'{metric}_2d_{corruption}'] for metric in metrics}

                    if all(not math.isnan(value) for value in metric_values.values()):
                        results[corruption] = metric_values
            
            if len(results) == len(common_corruptions):
                return model, results
        
        # run 2D Common Corruptions
        args = get_args_parser()
        config_path = get_config_path(model_name, backbone, dataset, crop_size)
        checkpoint_path = get_checkpoint_path(model_name, backbone, dataset, crop_size)
        
        args.config = str(config_path)
        args.checkpoint = str(checkpoint_path)
        
        results = {}

        for corruption in common_corruptions:
            args.cfg_options = {
                'model.data_preprocessor.corruption': corruption,
                'model.data_preprocessor.severity': severity,
            }
            args.work_dir = f'work_dirs/2d_corruptions/{config_path.stem}/{corruption}_severity{severity}'
            
            with torch.autocast(device_type="cuda"):
                main(args)
            
            # load results from json
            metric_values = get_results(args.work_dir)
            results[corruption] = metric_values

        return model, results
            
    elif threat_model in attacks:
        iterations = int(config['iterations'])
        epsilon = float(config['epsilon'])
        alpha = float(config['alpha'])
        lp_norm = config['lp_norm'].lower()

        if retrieve_existing and iterations == 20:
            default_combinations = {
                (8.0, 0.01, 'linf'),
                (64.0, 0.1, 'l2')
            }

            if (epsilon, alpha, lp_norm) in default_combinations:
                wrapper = SemSegWrapper("semsegbench/Config-Paths.csv")

                results_df = wrapper.get_result(
                    filters=_cached_filters(model_name, backbone, dataset, crop_size),
                    return_metric=["aAcc", "mIoU", "mAcc"],
                    test_type=threat_model,
                )

                row = _one_cached_row(results_df, model_name, backbone, dataset)
                if row is not None:
                    metrics = ['aAcc', 'mIoU', 'mAcc']
                    results = {metric: row[f'{metric}_{threat_model}_{lp_norm}'] for metric in metrics}
                    if all(not math.isnan(value) for value in results.values()):
                        return model, results

        # run Adversarial Attacks
        args = get_args_parser()
        config_path = get_config_path(model_name, backbone, dataset, crop_size)
        checkpoint_path = get_checkpoint_path(model_name, backbone, dataset, crop_size)
        
        args.config = str(config_path)
        args.checkpoint = str(checkpoint_path)
        args.cfg_options = {
            'model.perform_attack': True,
            'model.attack_cfg.name': threat_model,
            'model.attack_cfg.norm': lp_norm,
            'model.attack_cfg.iterations': iterations,
            'model.attack_cfg.alpha': alpha,
            'model.attack_cfg.epsilon': epsilon,
        }
        args.work_dir = f'work_dirs/adv_attacks/{config_path.stem}/{threat_model}_{lp_norm}_iterations{iterations}_alpha{alpha}_epsilon{epsilon}'
        
        with torch.autocast(device_type="cuda"):
            main(args)

        # load results from json
        results = get_results(args.work_dir)
        return model, results

    elif threat_model == 'acdc':
        condition = config['condition'].lower()

        if condition not in ['fog', 'night', 'rain', 'snow']:
            raise ValueError("Invalid condition specified: must be one of ['Fog', 'Night', 'Rain', 'Snow'].")

        if retrieve_existing:
            wrapper = SemSegWrapper("semsegbench/Config-Paths.csv")
            
            results_df = wrapper.get_result(
                filters=_cached_filters(model_name, backbone, dataset, crop_size),
                return_metric=["aAcc", "mIoU", "mAcc"],
                test_type='acdc',
            )

            row = _one_cached_row(results_df, model_name, backbone, dataset)
            if row is not None:
                metrics = ['aAcc', 'mIoU', 'mAcc']
                results = {metric: row[f'{metric}_acdc_{condition}'] for metric in metrics}
                if all(not math.isnan(value) for value in results.values()):
                    return model, results

        # run ACDC
        args = get_args_parser()
        config_path = get_config_path(model_name, backbone, dataset, crop_size)
        checkpoint_path = get_checkpoint_path(model_name, backbone, dataset, crop_size)
        
        args.config = str(config_path)
        args.checkpoint = str(checkpoint_path)
        args.work_dir = f'work_dirs/acdc/{config_path.stem}/{condition}'

        acdc_cfg = Config.fromfile(f'mmsegmentation/configs/_base_/datasets/acdc_{condition}.py')
        args.cfg_options = acdc_cfg

        with torch.autocast(device_type="cuda"):
            main(args)

        # load results from json
        results = get_results(args.work_dir)
        return model, results
        
    else:
        ValueError(f"Threat model {config['threat_model']} is not supported")

    return model, results


if __name__ == "__main__":
    model = load_model(
        model_name='DeepLabV3', 
        backbone='ResNet50', 
        dataset='ADE20K',
    )
    
    model, results = evaluate(
        model_name='DeepLabV3', 
        backbone='ResNet50', 
        dataset='ADE20K',
        retrieve_existing=False,
        threat_config='configs/threat/2d_corruptions.yml',
    )
    print(results)

    model, results = evaluate(
        model_name='DeepLabV3', 
        backbone='ResNet50', 
        dataset='ADE20K',
        retrieve_existing=False,
        threat_config='configs/threat/adv_attacks.yml',
    )
    print(results)

    model, results = evaluate(
        model_name='DeepLabV3', 
        backbone='ResNet50', 
        dataset='ADE20K',
        retrieve_existing=False,
        threat_config='configs/threat/acdc.yml',
    )
    print(results)
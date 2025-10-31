#!/usr/bin/env python3
"""
PCB (Parameter Competition Balancing) Merging for FusionBench

Based on: "PCB-Merging: Parameter Competition Balancing for Model Merging"
GitHub: https://github.com/duguodong7/pcb-merging

The algorithm addresses parameter competition in model merging by:
1. Computing intra-balancing (self-competition within tasks)
2. Computing inter-balancing (cross-task competition)
3. Using drop and rescale mechanism to balance parameter contributions
"""

import logging
from copy import deepcopy
from typing import Dict, Optional, Union
import gc

import torch
from torch import nn
from tqdm import tqdm

from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import state_dict_to_vector, vector_to_state_dict

log = logging.getLogger(__name__)

__all__ = ["PCBMergingAlgorithm"]


def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Normalize tensor along specified dimension"""
    return torch.nn.functional.normalize(tensor, p=2, dim=dim)


def clamp_by_percentile(tensor: torch.Tensor, keep_ratio: float, min_val: float = 0.0) -> torch.Tensor:
    """
    Clamp tensor values by removing top (1-keep_ratio) portion based on magnitude

    Args:
        tensor: Input tensor
        keep_ratio: Ratio of values to keep (e.g., 0.9 keeps bottom 90%)
        min_val: Minimum value after clamping

    Returns:
        Clamped tensor
    """
    if keep_ratio >= 1.0:
        return tensor

    # Sort by absolute value and find threshold
    sorted_abs, _ = torch.sort(torch.abs(tensor), dim=1, descending=True)
    threshold_idx = int(tensor.shape[1] * (1 - keep_ratio))
    threshold_idx = max(0, min(threshold_idx, tensor.shape[1] - 1))

    thresholds = sorted_abs[:, threshold_idx:threshold_idx+1]

    # Clamp values above threshold
    mask = torch.abs(tensor) <= thresholds
    clamped = torch.where(mask, tensor, torch.sign(tensor) * thresholds)

    return torch.clamp(clamped, min=min_val)


class PCBMergingAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    PCB (Parameter Competition Balancing) Merging Algorithm

    Balances parameter competition through:
    1. Intra-balancing: Competition within individual tasks
    2. Inter-balancing: Competition across different tasks
    3. Drop and rescale: Remove highly competitive parameters and rescale
    """

    def __init__(
        self,
        scaling_factor: float = 0.3,
        pcb_ratio: float = 0.1,
        min_ratio: float = 0.0001,
        max_ratio: float = 0.0001,
        **kwargs
    ):
        """
        Initialize PCB Merging Algorithm

        Args:
            scaling_factor: Final scaling factor for merged task vector
            pcb_ratio: Ratio of highest competition parameters to drop (adjustable hyperparameter)
            min_ratio: Minimum percentile for outlier clamping
            max_ratio: Maximum percentile for outlier clamping
        """
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor
        self.pcb_ratio = pcb_ratio  # Adjustable hyperparameter
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        # Statistics
        self.stats = {
            'total_parameters': 0,
            'dropped_parameters': 0,
            'competition_stats': {}
        }

    def pcb_merging(self, flat_task_checks: torch.Tensor) -> torch.Tensor:
        """
        Core PCB merging function

        Args:
            flat_task_checks: Tensor of shape [n_models, n_params] containing task vectors

        Returns:
            Merged task vector of shape [n_params]
        """
        n_models, n_params = flat_task_checks.shape
        device = flat_task_checks.device

        log.info(f"PCB merging {n_models} models with {n_params} parameters each")

        # Step 1: Outlier clamping using percentiles
        with self.profile("outlier clamping"):
            all_checks = clamp_by_percentile(flat_task_checks, 1 - self.max_ratio, self.min_ratio)
            all_checks_abs = torch.abs(all_checks)

        # Step 2: Intra-balancing (self-competition)
        with self.profile("intra-balancing"):
            # Normalize each task vector and square for importance
            self_pcb = normalize(all_checks_abs, dim=1) ** 2
            # Exponential amplification based on number of tasks
            self_pcb_act = torch.exp(n_models * self_pcb)

        # Step 3: Inter-balancing (cross-task competition)
        with self.profile("inter-balancing"):
            # Interaction between tasks: each param * sum across all tasks
            task_sum = torch.sum(all_checks, dim=0, keepdim=True)  # [1, n_params]
            cross_pcb = all_checks * task_sum  # [n_models, n_params]
            # Apply tanh activation
            cross_pcb_act = torch.tanh(cross_pcb)

        # Step 4: Combined competition score
        with self.profile("competition combination"):
            task_pcb = self_pcb_act * cross_pcb_act

        # Step 5: Drop and rescale mechanism
        with self.profile("drop and rescale"):
            # Drop top pcb_ratio% of highest competition parameters
            scale = clamp_by_percentile(task_pcb, 1 - self.pcb_ratio, 0.0)
            # Normalize scaling factors
            scale = normalize(scale, dim=1)

        # Step 6: Weighted merging
        with self.profile("weighted merging"):
            # Weighted sum of task vectors
            numerator = torch.sum(all_checks * scale, dim=0)
            denominator = torch.clamp(torch.sum(scale, dim=0), min=1e-12)
            merged_tv = numerator / denominator

        # Update statistics
        self.stats['total_parameters'] = n_params
        dropped_count = ((scale == 0).sum(dim=1).float().mean() * n_params).item()
        self.stats['dropped_parameters'] = int(dropped_count)

        log.info(f"PCB merging completed. Dropped ~{dropped_count:.0f} parameters ({dropped_count/n_params*100:.1f}%)")

        return merged_tv

    @torch.no_grad()
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]) -> nn.Module:
        """
        Execute PCB Merging algorithm

        Args:
            modelpool: Model pool containing pretrained and specialist models

        Returns:
            Merged model
        """
        if isinstance(modelpool, dict):
            modelpool = BaseModelPool(modelpool)

        # Get model names (excluding pretrained)
        task_model_names = [name for name in modelpool.model_names if not name.startswith('_')]
        n_models = len(task_model_names)

        if n_models < 2:
            raise ValueError(f"PCB Merging requires at least 2 task models, got {n_models}")

        log.info(f"🎯 PCB Merging Algorithm")
        log.info(f"Task models: {n_models} | PCB ratio: {self.pcb_ratio} | Scaling: {self.scaling_factor}")
        log.info(f"Models: {task_model_names}")

        # Load pretrained model
        with self.profile("load pretrained model"):
            pretrained_model = modelpool.load_model('_pretrained_')
            log.info("Using '_pretrained_' as base model")

        # Load all task models and compute task vectors
        task_vectors = []
        pretrained_state_dict = pretrained_model.state_dict()

        with self.profile("compute task vectors"):
            for model_name in tqdm(task_model_names, desc="Computing task vectors"):
                with self.profile(f"load model {model_name}"):
                    task_model = modelpool.load_model(model_name)
                    task_state_dict = task_model.state_dict()

                # Compute task vector: task_model - pretrained_model
                task_vector = {}
                for key in pretrained_state_dict:
                    if key in task_state_dict:
                        task_vector[key] = task_state_dict[key] - pretrained_state_dict[key]
                    else:
                        log.warning(f"Key {key} not found in {model_name}, using zero vector")
                        task_vector[key] = torch.zeros_like(pretrained_state_dict[key])

                task_vectors.append(task_vector)

        # Convert to flat tensors for PCB processing
        with self.profile("flatten parameters"):
            flat_task_vectors = []
            param_shapes = []
            param_names = []

            # Get parameter names and shapes from first task vector
            for name, param in task_vectors[0].items():
                param_names.append(name)
                param_shapes.append(param.shape)

            # Flatten all task vectors
            for task_vector in task_vectors:
                flat_params = []
                for name in param_names:
                    flat_params.append(task_vector[name].flatten())
                flat_task_vectors.append(torch.cat(flat_params))

            # Stack into matrix [n_models, n_params]
            flat_task_checks = torch.stack(flat_task_vectors, dim=0)

        # Apply PCB merging
        with self.profile("pcb merging"):
            merged_flat_tv = self.pcb_merging(flat_task_checks)

        # Reshape merged task vector back to original structure
        with self.profile("reshape parameters"):
            merged_task_vector = {}
            param_start = 0

            for i, name in enumerate(param_names):
                param_size = torch.prod(torch.tensor(param_shapes[i])).item()
                param_end = param_start + param_size

                flat_param = merged_flat_tv[param_start:param_end]
                merged_task_vector[name] = flat_param.reshape(param_shapes[i])

                param_start = param_end

        # Apply scaling and add to pretrained model
        with self.profile("apply scaling and merge"):
            merged_state_dict = {}
            for name in param_names:
                merged_state_dict[name] = (
                    pretrained_state_dict[name] +
                    self.scaling_factor * merged_task_vector[name]
                )

        # Load merged parameters into pretrained model
        result = pretrained_model.load_state_dict(merged_state_dict, strict=False)
        if result.unexpected_keys:
            log.warning(f"Unexpected keys in state dict: {result.unexpected_keys}")
        if result.missing_keys:
            log.warning(f"Missing keys in state dict: {result.missing_keys}")

        # Clean up memory
        del task_vectors, flat_task_vectors, flat_task_checks, merged_task_vector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Print statistics and profiling results
        self._print_statistics()
        self.print_profile_summary()

        log.info("PCB Merging completed successfully")
        return pretrained_model

    def _print_statistics(self):
        """Print PCB merging statistics"""
        log.info("=" * 60)
        log.info("📊 PCB Merging Statistics")
        log.info("=" * 60)
        log.info(f"Total parameters: {self.stats['total_parameters']:,}")
        log.info(f"Dropped parameters: {self.stats['dropped_parameters']:,}")
        if self.stats['total_parameters'] > 0:
            drop_rate = self.stats['dropped_parameters'] / self.stats['total_parameters'] * 100
            log.info(f"Drop rate: {drop_rate:.2f}%")
        log.info(f"PCB ratio: {self.pcb_ratio}")
        log.info(f"Scaling factor: {self.scaling_factor}")
        log.info("=" * 60)
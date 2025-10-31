#!/usr/bin/env python3
"""
Universal Model Merger V2 for FusionBench
新版本：先对留下的group平均（不计算0），然后再乘以alpha
"""

import logging
from typing import Dict, Literal, Union
import gc

import torch
from torch import nn
from tqdm import tqdm

from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool

log = logging.getLogger(__name__)

__all__ = ["UniversalMergerV2Algorithm"]


class UniversalMergerV2Algorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    Universal Model Merger V2 算法实现

    新版本特点：
    1. 先对留下的参数group平均（不计算被丢弃的0值）
    2. 然后再整体应用alpha系数
    3. 只支持Global ranking策略
    """

    def __init__(
        self,
        alpha: float = 0.5,
        importance_threshold: float = 0.3,
        seed: int = 42,
        **kwargs
    ):
        """
        初始化Universal Model Merger V2

        Args:
            alpha: 融合强度系数，适用于平均后的结果
            importance_threshold: 重要性阈值，用于决定冲突解决策略
            seed: 随机种子
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.importance_threshold = importance_threshold
        self.seed = seed

        # 统计信息
        self.global_stats = {
            'total_conflict_positions': 0,
            'threshold_decisions': 0,
            'random_decisions': 0,
            'per_model_stats': {}
        }

    def _compute_global_ranking(self, raw_importances: Dict[str, torch.Tensor], base_tensor: torch.Tensor, device: str):
        """Global Ranking: 整个tensor排序，使用 [1, n_elements]"""
        ranking_importances = {}

        for model_name in self.model_names:
            flat_imp = raw_importances[model_name].flatten()
            n_elements = flat_imp.numel()

            sorted_indices = torch.argsort(flat_imp)
            ranking_scores = torch.zeros_like(flat_imp, dtype=torch.float32)
            ranking_scores[sorted_indices] = torch.arange(
                1, n_elements + 1,
                dtype=torch.float32,
                device=device
            )

            ranking_importances[model_name] = ranking_scores.view_as(base_tensor)

        return ranking_importances

    def merge_single_tensor(self, base_tensor: torch.Tensor, specialist_tensors: Dict[str, torch.Tensor]):
        """合并单个tensor - V2版本"""
        if not torch.is_floating_point(base_tensor):
            return base_tensor

        original_dtype = base_tensor.dtype
        device = base_tensor.device
        total_elements = base_tensor.numel()

        # 步骤1: 计算task vectors
        task_vecs = {}
        raw_importances = {}
        signs = {}

        for model_name in self.model_names:
            delta = specialist_tensors[model_name] - base_tensor
            task_vecs[model_name] = delta
            raw_importances[model_name] = delta.abs()
            sign = torch.sign(delta)
            signs[model_name] = torch.where(sign == 0, torch.ones_like(sign), sign)

        # 步骤2: 检测冲突
        all_same = torch.ones_like(base_tensor, dtype=torch.bool)
        for i in range(self.n_models - 1):
            all_same = all_same & (signs[self.model_names[i]] == signs[self.model_names[i+1]])

        conflict_positions = ~all_same

        # 步骤3: 计算Global ranking
        ranking_importances = self._compute_global_ranking(raw_importances, base_tensor, device)

        # 步骤4: 初始化keep masks
        keep_masks = {model_name: torch.ones_like(base_tensor, dtype=torch.bool) for model_name in self.model_names}

        # 步骤5: 处理冲突
        if conflict_positions.any():
            n_conflict = conflict_positions.sum().item()
            self.global_stats['total_conflict_positions'] += n_conflict

            conflict_signs = torch.stack([
                signs[model_name][conflict_positions].flatten()
                for model_name in self.model_names
            ], dim=0)

            conflict_rankings = torch.stack([
                ranking_importances[model_name][conflict_positions].flatten()
                for model_name in self.model_names
            ], dim=0)

            is_positive = conflict_signs > 0
            is_negative = ~is_positive

            pos_ranking_sum = (conflict_rankings * is_positive.float()).sum(dim=0)
            neg_ranking_sum = (conflict_rankings * is_negative.float()).sum(dim=0)

            total_ranking = pos_ranking_sum + neg_ranking_sum
            ranking_diff_ratio = torch.abs(pos_ranking_sum - neg_ranking_sum) / (total_ranking + 1e-10)

            large_diff_mask = ranking_diff_ratio > self.importance_threshold
            small_diff_mask = ~large_diff_mask

            self.global_stats['threshold_decisions'] += large_diff_mask.sum().item()
            self.global_stats['random_decisions'] += small_diff_mask.sum().item()

            keep_decision = torch.zeros(self.n_models, n_conflict, dtype=torch.bool, device=device)

            if large_diff_mask.any():
                drop_pos = large_diff_mask & (pos_ranking_sum <= neg_ranking_sum)
                drop_neg = large_diff_mask & (pos_ranking_sum > neg_ranking_sum)
                keep_decision[:, drop_pos] = is_negative[:, drop_pos]
                keep_decision[:, drop_neg] = is_positive[:, drop_neg]

            if small_diff_mask.any():
                n_small = small_diff_mask.sum().item()
                random_choices = torch.rand(n_small, device=device) > 0.5

                random_drop_pos = torch.zeros(n_conflict, dtype=torch.bool, device=device)
                random_drop_pos[small_diff_mask] = ~random_choices
                random_drop_neg = torch.zeros(n_conflict, dtype=torch.bool, device=device)
                random_drop_neg[small_diff_mask] = random_choices

                keep_decision[:, random_drop_pos] = is_negative[:, random_drop_pos]
                keep_decision[:, random_drop_neg] = is_positive[:, random_drop_neg]

            for idx, model_name in enumerate(self.model_names):
                keep_masks[model_name][conflict_positions] = keep_decision[idx].view_as(
                    keep_masks[model_name][conflict_positions]
                )

        # 步骤6: Rescale
        rescale_factors = {}
        for model_name in self.model_names:
            keep_count = keep_masks[model_name].sum().item()
            keep_rate = keep_count / total_elements if total_elements > 0 else 1.0
            rescale_factors[model_name] = 1.0 / keep_rate if keep_rate > 0 else 1.0

            # 更新统计
            if model_name not in self.global_stats['per_model_stats']:
                self.global_stats['per_model_stats'][model_name] = {'dropped': 0, 'total': 0}

            dropped = total_elements - keep_count
            self.global_stats['per_model_stats'][model_name]['dropped'] += dropped
            self.global_stats['per_model_stats'][model_name]['total'] += total_elements

        # 步骤7: V2合并逻辑 - 先平均再乘alpha
        merged_delta = torch.zeros_like(base_tensor, dtype=torch.float32)
        total_contributions = torch.zeros_like(base_tensor, dtype=torch.float32)

        # 累加所有非零贡献
        for model_name in self.model_names:
            rescaled_task_vec = torch.where(
                keep_masks[model_name],
                task_vecs[model_name] * rescale_factors[model_name],
                torch.zeros_like(task_vecs[model_name])
            )
            merged_delta = merged_delta + rescaled_task_vec
            # 统计每个位置有多少模型贡献
            total_contributions = total_contributions + keep_masks[model_name].float()

        # 对非零位置求平均，避免除零
        total_contributions = torch.clamp(total_contributions, min=1.0)
        averaged_delta = merged_delta / total_contributions

        # 最后应用alpha
        final_delta = self.alpha * averaged_delta

        return (base_tensor + final_delta).to(original_dtype)

    @torch.no_grad()
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]) -> nn.Module:
        """执行Universal Model Merger V2算法"""
        if isinstance(modelpool, dict):
            modelpool = BaseModelPool(modelpool)

        # 设置随机种子
        torch.manual_seed(self.seed)

        # 获取模型名称
        self.model_names = [name for name in modelpool.model_names if not name.startswith('_')]
        self.n_models = len(self.model_names)

        if self.n_models < 2:
            raise ValueError(f"Universal Merger V2 requires at least 2 models, got {self.n_models}")

        # 初始化统计信息
        self.global_stats['per_model_stats'] = {
            model: {'dropped': 0, 'total': 0}
            for model in self.model_names
        }

        log.info(f"🚀 Universal Model Merger V2 (Global Ranking + Average-then-Scale)")
        log.info(f"Models: {self.n_models} | Threshold: {self.importance_threshold}")
        log.info(f"Model names: {self.model_names}")
        log.info(f"Alpha: {self.alpha}")

        # 加载基础模型
        with self.profile("load base model"):
            base_model = modelpool.load_model('_pretrained_')
            log.info("Using '_pretrained_' as base model")

        # 加载专家模型
        specialist_models = {}
        for model_name in self.model_names:
            with self.profile(f"load model {model_name}"):
                specialist_models[model_name] = modelpool.load_model(model_name)

        log.info("All models loaded successfully")

        # 合并模型参数
        base_state_dict = base_model.state_dict()
        specialist_state_dicts = {name: model.state_dict() for name, model in specialist_models.items()}

        merged_state_dict = {}

        with self.profile("merge parameters"):
            for param_name in tqdm(base_state_dict.keys(), desc="Merging parameters"):
                # 检查所有专家模型都有这个参数
                if all(param_name in specialist_state_dicts[name] for name in self.model_names):
                    specialist_tensors = {
                        name: specialist_state_dicts[name][param_name]
                        for name in self.model_names
                    }
                    merged_state_dict[param_name] = self.merge_single_tensor(
                        base_state_dict[param_name],
                        specialist_tensors
                    )
                else:
                    # 如果某些模型缺少这个参数，使用基础模型的参数
                    merged_state_dict[param_name] = base_state_dict[param_name]
                    log.warning(f"Parameter {param_name} not found in all models, using base model value")

        # 加载合并后的参数到基础模型
        base_model.load_state_dict(merged_state_dict, strict=False)

        # 清理内存
        del specialist_models, specialist_state_dicts, merged_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 打印统计信息
        self._print_statistics()

        # 打印性能报告
        self.print_profile_summary()

        log.info("Universal Model Merger V2 completed successfully")

        return base_model

    def _print_statistics(self):
        """打印统计信息"""
        log.info("="*60)
        log.info("📊 Universal Merger V2 Statistics")
        log.info("="*60)
        log.info(f"Strategy: Global Ranking + Average-then-Scale")
        log.info(f"Threshold: {self.importance_threshold}")
        log.info(f"Total conflicts: {self.global_stats['total_conflict_positions']:,}")
        log.info(f"Threshold decisions: {self.global_stats['threshold_decisions']:,}")
        log.info(f"Random decisions: {self.global_stats['random_decisions']:,}")

        log.info("\nPer-model drop rates:")
        for model in self.model_names:
            stats = self.global_stats['per_model_stats'][model]
            if stats['total'] > 0:
                drop_rate = stats['dropped'] / stats['total'] * 100
                log.info(f"  {model}: {drop_rate:.2f}% dropped ({stats['dropped']:,}/{stats['total']:,})")
        log.info("="*60)
#!/usr/bin/env python3
"""
测试Universal Model Merger在FusionBench中的集成效果
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, '/Users/fanzhiyuan1/workspace/fusion_bench')

from fusion_bench.method.universal_merger import UniversalMergerAlgorithm


def test_universal_merger():
    """测试Universal Merger算法类的基本功能"""

    print("🧪 Testing Universal Model Merger Integration")
    print("=" * 60)

    # 测试1: 创建算法实例
    print("\n1️⃣ Testing algorithm initialization...")

    try:
        # Global ranking strategy
        global_merger = UniversalMergerAlgorithm(
            importance_threshold=0.3,
            ranking_strategy="global",
            seed=42
        )
        print("✅ Global ranking algorithm created successfully")

        # Local ranking strategy
        local_merger = UniversalMergerAlgorithm(
            alphas={"model1": 0.4, "model2": 0.6},
            importance_threshold=0.25,
            ranking_strategy="local",
            seed=42
        )
        print("✅ Local ranking algorithm created successfully")

    except Exception as e:
        print(f"❌ Error creating algorithm: {e}")
        return False

    # 测试2: 检查配置参数
    print("\n2️⃣ Testing configuration parameters...")

    assert global_merger.importance_threshold == 0.3
    assert global_merger.ranking_strategy == "global"
    assert global_merger.seed == 42

    assert local_merger.alphas == {"model1": 0.4, "model2": 0.6}
    assert local_merger.ranking_strategy == "local"

    print("✅ Configuration parameters verified")

    # 测试3: 检查统计信息初始化
    print("\n3️⃣ Testing statistics initialization...")

    expected_stats = {
        'total_conflict_positions': 0,
        'threshold_decisions': 0,
        'random_decisions': 0,
        'per_model_stats': {}
    }

    assert global_merger.global_stats == expected_stats
    print("✅ Statistics initialization verified")

    print("\n🎉 All tests passed! Universal Merger is ready for benchmarking.")
    print("=" * 60)

    return True


def print_usage_examples():
    """打印使用示例"""

    print("\n📋 Usage Examples:")
    print("=" * 60)

    print("\n🔥 Basic Benchmarking Commands:")

    print("\n1. Global Ranking Strategy:")
    print("fusion_bench \\")
    print("    method=universal_merger_global \\")
    print("    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \\")
    print("    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8")

    print("\n2. Local Ranking Strategy:")
    print("fusion_bench \\")
    print("    method=universal_merger_local \\")
    print("    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \\")
    print("    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8")

    print("\n3. Custom Parameters:")
    print("fusion_bench \\")
    print("    method=universal_merger_global \\")
    print("    method.importance_threshold=0.4 \\")
    print("    method.seed=123 \\")
    print("    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \\")
    print("    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8")

    print("\n4. Small Scale Test (2-3 models):")
    print("fusion_bench \\")
    print("    method=universal_merger_global \\")
    print("    modelpool=CLIPVisionModelPool \\")
    print("    taskpool=CLIPVisionModelTaskPool \\")
    print("    modelpool.models._pretrained_=openai/clip-vit-base-patch32 \\")
    print("    modelpool.models.sun397=tanganke/clip-vit-base-patch32_sun397 \\")
    print("    modelpool.models.cars=tanganke/clip-vit-base-patch32_stanford-cars")

    print("\n💡 Available Configuration Files:")
    print("- universal_merger_global.yaml   (Global ranking)")
    print("- universal_merger_local.yaml    (Local ranking)")
    print("- universal_merger_custom.yaml   (Custom weights example)")

    print("\n🔧 Key Parameters:")
    print("- importance_threshold: 控制冲突解决策略的阈值 (0.0-1.0)")
    print("- ranking_strategy: 'global' 或 'local' 排名策略")
    print("- alphas: 各模型的融合权重字典")
    print("- seed: 随机种子，确保结果可重现")

    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    success = test_universal_merger()

    # 打印使用示例
    if success:
        print_usage_examples()

    sys.exit(0 if success else 1)
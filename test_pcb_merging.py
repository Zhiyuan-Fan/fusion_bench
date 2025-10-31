#!/usr/bin/env python3
"""
测试PCB Merging在FusionBench中的集成效果
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, '/Users/fanzhiyuan1/workspace/fusion_bench')

from fusion_bench.method.pcb_merging import PCBMergingAlgorithm


def test_pcb_merging():
    """测试PCB Merging算法类的基本功能"""

    print("🧪 Testing PCB Merging Integration")
    print("=" * 60)

    # 测试1: 创建算法实例
    print("\n1️⃣ Testing algorithm initialization...")

    try:
        # Default configuration
        pcb_default = PCBMergingAlgorithm(
            scaling_factor=0.3,
            pcb_ratio=0.1
        )
        print("✅ Default PCB algorithm created successfully")

        # Aggressive configuration
        pcb_aggressive = PCBMergingAlgorithm(
            scaling_factor=0.5,
            pcb_ratio=0.2,
            min_ratio=0.0001,
            max_ratio=0.0001
        )
        print("✅ Aggressive PCB algorithm created successfully")

        # Conservative configuration
        pcb_conservative = PCBMergingAlgorithm(
            scaling_factor=0.2,
            pcb_ratio=0.05
        )
        print("✅ Conservative PCB algorithm created successfully")

    except Exception as e:
        print(f"❌ Error creating algorithm: {e}")
        return False

    # 测试2: 检查配置参数
    print("\n2️⃣ Testing configuration parameters...")

    assert pcb_default.scaling_factor == 0.3
    assert pcb_default.pcb_ratio == 0.1
    assert pcb_default.min_ratio == 0.0001
    assert pcb_default.max_ratio == 0.0001

    assert pcb_aggressive.pcb_ratio == 0.2
    assert pcb_conservative.pcb_ratio == 0.05

    print("✅ Configuration parameters verified")

    # 测试3: 检查统计信息初始化
    print("\n3️⃣ Testing statistics initialization...")

    expected_stats = {
        'total_parameters': 0,
        'dropped_parameters': 0,
        'competition_stats': {}
    }

    assert pcb_default.stats == expected_stats
    print("✅ Statistics initialization verified")

    # 测试4: 测试核心函数存在性
    print("\n4️⃣ Testing core function availability...")

    assert hasattr(pcb_default, 'pcb_merging')
    assert hasattr(pcb_default, 'run')
    assert callable(pcb_default.pcb_merging)
    assert callable(pcb_default.run)

    print("✅ Core functions available")

    print("\n🎉 All tests passed! PCB Merging is ready for benchmarking.")
    print("=" * 60)

    return True


def print_usage_examples():
    """打印使用示例"""

    print("\n📋 PCB Merging Usage Examples:")
    print("=" * 60)

    print("\n🔥 Basic Benchmarking Commands:")

    print("\n1. Default PCB Merging:")
    print("fusion_bench \\")
    print("    method=pcb_merging \\")
    print("    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \\")
    print("    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8")

    print("\n2. Aggressive PCB Merging (more parameter dropping):")
    print("fusion_bench \\")
    print("    method=pcb_merging_aggressive \\")
    print("    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \\")
    print("    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8")

    print("\n3. Conservative PCB Merging (less parameter dropping):")
    print("fusion_bench \\")
    print("    method=pcb_merging_conservative \\")
    print("    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \\")
    print("    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8")

    print("\n4. Custom Parameters:")
    print("fusion_bench \\")
    print("    method=pcb_merging \\")
    print("    method.scaling_factor=0.4 \\")
    print("    method.pcb_ratio=0.15 \\")
    print("    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \\")
    print("    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8")

    print("\n5. Small Scale Test:")
    print("fusion_bench \\")
    print("    method=pcb_merging \\")
    print("    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_svhn_and_mnist \\")
    print("    taskpool=CLIPVisionModelTaskPool/clip-vit-base-patch32_svhn_and_mnist")

    print("\n💡 Available Configuration Files:")
    print("- pcb_merging.yaml           (Default: scaling=0.3, pcb_ratio=0.1)")
    print("- pcb_merging_aggressive.yaml (Aggressive: scaling=0.5, pcb_ratio=0.2)")
    print("- pcb_merging_conservative.yaml (Conservative: scaling=0.2, pcb_ratio=0.05)")

    print("\n🔧 Key Parameters:")
    print("- scaling_factor: Final scaling for merged task vector (0.1-0.7)")
    print("- pcb_ratio: Ratio of competitive parameters to drop (0.05-0.3)")
    print("- min_ratio/max_ratio: Outlier clamping percentiles (advanced)")

    print("\n🆚 Comparison with Other Methods:")
    print("- vs Task Arithmetic: Handles parameter competition explicitly")
    print("- vs TIES: Uses competition balancing instead of magnitude-based trimming")
    print("- vs Universal Merger: Focuses on competition rather than ranking")

    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    success = test_pcb_merging()

    # 打印使用示例
    if success:
        print_usage_examples()

    sys.exit(0 if success else 1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGCNN k值实验配置文件
"""

# 实验配置
EXPERIMENT_CONFIG = {
    # 基础配置
    "model_name": "CTRLightGCNCTR",
    "dataset": "your_dataset",  # 替换为你的数据集名称
    
    # 训练配置
    "epochs": 50,
    "batch_size": 256,
    "learning_rate": 0.001,
    "device": "cuda",  # 或 "cpu"
    
    # 其他超参数
    "emb_size": 64,
    "n_layers": 3,
    "hidden_channels": 32,
    "out_channels": 16,
}

# 要测试的k值列表
K_VALUES = {
    # 小k值组（稀疏连接）
    "small_k": [4, 6, 8],
    
    # 中等k值组（平衡连接）
    "medium_k": [10, 12, 15],
    
    # 大k值组（密集连接）
    "large_k": [20, 25, 30],
    
    # 特殊k值
    "special_k": [1, 62],  # 最小连接和全连接
    
    # 所有k值
    "all_k": [4, 6, 8, 10, 12, 15, 20, 25, 30],
    
    # 快速测试k值（用于快速验证）
    "quick_test": [4, 8, 15, 25],
}

# 实验场景配置
EXPERIMENT_SCENARIOS = {
    "quick_validation": {
        "description": "快速验证不同k值的基本功能",
        "k_values": K_VALUES["quick_test"],
        "epochs": 10,
        "batch_size": 128,
    },
    
    "performance_comparison": {
        "description": "比较不同k值的性能表现",
        "k_values": K_VALUES["all_k"],
        "epochs": 50,
        "batch_size": 256,
    },
    
    "sparse_vs_dense": {
        "description": "比较稀疏连接和密集连接的效果",
        "k_values": [4, 8, 15, 25, 30],
        "epochs": 50,
        "batch_size": 256,
    },
    
    "efficiency_analysis": {
        "description": "分析计算效率与k值的关系",
        "k_values": K_VALUES["all_k"],
        "epochs": 20,
        "batch_size": 512,
    }
}

# 预期结果分析
EXPECTED_TRENDS = {
    "small_k": {
        "advantages": [
            "计算效率高",
            "内存占用少",
            "训练速度快",
            "适合实时应用"
        ],
        "disadvantages": [
            "可能信息不足",
            "特征聚合有限",
            "对复杂模式捕捉能力弱"
        ],
        "best_for": "计算资源受限的场景"
    },
    
    "medium_k": {
        "advantages": [
            "平衡性能和效率",
            "适中的信息聚合",
            "稳定的训练过程"
        ],
        "disadvantages": [
            "需要调参找到最佳值",
            "可能不是最优选择"
        ],
        "best_for": "一般应用场景"
    },
    
    "large_k": {
        "advantages": [
            "信息聚合丰富",
            "捕捉复杂模式能力强",
            "特征表示更完整"
        ],
        "disadvantages": [
            "计算开销大",
            "内存占用高",
            "训练时间长",
            "可能过拟合"
        ],
        "best_for": "对精度要求高的场景"
    }
}

# 评估指标
EVALUATION_METRICS = [
    "AUC",
    "NDCG@10", 
    "NDCG@20",
    "Recall@10",
    "Recall@20",
    "Precision@10",
    "Precision@20"
]

# 实验命名规则
def get_experiment_name(k_value, scenario="default"):
    """生成实验名称"""
    return f"DGCNN_k{k_value}_{scenario}"

def get_result_filename(k_value, scenario="default"):
    """生成结果文件名"""
    timestamp = "YYYYMMDD_HHMMSS"  # 实际使用时替换为真实时间戳
    return f"results_k{k_value}_{scenario}_{timestamp}.json"

# 命令行参数模板
def get_command_template(k_value, scenario_config):
    """生成命令行参数模板"""
    base_cmd = [
        "python", "main.py",
        "--model", EXPERIMENT_CONFIG["model_name"],
        "--dataset", EXPERIMENT_CONFIG["dataset"],
        "--dgcnn_k", str(k_value),
        "--epoch", str(scenario_config.get("epochs", EXPERIMENT_CONFIG["epochs"])),
        "--batch_size", str(scenario_config.get("batch_size", EXPERIMENT_CONFIG["batch_size"])),
        "--lr", str(EXPERIMENT_CONFIG["learning_rate"]),
        "--device", EXPERIMENT_CONFIG["device"],
        "--emb_size", str(EXPERIMENT_CONFIG["emb_size"]),
        "--n_layers", str(EXPERIMENT_CONFIG["n_layers"])
    ]
    return base_cmd

# 结果分析模板
ANALYSIS_TEMPLATE = """
# DGCNN k值实验分析报告

## 实验设置
- 模型: {model_name}
- 数据集: {dataset}
- 测试k值: {k_values}
- 训练轮数: {epochs}

## 主要发现
1. **最佳k值**: {best_k} (AUC: {best_auc:.4f})
2. **性能趋势**: {trend_description}
3. **计算效率**: {efficiency_analysis}

## 详细结果
{detailed_results}

## 建议
{recommendations}
"""

if __name__ == "__main__":
    print("DGCNN k值实验配置")
    print("=" * 50)
    
    print("可用的实验场景:")
    for name, config in EXPERIMENT_SCENARIOS.items():
        print(f"- {name}: {config['description']}")
        print(f"  k值: {config['k_values']}")
        print(f"  轮数: {config['epochs']}")
        print()
    
    print("预期趋势:")
    for k_type, trends in EXPECTED_TRENDS.items():
        print(f"- {k_type}:")
        print(f"  优势: {', '.join(trends['advantages'])}")
        print(f"  劣势: {', '.join(trends['disadvantages'])}")
        print(f"  适用场景: {trends['best_for']}")
        print()

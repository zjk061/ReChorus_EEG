#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本：分析为什么不同k值会产生相同结果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.general.CTRLightGCN import EEGDGCNNEncoder, DynamicalGraphConv

def analyze_knn_graph(x, k_values=[4, 6, 8, 10, 12, 30]):
    """
    分析不同k值下的KNN图结构
    """
    print("=" * 60)
    print("KNN图结构分析")
    print("=" * 60)
    
    # 创建测试数据
    batch_size, num_nodes, channels = x.shape
    print(f"输入数据形状: {x.shape}")
    print(f"节点数量: {num_nodes}")
    print(f"特征维度: {channels}")
    
    # 分析特征分布
    print(f"\n特征统计:")
    print(f"  均值: {x.mean().item():.4f}")
    print(f"  标准差: {x.std().item():.4f}")
    print(f"  最小值: {x.min().item():.4f}")
    print(f"  最大值: {x.max().item():.4f}")
    
    # 检查特征是否过于相似
    x_reshaped = x.view(batch_size * num_nodes, channels)
    distances = torch.cdist(x_reshaped, x_reshaped)
    
    print(f"\n特征相似性分析:")
    print(f"  平均距离: {distances.mean().item():.4f}")
    print(f"  距离标准差: {distances.std().item():.4f}")
    print(f"  最小距离: {distances.min().item():.4f}")
    print(f"  最大距离: {distances.max().item():.4f}")
    
    # 分析不同k值的邻居选择
    for k in k_values:
        print(f"\n--- k={k} ---")
        
        # 创建DGCNN层
        dgcnn = DynamicalGraphConv(channels, 32, k=k)
        
        # 获取KNN图
        x_t = x.transpose(2, 1).contiguous()  # [batch, channels, num_nodes]
        idx = dgcnn.knn_graph(x_t, k)
        
        # 分析邻居分布
        unique_neighbors = set()
        for b in range(batch_size):
            for n in range(num_nodes):
                neighbors = idx[b, n].tolist()
                unique_neighbors.update(neighbors)
        
        print(f"  唯一邻居节点数: {len(unique_neighbors)}")
        print(f"  邻居覆盖率: {len(unique_neighbors)/num_nodes:.2%}")
        
        # 检查是否有重复邻居
        neighbor_counts = {}
        for b in range(batch_size):
            for n in range(num_nodes):
                for neighbor in idx[b, n]:
                    neighbor_counts[neighbor.item()] = neighbor_counts.get(neighbor.item(), 0) + 1
        
        max_count = max(neighbor_counts.values()) if neighbor_counts else 0
        print(f"  最大被选择次数: {max_count}")
        
        # 分析邻居选择的多样性
        if k < num_nodes:
            expected_unique = min(k * num_nodes, num_nodes * (num_nodes - 1))
            actual_unique = len(unique_neighbors)
            print(f"  邻居多样性: {actual_unique}/{expected_unique} ({actual_unique/expected_unique:.2%})")

def test_dgcnn_outputs(x, k_values=[4, 6, 8, 10, 12, 30]):
    """
    测试不同k值下的DGCNN输出
    """
    print("\n" + "=" * 60)
    print("DGCNN输出分析")
    print("=" * 60)
    
    batch_size, num_nodes, channels = x.shape
    
    outputs = {}
    for k in k_values:
        print(f"\n--- k={k} ---")
        
        # 创建DGCNN编码器
        encoder = EEGDGCNNEncoder(in_channels=channels, k=k)
        
        # 前向传播
        with torch.no_grad():
            output = encoder(x)
        
        outputs[k] = output
        
        print(f"  输出形状: {output.shape}")
        print(f"  输出均值: {output.mean().item():.6f}")
        print(f"  输出标准差: {output.std().item():.6f}")
        print(f"  输出范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
    
    # 比较不同k值的输出
    print(f"\n--- 输出比较 ---")
    k_list = list(k_values)
    for i in range(len(k_list)):
        for j in range(i+1, len(k_list)):
            k1, k2 = k_list[i], k_list[j]
            diff = torch.abs(outputs[k1] - outputs[k2])
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"  k={k1} vs k={k2}: 最大差异={max_diff:.8f}, 平均差异={mean_diff:.8f}")
            
            if max_diff < 1e-6:
                print(f"    ⚠️  输出几乎完全相同！")

def analyze_feature_similarity(x):
    """
    分析特征相似性
    """
    print("\n" + "=" * 60)
    print("特征相似性详细分析")
    print("=" * 60)
    
    batch_size, num_nodes, channels = x.shape
    
    # 计算所有节点对之间的相似性
    x_reshaped = x.view(batch_size * num_nodes, channels)
    
    # 计算余弦相似性
    x_norm = torch.nn.functional.normalize(x_reshaped, p=2, dim=1)
    cosine_sim = torch.mm(x_norm, x_norm.t())
    
    # 计算欧几里得距离
    distances = torch.cdist(x_reshaped, x_reshaped)
    
    print(f"余弦相似性统计:")
    print(f"  均值: {cosine_sim.mean().item():.4f}")
    print(f"  标准差: {cosine_sim.std().item():.4f}")
    print(f"  最小值: {cosine_sim.min().item():.4f}")
    print(f"  最大值: {cosine_sim.max().item():.4f}")
    
    print(f"\n欧几里得距离统计:")
    print(f"  均值: {distances.mean().item():.4f}")
    print(f"  标准差: {distances.std().item():.4f}")
    print(f"  最小值: {distances.min().item():.4f}")
    print(f"  最大值: {distances.max().item():.4f}")
    
    # 检查是否有太多相似的特征
    high_sim_count = (cosine_sim > 0.95).sum().item()
    low_dist_count = (distances < 0.1).sum().item()
    
    total_pairs = cosine_sim.numel()
    print(f"\n高相似性分析:")
    print(f"  相似性>0.95的节点对: {high_sim_count}/{total_pairs} ({high_sim_count/total_pairs:.2%})")
    print(f"  距离<0.1的节点对: {low_dist_count}/{total_pairs} ({low_dist_count/total_pairs:.2%})")
    
    if high_sim_count / total_pairs > 0.5:
        print("  ⚠️  警告：大量节点特征过于相似！")
        print("  这可能导致不同k值选择相似的邻居，从而产生相同的结果。")

def create_test_data():
    """
    创建不同特征的测试数据
    """
    print("创建测试数据...")
    
    # 方案1：随机数据
    random_data = torch.randn(4, 62, 5)
    
    # 方案2：高度相似的数据（模拟你的情况）
    base_feature = torch.randn(1, 1, 5)
    similar_data = base_feature.repeat(4, 62, 1) + torch.randn(4, 62, 5) * 0.01
    
    # 方案3：完全不同的数据
    diverse_data = torch.randn(4, 62, 5) * 10
    
    return {
        "random": random_data,
        "similar": similar_data,
        "diverse": diverse_data
    }

def main():
    """主函数"""
    print("DGCNN k值调试分析")
    print("=" * 60)
    
    # 创建测试数据
    test_data = create_test_data()
    
    k_values = [4, 6, 8, 10, 12, 30]
    
    for data_name, x in test_data.items():
        print(f"\n{'='*20} {data_name.upper()} 数据 {'='*20}")
        
        # 分析KNN图结构
        analyze_knn_graph(x, k_values)
        
        # 测试DGCNN输出
        test_dgcnn_outputs(x, k_values)
        
        # 分析特征相似性
        analyze_feature_similarity(x)
    
    print(f"\n{'='*60}")
    print("分析结论和建议")
    print("="*60)
    print("""
可能的原因：
1. 数据特征过于相似 - 如果所有节点的特征几乎相同，不同k值会选择相似的邻居
2. 全局池化掩盖了差异 - AdaptiveAvgPool1d可能平均化了不同k值的影响
3. 模型初始化问题 - 权重初始化可能导致输出相似
4. 数据预处理问题 - 特征标准化可能使特征过于相似

建议解决方案：
1. 检查你的EEG数据特征分布
2. 尝试不同的特征预处理方法
3. 在DGCNN层之间添加更多非线性变换
4. 使用不同的池化方法（如max pooling）
5. 增加模型的表达能力
    """)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：分析为什么不同k值会产生相同结果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.general.CTRLightGCN import EEGDGCNNEncoder, DynamicalGraphConv

def diagnose_data_characteristics(x):
    """诊断数据特征"""
    print("=" * 60)
    print("数据特征诊断")
    print("=" * 60)
    
    batch_size, num_nodes, channels = x.shape
    print(f"数据形状: {x.shape}")
    
    # 基本统计
    print(f"\n基本统计:")
    print(f"  均值: {x.mean().item():.6f}")
    print(f"  标准差: {x.std().item():.6f}")
    print(f"  最小值: {x.min().item():.6f}")
    print(f"  最大值: {x.max().item():.6f}")
    
    # 检查数据是否过于相似
    x_reshaped = x.view(batch_size * num_nodes, channels)
    
    # 计算所有节点对之间的欧几里得距离
    distances = torch.cdist(x_reshaped, x_reshaped)
    
    print(f"\n距离分析:")
    print(f"  平均距离: {distances.mean().item():.6f}")
    print(f"  距离标准差: {distances.std().item():.6f}")
    print(f"  最小距离: {distances.min().item():.6f}")
    print(f"  最大距离: {distances.max().item():.6f}")
    
    # 检查有多少节点对非常相似
    very_similar = (distances < 0.01).sum().item()
    similar = (distances < 0.1).sum().item()
    total_pairs = distances.numel()
    
    print(f"\n相似性分析:")
    print(f"  距离<0.01的节点对: {very_similar}/{total_pairs} ({very_similar/total_pairs:.2%})")
    print(f"  距离<0.1的节点对: {similar}/{total_pairs} ({similar/total_pairs:.2%})")
    
    if very_similar / total_pairs > 0.1:
        print("  ⚠️  警告：大量节点特征过于相似！")
        return True
    
    return False

def analyze_knn_behavior(x, k_values=[4, 6, 8, 10, 12, 30]):
    """分析KNN行为"""
    print("\n" + "=" * 60)
    print("KNN行为分析")
    print("=" * 60)
    
    batch_size, num_nodes, channels = x.shape
    x_t = x.transpose(2, 1).contiguous()  # [batch, channels, num_nodes]
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        
        # 创建DGCNN层
        dgcnn = DynamicalGraphConv(channels, 32, k=k)
        
        # 获取KNN图
        idx = dgcnn.knn_graph(x_t, k)
        
        # 分析邻居选择
        unique_neighbors = set()
        neighbor_counts = {}
        
        for b in range(batch_size):
            for n in range(num_nodes):
                neighbors = idx[b, n].tolist()
                unique_neighbors.update(neighbors)
                
                for neighbor in neighbors:
                    neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1
        
        print(f"  唯一邻居节点数: {len(unique_neighbors)}")
        print(f"  邻居覆盖率: {len(unique_neighbors)/num_nodes:.2%}")
        
        if neighbor_counts:
            max_count = max(neighbor_counts.values())
            min_count = min(neighbor_counts.values())
            print(f"  被选择次数范围: [{min_count}, {max_count}]")
            
            # 检查是否有节点被过度选择
            over_selected = sum(1 for count in neighbor_counts.values() if count > k * 2)
            print(f"  过度选择的节点数: {over_selected}")
        
        # 检查邻居选择的多样性
        if k < num_nodes:
            expected_unique = min(k * num_nodes, num_nodes * (num_nodes - 1))
            actual_unique = len(unique_neighbors)
            diversity_ratio = actual_unique / expected_unique
            print(f"  邻居多样性: {actual_unique}/{expected_unique} ({diversity_ratio:.2%})")
            
            if diversity_ratio < 0.5:
                print("    ⚠️  邻居选择多样性较低！")

def test_model_outputs(x, k_values=[4, 6, 8, 10, 12, 30]):
    """测试模型输出"""
    print("\n" + "=" * 60)
    print("模型输出测试")
    print("=" * 60)
    
    batch_size, num_nodes, channels = x.shape
    outputs = {}
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        
        # 创建编码器
        encoder = EEGDGCNNEncoder(in_channels=channels, k=k)
        
        # 前向传播
        with torch.no_grad():
            output = encoder(x)
        
        outputs[k] = output
        
        print(f"  输出形状: {output.shape}")
        print(f"  输出均值: {output.mean().item():.8f}")
        print(f"  输出标准差: {output.std().item():.8f}")
        print(f"  输出范围: [{output.min().item():.8f}, {output.max().item():.8f}]")
    
    # 比较输出
    print(f"\n--- 输出比较 ---")
    k_list = list(k_values)
    identical_count = 0
    
    for i in range(len(k_list)):
        for j in range(i+1, len(k_list)):
            k1, k2 = k_list[i], k_list[j]
            diff = torch.abs(outputs[k1] - outputs[k2])
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"  k={k1} vs k={k2}: 最大差异={max_diff:.10f}, 平均差异={mean_diff:.10f}")
            
            if max_diff < 1e-8:
                print(f"    ⚠️  输出完全相同！")
                identical_count += 1
            elif max_diff < 1e-6:
                print(f"    ⚠️  输出几乎相同！")
            else:
                print(f"    ✅  输出有差异")
    
    print(f"\n总结: {identical_count}/{len(k_list)*(len(k_list)-1)//2} 对输出完全相同")

def analyze_pooling_effect(x, k=8):
    """分析池化效果"""
    print("\n" + "=" * 60)
    print("池化效果分析")
    print("=" * 60)
    
    batch_size, num_nodes, channels = x.shape
    x_t = x.transpose(2, 1).contiguous()
    
    # 创建DGCNN层
    dgcnn = DynamicalGraphConv(channels, 32, k=k)
    
    # 获取中间特征
    with torch.no_grad():
        # 获取图特征
        feature = dgcnn.get_graph_feature(x_t, k)
        
        # 应用卷积
        conv_out = dgcnn.conv(feature)
        bn_out = dgcnn.bn(conv_out)
        relu_out = F.relu(bn_out)
        
        # 池化前
        before_pool = relu_out  # [batch, out_channels, num_nodes, k]
        
        # 不同池化方法
        max_pool = before_pool.max(dim=-1, keepdim=False)[0]  # [batch, out_channels, num_nodes]
        avg_pool = before_pool.mean(dim=-1, keepdim=False)    # [batch, out_channels, num_nodes]
        
        print(f"池化前特征形状: {before_pool.shape}")
        print(f"最大池化后形状: {max_pool.shape}")
        print(f"平均池化后形状: {avg_pool.shape}")
        
        print(f"\n池化前统计:")
        print(f"  均值: {before_pool.mean().item():.6f}")
        print(f"  标准差: {before_pool.std().item():.6f}")
        print(f"  范围: [{before_pool.min().item():.6f}, {before_pool.max().item():.6f}]")
        
        print(f"\n最大池化统计:")
        print(f"  均值: {max_pool.mean().item():.6f}")
        print(f"  标准差: {max_pool.std().item():.6f}")
        print(f"  范围: [{max_pool.min().item():.6f}, {max_pool.max().item():.6f}]")
        
        print(f"\n平均池化统计:")
        print(f"  均值: {avg_pool.mean().item():.6f}")
        print(f"  标准差: {avg_pool.std().item():.6f}")
        print(f"  范围: [{avg_pool.min().item():.6f}, {avg_pool.max().item():.6f}]")
        
        # 比较池化方法
        pool_diff = torch.abs(max_pool - avg_pool)
        print(f"\n池化方法差异:")
        print(f"  最大差异: {pool_diff.max().item():.6f}")
        print(f"  平均差异: {pool_diff.mean().item():.6f}")

def create_test_scenarios():
    """创建测试场景"""
    print("创建测试场景...")
    
    batch_size, num_nodes, channels = 4, 62, 5
    
    scenarios = {}
    
    # 场景1：完全相同的特征
    scenarios["identical"] = torch.ones(batch_size, num_nodes, channels) * 0.5
    
    # 场景2：高度相似的特征
    base = torch.randn(1, 1, channels)
    scenarios["similar"] = base.repeat(batch_size, num_nodes, 1) + torch.randn(batch_size, num_nodes, channels) * 0.01
    
    # 场景3：随机特征
    scenarios["random"] = torch.randn(batch_size, num_nodes, channels)
    
    # 场景4：结构化特征
    structured = torch.zeros(batch_size, num_nodes, channels)
    for b in range(batch_size):
        for n in range(num_nodes):
            structured[b, n, 0] = np.sin(n * 0.1) + np.random.normal(0, 0.1)
            structured[b, n, 1] = np.cos(n * 0.1) + np.random.normal(0, 0.1)
            structured[b, n, 2] = n / num_nodes + np.random.normal(0, 0.1)
            structured[b, n, 3] = (n % 10) / 10 + np.random.normal(0, 0.1)
            structured[b, n, 4] = np.random.normal(0, 1)
    scenarios["structured"] = structured
    
    return scenarios

def main():
    """主函数"""
    print("DGCNN k值问题诊断")
    print("=" * 60)
    
    # 创建测试场景
    scenarios = create_test_scenarios()
    k_values = [4, 6, 8, 10, 12, 30]
    
    for scenario_name, x in scenarios.items():
        print(f"\n{'='*20} {scenario_name.upper()} 场景 {'='*20}")
        
        # 诊断数据特征
        is_similar = diagnose_data_characteristics(x)
        
        # 分析KNN行为
        analyze_knn_behavior(x, k_values)
        
        # 测试模型输出
        test_model_outputs(x, k_values)
        
        # 分析池化效果
        analyze_pooling_effect(x)
        
        print(f"\n场景 {scenario_name} 总结:")
        if is_similar:
            print("  - 数据特征过于相似，这可能是导致不同k值产生相同结果的主要原因")
        else:
            print("  - 数据特征具有足够的多样性")
    
    print(f"\n{'='*60}")
    print("诊断结论和建议")
    print("=" * 60)
    print("""
可能的原因：
1. 数据特征过于相似 - 如果所有节点的特征几乎相同，KNN会选择相同的邻居
2. 全局池化掩盖差异 - AdaptiveAvgPool1d可能平均化了不同k值的影响
3. 模型初始化问题 - 权重初始化可能导致输出相似
4. 特征预处理问题 - 标准化可能使特征过于相似

建议解决方案：
1. 检查你的EEG数据预处理 - 确保特征有足够的多样性
2. 尝试不同的特征提取方法 - 使用更丰富的特征表示
3. 修改池化策略 - 使用最大池化或注意力池化
4. 增加模型复杂度 - 添加更多非线性变换
5. 使用改进的DGCNN实现 - 包含注意力机制和残差连接

请运行这个诊断脚本来确定具体的问题所在。
    """)

if __name__ == "__main__":
    main()

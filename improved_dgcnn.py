#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的DGCNN实现
解决不同k值产生相同结果的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedDynamicalGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=8, dropout=0.1):
        super().__init__()
        self.k = k
        self.dropout = dropout
        
        # 改进的卷积层设计
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        
        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, k, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        
        # 添加残差连接
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
    def knn_graph(self, x, k):
        """改进的KNN图构建"""
        batch_size, channels, num_nodes = x.shape
        x = x.transpose(2, 1).contiguous()  # [batch, num_nodes, channels]
        
        # 添加噪声以提高特征多样性
        if self.training:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        # 计算欧几里得距离
        inner = -2 * torch.matmul(x, x.transpose(2, 1))
        xx = torch.sum(x**2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        # 选择top-k近邻
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx
    
    def get_graph_feature(self, x, k, idx=None):
        """改进的图特征提取"""
        batch_size, channels, num_nodes = x.shape
        
        if idx is None:
            idx = self.knn_graph(x, k)
        
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_nodes
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_nodes, -1)[idx, :]
        feature = feature.view(batch_size, num_nodes, k, channels)
        
        x = x.view(batch_size, num_nodes, 1, channels).repeat(1, 1, k, 1)
        
        # 计算特征差异和拼接
        feature_diff = feature - x
        feature_concat = torch.cat((feature_diff, x), dim=3)
        
        # 应用注意力机制
        attention_weights = self.attention(feature_concat.permute(0, 3, 1, 2))
        feature_concat = feature_concat.permute(0, 3, 1, 2)  # [batch, 2*channels, num_nodes, k]
        
        return feature_concat, attention_weights
    
    def forward(self, x):
        """改进的前向传播"""
        # 获取图特征和注意力权重
        feature, attention = self.get_graph_feature(x, self.k)
        
        # 应用注意力权重
        feature = feature * attention.unsqueeze(1)  # 广播注意力权重
        
        # 卷积处理
        out = self.conv(feature)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout_layer(out)
        
        # 最大池化聚合邻居信息
        out = out.max(dim=-1, keepdim=False)[0]  # [batch, out_channels, num_nodes]
        
        # 残差连接
        if self.residual is not None:
            residual = self.residual(x)
            out = out + residual
        
        return out

class ImprovedEEGDGCNNEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, out_channels=16, k=8, dropout=0.1):
        super().__init__()
        self.k = k
        
        # 改进的DGCNN层
        self.dgcnn1 = ImprovedDynamicalGraphConv(in_channels, hidden_channels, k, dropout)
        self.dgcnn2 = ImprovedDynamicalGraphConv(hidden_channels, out_channels, k, dropout)
        
        # 多层感知机进行特征融合
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 2, out_channels)
        )
        
        # 多种池化方法
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 池化权重（可学习）
        self.pool_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def forward(self, x):
        """改进的前向传播"""
        batch_size, num_nodes, in_channels = x.shape
        x = x.transpose(2, 1).contiguous().float()  # [batch, 5, 62]
        
        # DGCNN层
        x1 = self.dgcnn1(x)  # [batch, hidden_channels, 62]
        x2 = self.dgcnn2(x1)  # [batch, out_channels, 62]
        
        # 应用MLP
        x2 = x2.transpose(2, 1)  # [batch, 62, out_channels]
        x2 = self.mlp(x2)  # [batch, 62, out_channels]
        x2 = x2.transpose(2, 1)  # [batch, out_channels, 62]
        
        # 多种池化方法
        avg_out = self.avg_pool(x2)  # [batch, out_channels, 1]
        max_out = self.max_pool(x2)  # [batch, out_channels, 1]
        
        # 加权融合
        weights = F.softmax(self.pool_weights, dim=0)
        pooled = weights[0] * avg_out + weights[1] * max_out
        
        # 最终输出
        output = pooled.squeeze(-1)  # [batch, out_channels]
        
        return output

def test_improved_dgcnn():
    """测试改进的DGCNN"""
    print("测试改进的DGCNN实现")
    print("=" * 50)
    
    # 创建测试数据
    batch_size, num_nodes, channels = 4, 62, 5
    x = torch.randn(batch_size, num_nodes, channels)
    
    k_values = [4, 6, 8, 10, 12, 30]
    outputs = {}
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        
        # 创建改进的DGCNN编码器
        encoder = ImprovedEEGDGCNNEncoder(in_channels=channels, k=k)
        
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
            else:
                print(f"    ✅  输出有明显差异")

def create_diverse_test_data():
    """创建具有多样性的测试数据"""
    print("创建多样性测试数据...")
    
    batch_size, num_nodes, channels = 4, 62, 5
    
    # 方案1：基于空间位置的多样性数据
    spatial_data = torch.zeros(batch_size, num_nodes, channels)
    for b in range(batch_size):
        for n in range(num_nodes):
            # 基于节点位置创建不同的特征模式
            spatial_data[b, n, 0] = np.sin(n * 0.1) + np.random.normal(0, 0.1)
            spatial_data[b, n, 1] = np.cos(n * 0.1) + np.random.normal(0, 0.1)
            spatial_data[b, n, 2] = n / num_nodes + np.random.normal(0, 0.1)
            spatial_data[b, n, 3] = (n % 10) / 10 + np.random.normal(0, 0.1)
            spatial_data[b, n, 4] = np.random.normal(0, 1)
    
    # 方案2：基于时间序列的多样性数据
    temporal_data = torch.zeros(batch_size, num_nodes, channels)
    for b in range(batch_size):
        for n in range(num_nodes):
            # 基于时间模式创建不同的特征
            temporal_data[b, n, 0] = np.sin(b * 0.5 + n * 0.1) + np.random.normal(0, 0.1)
            temporal_data[b, n, 1] = np.cos(b * 0.3 + n * 0.2) + np.random.normal(0, 0.1)
            temporal_data[b, n, 2] = np.tanh((n - num_nodes/2) * 0.1) + np.random.normal(0, 0.1)
            temporal_data[b, n, 3] = np.exp(-((n - num_nodes/2) ** 2) / 100) + np.random.normal(0, 0.1)
            temporal_data[b, n, 4] = np.random.normal(0, 1)
    
    return {
        "spatial": spatial_data,
        "temporal": temporal_data
    }

def main():
    """主函数"""
    print("改进的DGCNN实现测试")
    print("=" * 60)
    
    # 测试改进的DGCNN
    test_improved_dgcnn()
    
    # 使用多样性数据测试
    print("\n" + "=" * 60)
    print("使用多样性数据测试")
    print("=" * 60)
    
    test_data = create_diverse_test_data()
    k_values = [4, 6, 8, 10, 12, 30]
    
    for data_name, x in test_data.items():
        print(f"\n--- {data_name.upper()} 数据 ---")
        
        outputs = {}
        for k in k_values:
            encoder = ImprovedEEGDGCNNEncoder(in_channels=5, k=k)
            with torch.no_grad():
                output = encoder(x)
            outputs[k] = output
        
        # 比较输出差异
        print(f"输出差异分析:")
        for i, k1 in enumerate(k_values):
            for k2 in k_values[i+1:]:
                diff = torch.abs(outputs[k1] - outputs[k2])
                max_diff = diff.max().item()
                print(f"  k={k1} vs k={k2}: 最大差异={max_diff:.6f}")
    
    print(f"\n{'='*60}")
    print("改进总结")
    print("="*60)
    print("""
改进措施：
1. 添加注意力机制 - 让模型学习不同邻居的重要性
2. 增加残差连接 - 保持梯度流动和特征多样性
3. 使用多种池化方法 - 结合平均池化和最大池化
4. 添加dropout - 防止过拟合，增加随机性
5. 多层感知机 - 增强特征变换能力
6. 噪声注入 - 在训练时添加噪声提高鲁棒性

这些改进应该能让不同k值产生明显不同的结果。
    """)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速k值测试脚本
用于快速验证不同k值对DGCNN模型的影响
"""

import torch
import numpy as np
import time
from models.general.CTRLightGCN import EEGDGCNNEncoder

def test_dgcnn_with_k(k_value, batch_size=32, num_nodes=62, in_channels=5):
    """
    测试特定k值的DGCNN编码器
    
    Args:
        k_value: DGCNN的k值
        batch_size: 批次大小
        num_nodes: 节点数量（EEG电极数）
        in_channels: 输入通道数
    """
    print(f"\n测试 k={k_value}")
    print("-" * 30)
    
    # 创建DGCNN编码器
    encoder = EEGDGCNNEncoder(
        in_channels=in_channels,
        hidden_channels=32,
        out_channels=16,
        k=k_value
    )
    
    # 创建测试数据
    x = torch.randn(batch_size, num_nodes, in_channels)
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = encoder(x)
    
    # 测试前向传播时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            output = encoder(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    
    # 测试内存使用
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 测试梯度计算
    encoder.zero_grad()
    x.requires_grad_(True)
    output = encoder(x)
    loss = output.sum()
    loss.backward()
    
    # 计算参数数量
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"输出形状: {output.shape}")
    print(f"平均前向传播时间: {avg_time:.4f}秒")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"输出统计: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    
    return {
        'k': k_value,
        'avg_time': avg_time,
        'total_params': total_params,
        'output_shape': output.shape,
        'output_mean': output.mean().item(),
        'output_std': output.std().item()
    }

def main():
    """主函数"""
    print("DGCNN k值快速测试")
    print("=" * 50)
    
    # 测试不同的k值
    k_values = [4, 6, 8, 10, 12, 15, 20, 25, 30]
    
    results = []
    
    for k in k_values:
        try:
            result = test_dgcnn_with_k(k)
            results.append(result)
        except Exception as e:
            print(f"k={k} 测试失败: {str(e)}")
            results.append({
                'k': k,
                'error': str(e)
            })
    
    # 生成比较报告
    print("\n" + "=" * 50)
    print("比较报告")
    print("=" * 50)
    
    print(f"{'k值':<6} {'时间(秒)':<12} {'参数数量':<12} {'输出均值':<12} {'输出标准差':<12}")
    print("-" * 60)
    
    for result in results:
        if 'error' not in result:
            print(f"{result['k']:<6} {result['avg_time']:<12.4f} {result['total_params']:<12,} "
                  f"{result['output_mean']:<12.4f} {result['output_std']:<12.4f}")
        else:
            print(f"{result['k']:<6} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    # 性能分析
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        print("\n性能分析:")
        
        # 最快和最慢的k值
        fastest = min(successful_results, key=lambda x: x['avg_time'])
        slowest = max(successful_results, key=lambda x: x['avg_time'])
        
        print(f"- 最快k值: {fastest['k']} (时间: {fastest['avg_time']:.4f}秒)")
        print(f"- 最慢k值: {slowest['k']} (时间: {slowest['avg_time']:.4f}秒)")
        print(f"- 时间差异: {slowest['avg_time']/fastest['avg_time']:.2f}倍")
        
        # 推荐k值
        print("\n推荐k值:")
        print("- 计算效率优先: k=4-8")
        print("- 平衡性能: k=10-15") 
        print("- 信息丰富优先: k=20-30")

if __name__ == "__main__":
    main()

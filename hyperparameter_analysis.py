#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGCNN超参数分析脚本
用于测试不同k值对模型性能的影响
"""

import os
import json
import subprocess
import time
from datetime import datetime

def run_experiment(k_value, model_name="CTRLightGCNCTR", dataset="your_dataset"):
    """
    运行单个实验
    
    Args:
        k_value: DGCNN的k值
        model_name: 模型名称
        dataset: 数据集名称
    """
    print(f"\n{'='*50}")
    print(f"开始实验: k={k_value}")
    print(f"{'='*50}")
    
    # 构建命令
    cmd = [
        "python", "main.py",
        "--model", model_name,
        "--dataset", dataset,
        "--dgcnn_k", str(k_value),
        "--epoch", "50",  # 可以根据需要调整
        "--batch_size", "256",
        "--lr", "0.001",
        "--device", "cuda"  # 或 "cpu"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行实验
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        # 解析结果
        if result.returncode == 0:
            # 从输出中提取指标（需要根据你的输出格式调整）
            output = result.stdout
            print(f"实验成功完成，耗时: {duration:.2f}秒")
            print("输出:")
            print(output)
            
            # 这里需要根据你的输出格式来解析具体的指标
            # 例如：AUC, NDCG, Recall等
            metrics = parse_metrics(output)
            
            return {
                "k_value": k_value,
                "status": "success",
                "duration": duration,
                "metrics": metrics,
                "output": output
            }
        else:
            print(f"实验失败，错误信息:")
            print(result.stderr)
            return {
                "k_value": k_value,
                "status": "failed",
                "duration": duration,
                "error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"实验超时 (k={k_value})")
        return {
            "k_value": k_value,
            "status": "timeout",
            "duration": 3600
        }
    except Exception as e:
        print(f"实验异常 (k={k_value}): {str(e)}")
        return {
            "k_value": k_value,
            "status": "error",
            "error": str(e)
        }

def parse_metrics(output):
    """
    从输出中解析指标
    需要根据你的实际输出格式来调整
    """
    metrics = {}
    
    # 示例解析逻辑（需要根据实际输出调整）
    lines = output.split('\n')
    for line in lines:
        if 'AUC' in line:
            try:
                auc = float(line.split()[-1])
                metrics['AUC'] = auc
            except:
                pass
        elif 'NDCG' in line:
            try:
                ndcg = float(line.split()[-1])
                metrics['NDCG'] = ndcg
            except:
                pass
        elif 'Recall' in line:
            try:
                recall = float(line.split()[-1])
                metrics['Recall'] = recall
            except:
                pass
    
    return metrics

def main():
    """主函数"""
    print("DGCNN超参数分析")
    print("="*50)
    
    # 定义要测试的k值
    k_values = [4, 6, 8, 10, 12, 15, 20, 25, 30]
    
    # 实验配置
    model_name = "CTRLightGCNCTR"  # 根据你的模型名称调整
    dataset = "your_dataset"       # 根据你的数据集名称调整
    
    # 存储所有实验结果
    results = []
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hyperparameter_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"结果将保存到: {results_dir}")
    
    # 运行所有实验
    for k in k_values:
        result = run_experiment(k, model_name, dataset)
        results.append(result)
        
        # 保存中间结果
        with open(f"{results_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # 等待一段时间再开始下一个实验
        time.sleep(10)
    
    # 生成分析报告
    generate_report(results, results_dir)
    
    print(f"\n所有实验完成！结果保存在: {results_dir}")

def generate_report(results, results_dir):
    """生成分析报告"""
    
    # 过滤成功的实验
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        print("没有成功的实验结果")
        return
    
    # 创建报告
    report = []
    report.append("# DGCNN超参数分析报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 实验概览
    report.append("## 实验概览")
    report.append(f"- 总实验数: {len(results)}")
    report.append(f"- 成功实验数: {len(successful_results)}")
    report.append(f"- 失败实验数: {len(results) - len(successful_results)}")
    report.append("")
    
    # 详细结果
    report.append("## 详细结果")
    report.append("")
    report.append("| k值 | 状态 | 耗时(秒) | AUC | NDCG | Recall |")
    report.append("|-----|------|----------|-----|------|--------|")
    
    for result in results:
        k = result['k_value']
        status = result['status']
        duration = result.get('duration', 'N/A')
        
        if status == 'success':
            metrics = result.get('metrics', {})
            auc = metrics.get('AUC', 'N/A')
            ndcg = metrics.get('NDCG', 'N/A')
            recall = metrics.get('Recall', 'N/A')
        else:
            auc = ndcg = recall = 'N/A'
        
        report.append(f"| {k} | {status} | {duration} | {auc} | {ndcg} | {recall} |")
    
    report.append("")
    
    # 性能分析
    if successful_results:
        report.append("## 性能分析")
        report.append("")
        
        # 找到最佳k值
        best_k = None
        best_auc = -1
        
        for result in successful_results:
            metrics = result.get('metrics', {})
            auc = metrics.get('AUC', 0)
            if auc > best_auc:
                best_auc = auc
                best_k = result['k_value']
        
        if best_k is not None:
            report.append(f"**最佳k值**: {best_k} (AUC: {best_auc:.4f})")
            report.append("")
        
        # 趋势分析
        report.append("### 趋势分析")
        report.append("- 小k值 (4-8): 图结构稀疏，计算效率高，但可能信息不足")
        report.append("- 中等k值 (10-15): 平衡连接密度和计算复杂度")
        report.append("- 大k值 (20-30): 图结构密集，信息丰富，但计算开销大")
        report.append("")
    
    # 保存报告
    with open(f"{results_dir}/report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"分析报告已生成: {results_dir}/report.md")

if __name__ == "__main__":
    main()

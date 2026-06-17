import pandas as pd
import math

# 参数配置
input_csv = "data(eeg)%30.csv"    # 原始文件路径
output_csv = "test.csv"  # 新文件路径
target_column = "user_id"  # 需要处理的列名（替换为你的列名）

# 读取原始文件
df = pd.read_csv(input_csv)

def take_remaining(group):
    total = len(group)
    n_remove = math.ceil(total * 0.33)  # 计算前70%要删除的行数
    return group.tail(total - n_remove)  # 取剩余的行

# 按列分组处理
result_df = df.groupby(target_column, group_keys=False).apply(take_remaining)

# 保存结果
result_df.to_csv(output_csv, index=False)

print(f"剩余30%数据已保存到: {output_csv}")
print(f"原始文件行数: {len(df)} → 新文件行数: {len(result_df)}")
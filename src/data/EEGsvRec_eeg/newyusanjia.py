import pandas as pd
import math

# 参数配置
input_csv = "data(eeg)%30.csv"      # 原始文件路径
output_csv = "dev.csv"    # 新文件路径
target_column = "user_id"  # 需要筛选的列名（替换为你的列名）

# 读取CSV文件
df = pd.read_csv(input_csv)

# 按目标列分组，并对每个组取前70%的行
def take_top_percent(group):
    n = math.ceil(len(group) * 0.33)  # 计算需要取的行数（向上取整）
    return group.head(n)

# 执行分组操作
result_df = df.groupby(target_column, group_keys=False).apply(take_top_percent)

# 保存结果到新文件
result_df.to_csv(output_csv, index=False)

print(f"新文件已生成: {output_csv}")
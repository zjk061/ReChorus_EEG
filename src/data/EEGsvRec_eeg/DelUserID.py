import pandas as pd

# 1. 读取CSV文件
df = pd.read_csv("data(eeg).csv")

# 2. 指定要修改的列名（例如"column_name"）
column_name = "user_id"
df[column_name] = df[column_name] - 1

# 5. 保存修改后的数据到新文件（或覆盖原文件）
df.to_csv("data(eeg).csv", index=False)
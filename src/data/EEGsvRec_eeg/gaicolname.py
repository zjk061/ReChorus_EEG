import pandas as pd

# 参数设置
input_file = "test.csv"       # 原始文件路径
output_file = "test.csv"     # 新文件路径
old_col_name = "start_time"    # 需要修改的旧列名
new_col_name = "time"    # 新列名

# 读取CSV文件
df = pd.read_csv(input_file)

# 修改单个列名
df.rename(columns={old_col_name: new_col_name}, inplace=True)

# 保存修改后的文件
df.to_csv(output_file, index=False)
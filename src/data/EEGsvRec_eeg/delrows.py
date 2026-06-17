import pandas as pd

# 示例 CSV 文件路径
file_path = "data(eeg).csv"

# 加载 CSV 文件
data_df = pd.read_csv(file_path)

# 定义要删除的特定数字
"""
把御三家里有的item_id但是item_meta中没有的item_id条目全部从御三家中删去
"""
specific_numbers = [64, 81, 92, 160, 190, 238, 310, 354, 360, 367, 377, 410, 411, 415, 418, 479, 481, 517, 609, 618, 629, 657, 735, 788, 909, 918, 1008, 1012, 1042, 1086, 1087, 1108, 1146, 1149, 1180, 1195, 1206, 1215, 1245, 1246, 1261, 1264, 1293, 1316, 1329, 1351, 1365, 1411, 1443, 1508, 1582, 1585, 1661, 1677, 1683, 1787, 1878, 1885, 1894, 1933, 1958, 1985, 1997, 2038, 2045, 2074, 2086, 2115, 2126, 2135, 2212, 2226, 2263, 2335, 2338, 2404, 2427, 2451, 2454, 2459, 2476]

# 假设要检查的列名为 'column_name'
column_name = 'item_id'

# 删除包含特定数字的行
filtered_df = data_df[~data_df[column_name].isin(specific_numbers)]

# 保存修改后的数据到新的 CSV 文件
output_file_path = "data(eeg).csv"
filtered_df.to_csv(output_file_path, index=False)

print(f"修改后的 CSV 文件已生成：{output_file_path}")
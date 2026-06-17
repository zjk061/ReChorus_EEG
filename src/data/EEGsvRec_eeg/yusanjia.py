import csv

input_csv_path = "data(eeg).csv"  # 原始的 CSV 文件路径
output_csv_path = "dev.csv"  # 输出的 CSV 文件路径
start_row = 3175  # 起始行（从0开始计数）
end_row = 3556   # 结束行（不包括这一行）

# 打开原始文件并读取内容
with open(input_csv_path, mode="r", newline="", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    rows = list(reader)  # 将文件内容读取为列表

# 截取指定范围的行
# 注意：需要保留表头，因此从 start_row + 1 开始截取
selected_rows = [rows[0]] + rows[start_row + 1:end_row + 1]

# 将截取的数据写入新的 CSV 文件
with open(output_csv_path, mode="w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(selected_rows)

print(f"新的 CSV 文件已生成：{output_csv_path}")
import pandas as pd
import numpy as np
# item_features = None

# item_meta_df = pd.read_csv("./item_meta.csv",sep=',')
# item_features = item_meta_df.set_index('item_id').to_dict(orient='index')

# print(item_features)

original_dict = {
    2416: {
        'i_music_tempo_f': 135.9991776,
        'i_count_f': 1387,
        'i_height_c': 1024,
    },
    2417: {
        'i_music_tempo_f': 136.9991776,
        'i_count_f': 13,
        'i_height_c': 10,        
    }
}

flattened_dict = {
    outer_key: list(inner_dict.values())
    for outer_key, inner_dict in original_dict.items()
}

# 输出结果示例
print(flattened_dict)

tensor_np = np.array(list(flattened_dict.values()))
print(tensor_np)
# 输出（示例）:
# [[1.35999178e+02 1.38700000e+03 1.02400000e+03]
#  [1.36999178e+02 1.30000000e+01 1.00000000e+01]]

import os
import json

cur_dir = os.path.dirname(os.path.abspath(__file__))
def get_colors_lst():
    with open(f'{cur_dir}/colors.json') as f:
        colors_dict = json.load(f)
    colors = []
    for name, values in colors_dict.items():
        colors.append(values)
    
    # transpose color_matrix
    trans_color = []
    for row in zip(*colors):
        trans_color.extend(row)
    
    return trans_color
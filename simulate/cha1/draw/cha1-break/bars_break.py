# usage example: python3 bars.py -c 2 -if ./bars_data.json

import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{cur_dir}/../common')
import colors, parse, loaddata

import matplotlib.pyplot as plt

import numpy as np
import json
import math
import re, collections

import warnings
# Filter out a specific warning
warnings.simplefilter(action="ignore")

# parse command
args = parse.parse_cmd()
# parse input data file, parse it, and do sanity check
data,(rows, cols), (nblks, blkwdth), (ncomprs, legends) = loaddata.load_and_parse_data_bar_break(os.path.join(os.path.dirname(__file__), args.inputfile), args)

# Create subplots
fig, axs = plt.subplots(nrows=rows, ncols=args.cols)
fig.subplots_adjust(hspace=0.5, wspace=0.3)
if rows==args.cols==1:
    axs = [axs]
else:
    axs = axs.flatten() # Flatten the array of axes for easier indexing

# Define colors
clrs = colors.get_colors_lst()

# parse data and other metadatas for each subplots from data dict
for figname, figitems in data.items():
    fid = int(re.search(r'fig(\d+)', figname).group(1))
    if 'group-data' in figitems:
        grpdata = figitems['group-data']
        # base x coor
        xcoor = [i*(blkwdth+1) for i in range(nblks)]
        # draw breakdown bars by groups
        clr_idx = -1
        legend_clr_map = collections.defaultdict() # 记录legend item名称与color id的对应关系
        for i in range(ncomprs):
            btm_yvalue = [0 for _ in range(nblks)]
            for j in range(len(legends[f'fig{fid}'][i])):
                # legend item (label) name
                lname = legends[f'fig{fid}'][i][j]
                # ensure the clr_idx
                if lname in legend_clr_map:
                    clr_idx = legend_clr_map[lname]
                    axs[fid].bar(np.array(xcoor)+args.barwidth*i, grpdata[f'c{i}'][j], color=clrs[clr_idx], width=args.barwidth, bottom = btm_yvalue)
                else:
                    clr_idx += 1
                    axs[fid].bar(np.array(xcoor)+args.barwidth*i, grpdata[f'c{i}'][j], color=clrs[clr_idx], width=args.barwidth, label=lname, bottom = btm_yvalue)
                
                # record color map
                legend_clr_map[lname] = clr_idx
                for k in range(len(btm_yvalue)):
                    btm_yvalue[k] += grpdata[f'c{i}'][j][k]
        # set x-labels
        axs[fid].set_xticks(np.array(xcoor)+blkwdth/2 - args.barwidth/2)
        axs[fid].set_xticklabels([xln.strip() for xln in figitems['x-labels'].split(',')])
        fig.autofmt_xdate() # Rotate the x-axis label based on available space
        # set x-title, y-title, fig-title
        axs[fid].set_xlabel(figitems['x-title'])
        axs[fid].set_ylabel(figitems['y-title'])
        axs[fid].set_title(figitems['fig-title'])
        # show legend
        axs[fid].legend()

# Save the figure to a file
plt.savefig(args.outputfile)

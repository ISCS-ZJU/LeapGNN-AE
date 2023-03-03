# usage example: python3 bars.py -c 2 -if ./bars_data.json

import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{cur_dir}/../common')
import colors, parse, loaddata

import matplotlib.pyplot as plt

import numpy as np
import json
import math
import re

import warnings
# Filter out a specific warning
warnings.simplefilter(action="ignore")

# parse command
args = parse.parse_cmd()
# parse input data file, parse it, and do sanity check
data,(rows, cols), (nblks, blkwdth), (ncomprs, legends) = loaddata.load_and_parse_data_bar(os.path.join(os.path.dirname(__file__), args.inputfile), args)

# Create subplots
fig, axs = plt.subplots(nrows=rows, ncols=args.cols)
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
        # draw bars by groups
        for j in range(ncomprs):
            axs[fid].bar(np.array(xcoor)+args.barwidth*j, grpdata[f'c{j}'], color=clrs[j+3], width=args.barwidth, label=legends[j], edgecolor='black', linewidth=1.2)
        # set x-labels
        axs[fid].set_xticks(np.array(xcoor)+blkwdth/2 - args.barwidth/2)
        axs[fid].set_xticklabels([xln.strip() for xln in figitems['x-labels'].split(',')])
        # set x-title, y-title, fig-title
        axs[fid].set_xlabel(figitems['x-title'])
        axs[fid].set_ylabel(figitems['y-title'])
        axs[fid].set_title(figitems['fig-title'])
        # show legend
        axs[fid].legend()

# Save the figure to a file
plt.savefig(args.outputfile)

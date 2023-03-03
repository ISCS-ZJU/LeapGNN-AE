import os
import json
import collections

def load_and_parse_data_bar(filepath, args):
    # Load data from json file and parse data
    with open(filepath) as f:
        data = json.load(f)
    # sanity check
    for figname, figitems in data.items():
        if 'x-labels' in figitems and 'group-data' in figitems:
            nblks = len(list(figitems['group-data'].values())[0])
            nxlabels = len(figitems['x-labels'].split(','))
            ncomprs = len(figitems['group-data'])
            assert nblks == nxlabels, f'number of nblks {nblks} is not equal to number of nxlabels {nxlabels} in {figname}'
            assert len(figitems['group-data']) == len(figitems['legends']), f'number of legends or group data error, plz check again in {figname}'
    # number of figures for rows and cols
    rows = (len(data) + args.cols - 1) // args.cols
    print(f'=> rows={rows}, cols={args.cols}')
    # legend names
    legend_names = [cname.strip() for cname in figitems['legends']]
    # block width
    block_width = ncomprs * args.barwidth
    
    return data,(rows, args.cols), (nblks, block_width), (ncomprs, legend_names)

def load_and_parse_data_bar_break(filepath, args):
    # legend names
    legends = collections.defaultdict()
    # Load data from json file and parse data
    with open(filepath) as f:
        data = json.load(f)
    # sanity check
    for figname, figitems in data.items():
        if 'x-labels' in figitems and 'group-data' in figitems:
            # check x-labels
            nblks = len(list(figitems['group-data'].values())[0][0])
            nxlabels = len(figitems['x-labels'].split(','))
            assert nblks == nxlabels, f'number of nblks {nblks} is not equal to number of nxlabels {nxlabels} in {figname}'
            # check legends
            assert len(figitems['legends']) == len(figitems['group-data']), f' len of legends should be equal to len of group-data'
            for j, sublst in enumerate(figitems['legends']):
                assert len(sublst)==len(figitems['group-data'][f'c{j}']), f'len of legends {j} is not equal to len of group-data c{j}'
            # legend_names
            legends[figname] = figitems['legends']
            
    # number of figures for rows and cols
    rows = (len(data) + args.cols - 1) // args.cols
    print(f'=> rows={rows}, cols={args.cols}')
    # number of comparisons
    ncomprs = len(figitems['legends'])
    # block width
    block_width = ncomprs * args.barwidth
    
    return data,(rows, args.cols), (nblks, block_width), (ncomprs, legends)
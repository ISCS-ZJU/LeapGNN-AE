import argparse

# Set up command line argument parser
def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", '-c', type=int, default=1, help="number of subplots on each row")
    parser.add_argument("--inputfile", '-if', type=str, default="bars_data.json", help="name of the file to load data from")
    parser.add_argument("--outputfile", '-of', type=str, default="figure.png", help="name of the file to save the figure as")
    parser.add_argument("--barwidth", '-bw', type=int, default=1, help="barwidth")

    args = parser.parse_args()
    return args
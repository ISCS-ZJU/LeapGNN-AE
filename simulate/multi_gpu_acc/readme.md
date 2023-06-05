## 使用单 GPU 模拟大型分布式集群 DP 训练的精度

global shuffling: time python3 simulate/multi_gpu_acc/dgl_default_simuacc.py -mn gcn -bs 6000 -s 5-5 -ep 50 -d ./dist/repgnn_data/reddit0/ --log --eval --distnodes 8

local shuffling: time python3 simulate/multi_gpu_acc/dgl_default_simuacc.py -mn gcn -bs 6000 -s 5-5 -ep 50 -d ./dist/repgnn_data/reddit0/ --log --eval --distnodes 8 --local

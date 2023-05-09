## 使用单 GPU 模拟大型分布式集群 DP 训练的精度

time python3 simulate/multi_gpu_acc/dgl_default_simuacc.py -mn gcn -bs 8000 -s 2 -ep 1 -d ./dist/repgnn_data/ogbn_arxiv128/ --log --eval --distnodes 8
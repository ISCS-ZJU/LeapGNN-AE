# ToDoList

- [ ] 新增模型(需要基于我们使用的DGL版本重写代码)：
    - [x] GraphSAGE；
    - [x] GAT；
    - [ ] deeper layer的GNN模型，DeepGCN、GNN'1000 (参考链接：https://github.com/lightaime/deep_gcns_torch)；
- [ ] 适配新的数据集，如ogbn-papers100M, ogbn-mag240M等等（参考钉钉共享文档里的“相关工作整理”）。现在已经有的数据集是ogbn-arxiv和ogbn-products，可以再搞两个更大的数据集。
    - [ ] metis出现OOM问题:yq1机器上运行`python3 simulate/dgl_default_sampling_simulate.py -bs 8000 -ep 1 --world-size 2 --sampling 2-2-2 -d ./dist/repgnn_data/ogbn_papers100M0`
- [ ] 训练代码缺少validation accuracy；
- [ ] grpc调用时候，传输给golang端graph node id的时间开销有30%左右，不知道是否需要/可以优化？
- [ ] 现在的代码层间没有

## 检测k-hop对训练输入数据batch size的点数扩增程度

time python3 simulate/batch_size_amplify/dgl_default_bs_amplify.py -mn gcn -bs 2048 -s 5-5 -ep 3 -d ./dist/repgnn_data/ogbn_products0/ --log --distnode 2

time python3 simulate/batch_size_amplify/dgl_default_bs_amplify.py -mn gcn -bs 8192 -s 5-5 -ep 3 -d ./dist/repgnn_data/ogbn_products0/ --log --distnode 4


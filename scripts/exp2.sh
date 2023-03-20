# 以在yq1和yq2跑为例
# yq1:
export GLOO_SOCKET_IFNAME=eno3 && time python3 dgl_default.py -mn gcn -bs 256 -s 2 -ep 1 --dist-url 'tcp://10.214.242.140:23456' --world-size 2 --rank 0 --grpc-port 10.214.242.140:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
export GLOO_SOCKET_IFNAME=eno3 && time python3 dgl_jpgnn_trans.py -mn gcn -bs 256 -s 2 -ep 1 --dist-url 'tcp://10.214.242.140:23456' --world-size 2 --rank 0 --grpc-port 10.214.242.140:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
export GLOO_SOCKET_IFNAME=eno3 && time python3 dgl_jpgnn_trans_multiplenfs.py -mn gcn -bs 256 -s 2 -ep 1 --dist-url 'tcp://10.214.242.140:23456' --world-size 2 --rank 0 --grpc-port 10.214.242.140:18110 -d ./dist/repgnn_data/ogbn_products0/ --log --nodedup
export GLOO_SOCKET_IFNAME=eno3 && time python3 dgl_jpgnn_trans_multiplenfs.py -mn gcn -bs 256 -s 2 -ep 1 --dist-url 'tcp://10.214.242.140:23456' --world-size 2 --rank 0 --grpc-port 10.214.242.140:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
# yq2
export GLOO_SOCKET_IFNAME=ens5f0 && time python3 dgl_default.py -mn gcn -bs 256 -s 2 -ep 1 --dist-url 'tcp://10.214.242.140:23456' --world-size 2 --rank 1 --grpc-port 10.214.243.19:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
export GLOO_SOCKET_IFNAME=ens5f0 && time python3 dgl_jpgnn_trans.py -mn gcn -bs 256 -s 2 -ep 1 --dist-url 'tcp://10.214.242.140:23456' --world-size 2 --rank 1 --grpc-port 10.214.243.19:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
export GLOO_SOCKET_IFNAME=ens5f0 && time python3 dgl_jpgnn_trans_multiplenfs.py -mn gcn -bs 256 -s 2 -ep 1 --dist-url 'tcp://10.214.242.140:23456' --world-size 2 --rank 1 --grpc-port 10.214.243.19:18110 -d ./dist/repgnn_data/ogbn_products0/ --log --nodedup
export GLOO_SOCKET_IFNAME=ens5f0 && time python3 dgl_jpgnn_trans_multiplenfs.py -mn gcn -bs 256 -s 2 -ep 1 --dist-url 'tcp://10.214.242.140:23456' --world-size 2 --rank 1 --grpc-port 10.214.243.19:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
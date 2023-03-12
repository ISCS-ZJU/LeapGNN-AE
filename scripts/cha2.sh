# 该脚本为了测试fetch remote feats次数的增加是否会让data fetch的总时间增加。通过改变bs从而修改fetch remote fetch的总次数;
# 执行环境：yq1-2080ti和yq2-a100
# yq1
export GLOO_SOCKET_IFNAME=eno3 && time python3 dgl_default.py -mn gcn -bs 8000 -s 10-10 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.214.242.140:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
export GLOO_SOCKET_IFNAME=eno3 && time python3 dgl_default.py -mn gcn -bs 80 -s 10-10 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.214.242.140:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
export GLOO_SOCKET_IFNAME=eno3 && time python3 dgl_default.py -mn gcn -bs 800 -s 10-10 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.214.242.140:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
# yq2
# export GLOO_SOCKET_IFNAME=ens5f0 && time python3 dgl_default.py -mn gcn -bs 8000 -s 10-10 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 0 --grpc-port 10.214.243.19:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
# export GLOO_SOCKET_IFNAME=ens5f0 && time python3 dgl_default.py -mn gcn -bs 80 -s 10-10 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 0 --grpc-port 10.214.243.19:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
# export GLOO_SOCKET_IFNAME=ens5f0 && time python3 dgl_default.py -mn gcn -bs 800 -s 10-10 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 0 --grpc-port 10.214.243.19:18110 -d ./dist/repgnn_data/ogbn_products0/ --log
BATCHSIZE=512
HIDDEN=256
EPOCH=1
DISTURL='tcp://10.214.241.227:23457'
GRPC='10.214.241.227:18112'
DATASET='./dist/repgnn_data/ogbn_products0/'
RANK=0
WORLD=4
FANOUT='10-10'
MODEL='gcn'
WORKERS=16
lr=3e-2
dr=0.5

export GLOO_SOCKET_IFNAME=ens17f0

time python3 dgl_default.py -bs $BATCHSIZE -ep $EPOCH -wkr $WORKERS\
 --dist-url $DISTURL --world-size $WORLD --rank $RANK \
  --grpc-port $GRPC --log -d \
  $DATASET -s $FANOUT -mn $MODEL -ncls 6 -lr $lr -hd $HIDDEN -dr $dr

time python3 dgl_jpgnn_trans.py -bs $BATCHSIZE -ep $EPOCH -wkr $WORKERS\
 --dist-url $DISTURL --world-size $WORLD --rank $RANK \
  --grpc-port $GRPC --log -d \
  $DATASET -s $FANOUT -mn $MODEL -ncls 6 -lr $lr -hd $HIDDEN -dr $dr

time python3 dgl_jpgnn_trans_multiplenfs.py -bs $BATCHSIZE -ep $EPOCH -wkr $WORKERS\
 --dist-url $DISTURL --world-size $WORLD --rank $RANK \
  --grpc-port $GRPC --log -d \
  $DATASET -s $FANOUT -mn $MODEL -ncls 6 -lr $lr -hd $HIDDEN -dr $dr --nodedup

# time python3 dgl_jpgnn_trans_multiplenfs.py -bs $BATCHSIZE -ep $EPOCH -wkr $WORKERS\
#  --dist-url $DISTURL --world-size $WORLD --rank $RANK \
#   --grpc-port $GRPC --log -d \
#   $DATASET -s $FANOUT -mn $MODEL -ncls 6 -lr $lr -hd $HIDDEN -dr $dr

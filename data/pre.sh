SCRIPT_DIR="$(dirname "$0")" # $0 表示本脚本文件名的full path
SETPATH="$SCRIPT_DIR/../dist/repgnn_data/"
ORINAME='ogbn-products' # ogbn-arxiv cora_full
SEED=2022
LEN=${1:-0} # 要使用原来数据集的标准长度设置为0，否则设置目标长度值；
# NAME='ogbn-products'
NAME=${ORINAME/-/_}
RESUTLDIR=$SETPATH$NAME$LEN

# data/set/clean.sh
# rm $RESUTLDIR/adj.npz

if [ ! -d "$RESUTLDIR" ]; then
    mkdir $RESUTLDIR 
fi
if [ -f "$RESUTLDIR/adj.npz" ]; then
    rm $RESUTLDIR/adj.npz
fi
if [ -f "$RESUTLDIR/feat.npy" ]; then
    rm $RESUTLDIR/feat.npy
fi
if [ -f "$RESUTLDIR/labels.npy" ]; then
    rm $RESUTLDIR/labels.npy
fi
if [ -f "$RESUTLDIR/train.npy" ]; then
    rm $RESUTLDIR/train.npy
fi
if [ -f "$RESUTLDIR/val.npy" ]; then
    rm $RESUTLDIR/val.npy
fi
if [ $ORINAME = 'citeseer' ] || [ $ORINAME = 'pubmed' ]
then
python dataset.py -n $ORINAME -p $SETPATH -l $LEN
elif [ $ORINAME = 'reddit' ]
then 
python reddit_process.py -n $ORINAME$LEN -p $SETPATH -l $LEN # 利用 dgl.data 直接产生 adj.npz, train/val/test.npy, feat.npy, labels.npy
exit 0
elif [ $ORINAME = 'cora_full' ]
then
eval "$(conda shell.bash hook)"
conda activate 0.8dgl # change conda env
python corafull_process.py -n $ORINAME$LEN -p $SETPATH # 利用 dgl.data 产生 feat.npy labels.npy pp.txt; dgl version >= 0.6.1
eval "$(conda shell.bash hook)"
conda activate repgnn # change conda env back
(exit 1) # direcited graph
# elif [ $ORINAME$LEN = 'ogbn-arxiv0' ]
# then
# eval "$(conda shell.bash hook)"
# conda activate 0.8dgl # change conda env
# python ogbn_arxiv_process.py -n $ORINAME$LEN -p $SETPATH -l $LEN # 利用 dgl.data 产生 feat.npy labels.npy pp.txt train/val/test.npy dgl version >= 0.6.1
# eval "$(conda shell.bash hook)"
# conda activate repgnn # change conda env back
# (exit 1) # direcited graph
else
python ogb_fmt.py -n $ORINAME -p $SETPATH -l $LEN # RESUTLDIR下产生feat.npy labels.npy pp.txt
fi
# RESUTLDIR下产生train.npy test.npy val.npy, adj.npz
if [ $? = 1 ]
then
echo "-> running with --directed"
    if [ $ORINAME = 'citeseer' ] || [ $ORINAME = 'pubmed' ]
    then
    python preprocess.py --ppfile pp.txt --directed --dataset $SETPATH$NAME$LEN --seed $SEED # 原来已经分出来了 train/val/test
    else
    python preprocess.py --ppfile pp.txt --directed --gen-set --dataset $SETPATH$NAME$LEN --seed $SEED
    fi
else
echo "-> running without --directed"
    if [ $ORINAME = 'citeseer' ] || [ $ORINAME = 'pubmed' ]
    then
    python preprocess.py --ppfile pp.txt --dataset $SETPATH$NAME$LEN --seed $SEED
    else
    python preprocess.py --ppfile pp.txt --gen-set --dataset $SETPATH$NAME$LEN --seed $SEED
    fi
fi

SCRIPT_DIR="$(dirname "$0")" # $0 表示本脚本文件名的full path
SETPATH="$SCRIPT_DIR/dist/repgnn_data/"
ORINAME='citeseer'
SEED=2022
LEN=0 # 要使用原来数据集的标准长度设置为0，否则设置目标长度值；
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
else
python ogb_fmt.py -n $ORINAME -p $SETPATH -l $LEN # RESUTLDIR下产生feat.npy labels.npy pp.txt
fi
# RESUTLDIR下产生train.npy test.npy val.npy, adj.npz
if [ $? ]
then
echo "running with --directed"
    if [ $ORINAME = 'citeseer' ] || [ $ORINAME = 'pubmed' ]
    then
    python ./data/preprocess.py --ppfile pp.txt --directed --dataset $SETPATH$NAME$LEN --seed $SEED
    else
    python ./data/preprocess.py --ppfile pp.txt --directed --gen-set --dataset $SETPATH$NAME$LEN --seed $SEED
    fi
else
echo "running without --directed"
    if [ $ORINAME = 'citeseer' ] || [ $ORINAME = 'pubmed' ]
    then
    python ./data/preprocess.py --ppfile pp.txt --dataset $SETPATH$NAME$LEN --seed $SEED
    else
    python ./data/preprocess.py --ppfile pp.txt --gen-set --dataset $SETPATH$NAME$LEN --seed $SEED
    fi
fi

SETPATH='/data/cwj/repgnn/'
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
python ./data/preprocess.py --ppfile pp.txt --directed --gen-set --dataset $SETPATH$NAME$LEN --seed $SEED
else
echo "running without --directed"
python ./data/preprocess.py --ppfile pp.txt --gen-set --dataset $SETPATH$NAME$LEN --seed $SEED
fi


SETPATH='/data/cwj/repgnn/'
ORINAME='ogbn-papers100M'
LEN=0 # 要使用原来数据集的标准长度设置为0，否则设置目标长度值；
# NAME='ogbn-products'
NAME=${ORINAME/-/_}
RESUTLDIR=$SETPATH$NAME$LEN

# data/set/clean.sh
rm $RESUTLDIR/adj.npz
rm $RESUTLDIR/feat.npy
rm $RESUTLDIR/labels.npy
rm $RESUTLDIR/test.npy
rm $RESUTLDIR/train.npy
rm $RESUTLDIR/val.npy
python ogb_fmt.py -n $ORINAME -p $SETPATH -l $LEN # RESUTLDIR下产生feat.npy labels.npy pp.txt

# RESUTLDIR下产生train.npy test.npy val.npy, adj.npz
if [ $? ]
then
echo "running with --directed"
python ./data/preprocess.py --ppfile pp.txt --directed --gen-set --dataset $SETPATH$NAME$LEN
else
echo "running without --directed"
python ./data/preprocess.py --ppfile pp.txt --gen-set --dataset $SETPATH$NAME$LEN
fi

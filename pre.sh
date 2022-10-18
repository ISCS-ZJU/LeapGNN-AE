
SETPATH='/data/pagraph/ogb/set/'
ORINAME='ogbn-arxiv'
LEN=256
# NAME='ogbn-products'
NAME=${ORINAME/-/_}
RESUTLDIR=$SETPATH$NAME

# data/set/clean.sh
rm $RESUTLDIR/adj.npz
rm $RESUTLDIR/feat.npy
rm $RESUTLDIR/labels.npy
rm $RESUTLDIR/test.npy
rm $RESUTLDIR/train.npy
rm $RESUTLDIR/val.npy
python ogb_fmt.py -n $ORINAME -p $SETPATH -l $LEN
if [ $? ]
then
echo "running with --directed"
python ./data/preprocess.py --ppfile pp.txt --directed --gen-set --dataset $SETPATH$NAME
else
echo "running without --directed"
python ./data/preprocess.py --ppfile pp.txt --gen-set --dataset $SETPATH$NAME
fi

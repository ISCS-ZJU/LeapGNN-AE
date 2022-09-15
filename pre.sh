
SETPATH='/data/pagraph/ogb/set'
# NAME='ogbn-arxiv'
NAME='ogbn-products'

# data/set/clean.sh
rm $SETPATH/adj.npz
rm $SETPATH/feat.npy
rm $SETPATH/labels.npy
rm $SETPATH/test.npy
rm $SETPATH/train.npy
rm $SETPATH/val.npy
python ogb_fmt.py -n $NAME -p $SETPATH
if $?
then
python ./data/preprocess.py --ppfile pp.txt --directed --gen-set --dataset $SETPATH
else
python ./data/preprocess.py --ppfile pp.txt --gen-set --dataset $SETPATH
fi


SETPATH='/data/repgnn/'
# NAME='ogbn-arxiv'
NAME='ogbn-arxiv'

# data/set/clean.sh
rm $SETPATH/adj.npz
rm $SETPATH/feat.npy
rm $SETPATH/labels.npy
rm $SETPATH/test.npy
rm $SETPATH/train.npy
rm $SETPATH/val.npy
python ogb_fmt.py -n $NAME -p $SETPATH
if [ $? ]
then
echo "running with --directed"
python ./data/preprocess.py --ppfile pp.txt --directed --gen-set --dataset $SETPATH
else
echo "running without --directed"
python ./data/preprocess.py --ppfile pp.txt --gen-set --dataset $SETPATH
fi

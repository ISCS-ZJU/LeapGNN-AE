# 52:02.16
NAME=random
cd ..
mkdir logs/$NAME
rm logs/$NAME/*.log
cd auto_test
python script_auto_server_client.py --name $NAME
cd ..
rm logs/server_output_*.log
mv logs/*.log logs/$NAME
cp logs/overall/jpgnn_trans_lessjp_*hd16* logs/$NAME
python utils/log_analys.py --dir ./logs/$NAME 
mv $NAME.csv test/figure18.csv
cd test
# figure 18(b) can be find in logs/random/jpgnn_trans_random_lessjp_dedup_True..., marked as "table:"
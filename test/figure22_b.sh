# 1:25:02
NAME=hd
cd ..
mkdir logs/$NAME
rm logs/*.log
rm logs/$NAME/*.log
cd auto_test
python script_auto_server_client.py --name $NAME
cd ..
rm logs/server_output_*.log
mv logs/*.log logs/$NAME
python utils/log_analys.py --dir ./logs/$NAME 
mv $NAME.csv test/figure22_b.csv
cd test
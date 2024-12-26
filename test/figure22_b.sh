# 1:25:02
NAME=hd
cd ..
mkdir logs/$NAME
rm logs/$NAME/*.log
cd auto_test
python script_auto_server_client.py --name $NAME
cd ..
rm logs/server_output_*.log
mv logs/*.log logs/$NAME
python utils/log_analys.py --dir ./logs/$NAME 
mv $NAME.csv test/figure22_b.csv
cd test
# to use other dataset, change '--dataset' of all_server_cmd['hd'] and all_client_cmd['hd'] in file 'script_auto_server_client.py'
# to use other model, change '--model' of all_client_cmd['hd'] in file 'script_auto_server_client.py'
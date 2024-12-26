#24:06.32
NAME=deep
cd ..
mkdir logs/$NAME
rm logs/$NAME/*.log
cd auto_test
python script_auto_server_client.py --name $NAME
cd ..
rm logs/server_output_*.log
mv logs/*.log logs/$NAME
python utils/log_analys.py --dir ./logs/$NAME 
mv $NAME.csv test/figure12.csv
cd test
# to use other dataset, change '--dataset' of all_server_cmd['deep'] and all_client_cmd['deep'] in file 'script_auto_server_client.py'
#3:09:20
NAME=overall
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
mv $NAME.csv test/figure11.csv
cd test
# to use other dataset, change '--dataset' of all_server_cmd['overall'] and all_client_cmd['overall'] in file 'script_auto_server_client.py'
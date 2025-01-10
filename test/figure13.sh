#3:10:46
NAME=indv
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
mv $NAME.csv test/figure13.csv
cd test
# to use other dataset, change '--dataset' of all_server_cmd['indv'] and all_client_cmd['indv'] in file 'script_auto_server_client.py'
# partition time: 4m48.052 * machinenum one products dataset
# 36:45.01 for 2,3,4 machine
NAME=machine
cd ..
mkdir logs/$NAME
rm logs/$NAME/*.log
cd auto_test
python script_auto_server_client.py --name $NAME
cd ..
rm logs/server_output_*.log
mv logs/*.log logs/$NAME
python utils/log_analys.py --dir ./logs/$NAME 
mv $NAME.csv test/figure23_b.csv
cd test
# make sure ip of all machines are configured in auto_test/test_config.yaml

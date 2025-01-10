#8:42.86
NAME=gpu_util
cd ..
mkdir logs/$NAME
rm logs/*.log
rm logs/$NAME/*.log
cd auto_test
python script_auto_server_client.py --name $NAME
cd ..
rm logs/server_output_*.log
mv logs/*.log logs/$NAME
python utils/get_gpu_util.py --dir ./logs/$NAME 
mv $NAME.csv test/figure20.csv
cd test
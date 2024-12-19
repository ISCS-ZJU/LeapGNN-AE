cd ../auto_test
python script_auto_server_client.py --name machine
cd ..
mkdir logs/machine
rm logs/server_output_*.log
mv logs/*.log logs/machine
python utils/log_analys.py --dir ./logs/machine 
mv machine.xlsx test/figure23_b.xlsx
cd test
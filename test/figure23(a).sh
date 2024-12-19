cd ../auto_test
python script_auto_server_client.py --name fanout
cd ..
mkdir logs/fanout
rm logs/server_output_*.log
mv logs/*.log logs/fanout
python utils/log_analys.py --dir ./logs/fanout 
mv fanout.xlsx test/figure23_a.xlsx
cd test
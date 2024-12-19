cd ../auto_test
python script_auto_server_client.py --name deep
cd ..
mkdir logs/deep
rm logs/server_output_*.log
mv logs/*.log logs/deep
python utils/log_analys.py --dir ./logs/deep 
mv deep.xlsx test/figure12.xlsx
cd test
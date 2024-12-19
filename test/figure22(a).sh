cd ../auto_test
python script_auto_server_client.py --name bs
cd ..
mkdir logs/bs
rm logs/server_output_*.log
mv logs/*.log logs/bs
python utils/log_analys.py --dir ./logs/bs 
mv bs.xlsx test/figure22_a.xlsx
cd test
cd ../auto_test
python script_auto_server_client.py --name hd
cd ..
mkdir logs/hd
rm logs/server_output_*.log
mv logs/*.log logs/hd
python utils/log_analys.py --dir ./logs/hd 
mv hd.xlsx test/figure22_b.xlsx
cd test
cd ../auto_test
python script_auto_server_client.py --name overall
cd ..
mkdir logs/overall
rm logs/server_output_*.log
mv logs/*.log logs/overall
python utils/log_analys.py --dir ./logs/overall 
mv overall.xlsx test/figure11.xlsx
cd test
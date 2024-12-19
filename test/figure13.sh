cd ../auto_test
python script_auto_server_client.py --name indv
cd ..
mkdir logs/indv
rm logs/server_output_*.log
mv logs/*.log logs/indv
python utils/log_analys.py --dir ./logs/indv 
mv indv.xlsx test/figure13.xlsx
cd test
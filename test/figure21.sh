cd ../auto_test
python script_auto_server_client.py --name full_batch
cd ..
mkdir logs/full_batch
rm logs/server_output_*.log
mv logs/*.log logs/full_batch
python utils/log_analys.py --dir ./logs/full_batch 
mv full_batch.xlsx test/figure21.xlsx
cd test
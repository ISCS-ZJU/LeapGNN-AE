#0.472s
cd ..
python utils/log_analys_gather.py --dir ./logs/indv --type miss-rate
mv indv.csv test/figure14.csv
cd test
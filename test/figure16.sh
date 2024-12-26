#0.514s
cd ..
python utils/log_analys_gather.py --dir ./logs/indv  --type request-num
mv indv.csv test/figure16.csv
cd test
#figure13.sh should be run before this script
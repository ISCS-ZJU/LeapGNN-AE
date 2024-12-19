cd ..
python utils/log_analys_gather.py --dir ./logs/indv  --type request-num
mv indv.xlsx test/figure15.xlsx
cd test
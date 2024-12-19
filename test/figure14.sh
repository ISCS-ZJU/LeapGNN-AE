cd ..
python utils/log_analys_gather.py --dir ./logs/indv --type miss-rate
mv indv.xlsx test/figure14.xlsx
cd test
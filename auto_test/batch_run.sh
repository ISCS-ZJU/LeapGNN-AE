# motiv-bottleneck
# # bs=2048 default
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0
# python3 clients_start.py --model_name graphsage --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0
# python3 clients_start.py --model_name gat --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --hidden_size 8
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --hidden_size 32
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --hidden_size 64
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 5 --hidden_size 128
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 2048 --hidden_size 16
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 2048 --hidden_size 16
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 2048 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 2048 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 2048 --hidden_size 32
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 2048 --hidden_size 32
# # bs=8000 default
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name graphsage --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gat --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
### python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 2 --dataset ogbn_products0 --batch_size 8000 --n_epochs 3 --hidden_size 16 # tmp redundant
### python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name graphsage --sampling 2-2 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gat --sampling 2-2 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000
### python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 32
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 32

# exp-overall
# # bs=2048 ours
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 2 --dataset ogbn_products0
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 3 --dataset ogbn_products0
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 2 --dataset ogbn_products0 --n_epochs 3
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 3
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 1 --dataset ogbn_arxiv0 --hidden_size 8
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 2 --dataset ogbn_arxiv0 --hidden_size 8
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --hidden_size 8
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --n_epochs 20
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 1 --dataset ogbn_arxiv0 --n_epochs 3
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 2 --dataset ogbn_arxiv0 --n_epochs 3
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 1 --dataset ogbn_arxiv0 --n_epochs 3
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 2 --dataset ogbn_arxiv0 --n_epochs 3
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 1 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 5 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 1 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 5 --hidden_size 32
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 1 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 5 --hidden_size 32
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 2 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 5 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 2 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 5 --hidden_size 32
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 2 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 5 --hidden_size 32
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 5 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 5 --hidden_size 32
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 5 --hidden_size 32

# # bs=2048 p3
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 4 --dataset ogbn_products0
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --hidden_size 32
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --hidden_size 64
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 5 --hidden_size 128
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 20
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_arxiv0 --batch_size 2048 --n_epochs 20
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --hidden_size 16 --n_epochs 3
### python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --hidden_size 32 --n_epochs 3
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --hidden_size 16
### python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 3 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 3 --hidden_size 32

# # bs=8000 ours
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 2 --dataset ogbn_products0 --batch_size 8000 --n_epochs 3
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 3 --dataset ogbn_products0 --batch_size 8000 --n_epochs 5
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 2 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
### python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 1 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
### python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16
# 6h 7.17 night
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 2 --dataset ogbn_products0 --batch_size 8000 --n_epochs 3 --hidden_size 32
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 2 --dataset ogbn_products0 --batch_size 8000 --n_epochs 3 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 2 --dataset ogbn_products0 --batch_size 8000 --n_epochs 3 --hidden_size 16
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 2 --dataset ogbn_products0 --batch_size 8000 --n_epochs 3 --hidden_size 16
# # 7.19 night start
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 1 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 16
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 1 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 16
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 1 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 1 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 32
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 1 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 1 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 32
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 2 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 32
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 32
# 7.19 night end
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 32
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 16
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --batch_size 8000 --n_epochs 10 --hidden_size 16

# bs=8000 p3
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# # python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 128 --hidden_size 16
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 32
# # python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 128 --hidden_size 32

# # papers100M0 default bs=16000
# python3 clients_start.py --model_name gcn --sampling 5-5 --run_client_idx 0 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name graphsage --sampling 5-5 --run_client_idx 0 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 5-5 --run_client_idx 0 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 5-5 --run_client_idx 0 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 32

# # papers100M0 ours jp_less
# python3 clients_start.py --model_name gcn --sampling 5-5 --run_client_idx 4 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name graphsage --sampling 5-5 --run_client_idx 4 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gat --sampling 5-5 --run_client_idx 4 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 5-5 --run_client_idx 4 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 32
# python3 clients_start.py --model_name graphsage --sampling 5-5 --run_client_idx 4 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 32
# python3 clients_start.py --model_name gat --sampling 5-5 --run_client_idx 4 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 32

# # 也许会报错所以放后面跑
# python3 clients_start.py --model_name gat --sampling 5-5 --run_client_idx 0 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gat --sampling 5-5 --run_client_idx 0 --dataset ogbn_papers100M0 --n_epochs 3 --batch_size 8000 --hidden_size 32


## ## sensitivity analysis ## ##
# # bs default
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 16384 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8192 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 4096 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 2048 --hidden_size 16
# # bs ours
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 16384 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8192 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 4096 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 2048 --hidden_size 16
# machine default+ours
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16 --cluster_server '["10.214.241.227", "10.214.241.228"]'
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --cluster_server '["10.214.241.227", "10.214.241.228"]'
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16 --cluster_server '["10.214.241.227", "10.214.241.228", "10.214.241.229"]'
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --cluster_server '["10.214.241.227", "10.214.241.228", "10.214.241.229"]'
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16 --cluster_server '["10.214.241.227", "10.214.241.228", "10.214.241.229", "10.214.241.232"]'
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16 --cluster_server '["10.214.241.227", "10.214.241.228", "10.214.241.229", "10.214.241.232"]'
# featdim default+ours
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products800 --n_epochs 3 --batch_size 8000 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products800 --n_epochs 5 --batch_size 8000 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products400 --n_epochs 3 --batch_size 8000 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products400 --n_epochs 5 --batch_size 8000 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products200 --n_epochs 3 --batch_size 8000 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products200 --n_epochs 5 --batch_size 8000 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products50 --n_epochs 3 --batch_size 8000 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products50 --n_epochs 5 --batch_size 8000 --hidden_size 16
# # nlayer default
# python3 clients_start.py --model_name gcn --sampling 5-5-5-5-5 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 5-5-5-5 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 5-5-5 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 5-5 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 5 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# # nlayer ours
# python3 clients_start.py --model_name gcn --sampling 5-5-5-5-5 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 5-5-5-5 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 5-5-5 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 5-5 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 5 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16
# # fanout default
# python3 clients_start.py --model_name gcn --sampling 5 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 20 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 40 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000 --hidden_size 16
# # fanout ours
# python3 clients_start.py --model_name gcn --sampling 5 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 20 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 40 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000 --hidden_size 16

# 优先级置后

# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 1024 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 512 --hidden_size 16
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 256 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 128 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 128 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 3 --batch_size 512 --hidden_size 16
python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 3 --batch_size 256 --hidden_size 16
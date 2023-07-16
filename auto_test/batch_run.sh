# motiv-bottleneck
# # bs=2048 default
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0
# python3 clients_start.py --model_name graphsage --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0
# python3 clients_start.py --model_name gat --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3
# # bs=8000 default
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name graphsage --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gat --sampling 2-2 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 0 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000

# exp-overall
# # bs=2048 ours
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 2 --dataset ogbn_products0
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 3 --dataset ogbn_products0
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 2 --dataset ogbn_products0 --n_epochs 3
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 3
# # bs=2048 p3
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 4 --dataset ogbn_products0
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3
# # bs=8000 ours
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 2 --dataset ogbn_products0 --batch_size 8000 --n_epochs 3
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 3 --dataset ogbn_products0 --batch_size 8000 --n_epochs 5
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 2 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 3 --dataset ogbn_products0 --n_epochs 5 --batch_size 8000
# bs=8000 p3
# python3 clients_start.py --model_name gcn --sampling 2-2 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
# python3 clients_start.py --model_name gcn --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 4 --dataset ogbn_products0 --n_epochs 3 --batch_size 8000
python3 clients_start.py --model_name graphsage --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000
python3 clients_start.py --model_name gat --sampling 10-10 --run_client_idx 4 --dataset ogbn_arxiv0 --n_epochs 3 --batch_size 8000
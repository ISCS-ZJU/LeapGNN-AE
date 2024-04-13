# # neighborhood sampling
# # arxiv
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_arxiv0/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_arxiv0/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_arxiv0/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_arxiv0/

# # products
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_products0/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_products0/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_products0/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_products0/

# # papers
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_papers100M0/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_papers100M0/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_papers100M0/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_papers100M0/

# # IT 2 layers
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2 -d ./dist/repgnn_data/it/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2 -d ./dist/repgnn_data/it/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2 -d ./dist/repgnn_data/it/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2 -d ./dist/repgnn_data/it/
# # IT 10 layers
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/it/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/it/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/it/
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/it/

# # layerwise sampling
# # arxiv
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2 -d ./dist/repgnn_data/ogbn_arxiv0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2 -d ./dist/repgnn_data/ogbn_arxiv0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2 -d ./dist/repgnn_data/ogbn_arxiv0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2 -d ./dist/repgnn_data/ogbn_arxiv0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_arxiv0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_arxiv0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_arxiv0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_arxiv0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]

# products
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2 -d ./dist/repgnn_data/ogbn_products0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2 -d ./dist/repgnn_data/ogbn_products0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2 -d ./dist/repgnn_data/ogbn_products0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2 -d ./dist/repgnn_data/ogbn_products0/ --ls-lst [50,1]
python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_products0/ --ls-lst [50,50,50,50,50,50,50,50,50,1] 
python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_products0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_products0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_products0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]

# papers
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2 -d ./dist/repgnn_data/ogbn_papers100M0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2 -d ./dist/repgnn_data/ogbn_papers100M0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2 -d ./dist/repgnn_data/ogbn_papers100M0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2 -d ./dist/repgnn_data/ogbn_papers100M0/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_papers100M0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_papers100M0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_papers100M0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/ogbn_papers100M0/ --ls-lst [50,50,50,50,50,50,50,50,50,1]

# # IT 2 layers
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2 -d ./dist/repgnn_data/it/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2 -d ./dist/repgnn_data/it/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2 -d ./dist/repgnn_data/it/ --ls-lst [50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2 -d ./dist/repgnn_data/it/ --ls-lst [50,1]
# # IT 10 layers
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 2 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/it/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 4 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/it/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 8 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/it/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
# python3 simulate/obsv1/dgl_default_layerwise_sampling_simulate_avoid_oom.py -bs 1 -ep 1 --world-size 16 --sampling 2-2-2-2-2-2-2-2-2 -d ./dist/repgnn_data/it/ --ls-lst [50,50,50,50,50,50,50,50,50,1]
## RepGNN

### Setup
1. conda create -n repgnn python==3.9 -y
2. conda activate repgnn
3. pip3 install torch torchvision
4. pip3 install psutil tqdm 

5. install dgl==0.4.1 from source:
    ```
    git clone --recurse-submodules https://github.com/dmlc/dgl.git
    git checkout 263656f89348d882478da2f993d96293e9603a22
    git submodule update --init --recursive
    sudo apt-get update
    sudo apt-get install -y build-essential python3-dev make cmake
    mkdir build
    cd build
    cmake -DUSE_CUDA=ON .. # CUDA build
    make -j4
    cd ../python
    python setup.py install
    ```

6. 如果要使用模拟生成的图，可以安装PaRMAT
    ```
    git clone https://github.com/farkhor/PaRMAT.git
    cd PaRMAT/Release
    make
    ```

### Steps
+ 产生模拟小图数据，10个点，25条边，没有重复，无方向; 图文件保存在pp.txt中；
    ```
    ./PaRMAT -noDuplicateEdges -undirected -threads 16 -nVertices 20 -nEdges 50 -output /data/pagraph/gendemo/pp.txt
    python data/preprocess.py --ppfile pp.txt --gen-feature --gen-label --gen-set --dataset /data/pagraph/gendemo
    ```
+ 启动server和client，例如4卡运行：使用metis分图，从local/remote gpu, or cpu 取数据
  ```
  python3 cpu_graph_server.py -ngpu 4
  python3 dgl_from_cpu_gpu_client.py -ngpu 4
  ```

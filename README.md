## RepGNN

### Setup
1. conda create -n repgnn python==3.9 -y
2. conda activate repgnn
3. pip3 install torch torchvision
4. pip3 install psutil tqdm pymetis

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
    
    在A100上无法编译dgl的这个版本，需要进行修改，参见dgl的提交 715b3b167d707e397f41881e481408b18eed22cd 和 5cff2f1cb2e3e307617bfa5b225df05555effb4b

6. 如果要使用模拟生成的图，可以安装PaRMAT
    ```
    git clone https://github.com/farkhor/PaRMAT.git
    cd PaRMAT/Release
    make
    ```

### Steps
+ 产生模拟小图数据，10个点，25条边，没有重复，无方向; 图文件保存在pp.txt中；
    ```
    ./PaRMAT -noDuplicateEdges -undirected -threads 16 -nVertices 20 -nEdges 50 -output /data/cwj/pagraph/gendemo/pp.txt
    python data/preprocess.py --ppfile pp.txt --gen-feature --gen-label --gen-set --dataset /data/cwj/pagraph/gendemo
    ```
+ Transfer Ogbn Dataset Format
    1. 修改pre.sh中的SETPATH为数据最终要存储的文件夹路径(不包括文件名), NAME为要下载的ogbn的数据集名称
    2. 修改pre.sh中LEN参数为需要feature长度(0表示不做处理，直接使用原本的feature)
    3. 然后添加shell脚本可执行权限：chmod u+x pre.sh 最后直接执行pre.sh 

+ 启动server和client，例如4卡运行：使用metis分图，从local/remote gpu, or cpu 取数据
  ```
  python3 cpu_graph_server.py -ngpu 4
  python3 dgl_from_cpu_gpu_client.py -ngpu 4

  # 运行ogbn-arxiv
  source activate repgnn
  python3 cpu_graph_server.py -ngpu 100 -d /data/cwj/repgnn/ogbn_arxiv600
  CUDA_VISIBLE_DEVICES=2,3,4,5 python3 jpgnn_from_cpu_client_v3.py -ngpu 4 -s 10-10 -bs 512 -d /data/cwj/repgnn/ogbn_arxiv128 -ncls 40 -ep 1
  CUDA_VISIBLE_DEVICES=2,3,4,5 python3 dgl_from_cpu_client.py -ngpu 4 -s 10-10 -bs 512 -d /data/cwj/repgnn/ogbn_arxiv128 -ncls 40 -ep 1
  ```

### Note
### jpgnn_from_cpu_client_v2.py的开发思路：
1. jpgnn_from_cpu_client_v1.py的实现思路：
   1. 每个GPU从epoch分得的数组中，分离得到一个batch的train nid；
   2. 根据每个target nid判断分图时所在的GPU，进行GPU间train nid的交换；最终得到k组sub-batch nid；(k=ngpu)
   3. 对于每组sub-batch nid，加载对应的模型ckpt参数；
   4. 通过sampler得到nid-tree；如果nid-tree不空，进行前传反传得到梯度；为空则跳过5.
   5. 若之前已经存储过当前模型的grad，那么对grad进行累加；覆写新的grad；否则直接写入新的grad；
   6. 同步，等各GPU都完成一个sub-batch的处理；然后加载下个sub-batch，共循环4-5步骤k次，直到batch train nid处理完成；
   7. 各模型加载对应rank的模型参数、梯度，进行梯度的allreduce，然后使用优化器更新参数；覆写最新的模型参数和归零后的梯度值；
   8. 从1-7执行下来，一个Batch的数据并行训练完毕。
2. jpgnn_from_cpu_client_v2.py实现动机：a)v1中每次sampler和GPU运算都是串行的，能否还是像原来那样并行？ b)一个iteration就要scatter，通信可能过于频繁。理论分析：由于每个epoch分配给每个GPU的nid是确定的，因此每个GPU在当前epoch需要训练的batch的nid也确定，sub-batch nid也确定；但是由于sub-batch nid可能不均匀，而原来的NeighborSampler不支持不同batchsize的采样；因此，能否通过每个sub-batch nid临时构造一个sampler来实现？
   1. 确定当前GPU在当前epoch需要训练的所有batch train nid，进一步通过scatter确定sub-batch train nid；（也可以不通过scatter，因为全局的epoch序列都一样的，直接计算出来即可）
   2. 产生一个异步进程根据sub-batch train nid进行sampler采样，并将结果放入队列中（仅供当前GPU使用）（是空子树也要放入）；
   3. 遍历队列，得到一个子树；
   4. 如果子树不为空：加载对应的模型，前传反传得到梯度；为空跳过5；
   5. 若之前已经存储过当前模型的grad，那么对grad进行累加；覆写新的grad；否则直接写入新的grad；
   6. 同步，等各GPU都完成一个sub-batch的处理；循环3-5步，每k次表示一个batch的数据并行训练完成，因此执行一次梯度同步，即上面的7；

### 代码流程：
#### dgl_from_cpu_client.py


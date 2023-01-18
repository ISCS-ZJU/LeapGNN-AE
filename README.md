## RepGNN

## Setup
### 0. 创建conda python环境，在python环境中安装相关包
```bash
1. conda create -n repgnn python==3.9 -y
2. conda activate repgnn
3. pip3 install torch torchvision # cuda version, like: pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
4. pip3 install psutil tqdm pymetis grpcio grpcio-tools
5. clone repgnn库以及三方库
    ```
    # clone repgnn库
    git clone https://gitee.com/nustart/repgnn.git
    # 切换到分布式分支
    git checkout --track distributed_version
    # 继续clone submodule: dgl
    git submodule init
    git submodule update
    # 进入dgl，clone dgl依赖的submodule
    cd 3rdparties/dgl
    git submodule init
    git submodule update
    # 源码编译安装dgl（已经修改为支持cuda 11的版本，如果cuda版本低于11，可以撤销dgl 9cecc3e 的提交；或者安装cuda 11环境
    conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit # 可选
    rm -rf build && mkdir build && cd build
    cmake -DUSE_CUDA=ON ..  # CUDA build, 如果提示gcc版本太高，-DCMAKE_CXX_COMPILER=/usr/bin/gcc-4.8
    make -j4
    cd ../python && python setup.py install
    ```

6. 如果要使用模拟生成的图，可以安装PaRMAT
    ```
    git clone https://github.com/farkhor/PaRMAT.git
    cd PaRMAT/Release
    make
    ```
```
### 1. 数据集（2种选择）
+ 产生模拟小图数据，10个点，25条边，没有重复，无方向; 图文件保存在pp.txt中；
    ```bash
    ./PaRMAT -noDuplicateEdges -undirected -threads 16 -nVertices 20 -nEdges 50 -output /data/cwj/pagraph/gendemo/pp.txt
    python data/preprocess.py --ppfile pp.txt --gen-feature --gen-label --gen-set --dataset /data/cwj/pagraph/gendemo
    ```
+ 使用OGBN中的开源数据集（ Transfer Ogbn Dataset Format )
    1. 修改pre.sh中的SETPATH为数据要存储的文件夹路径(不包括文件名), NAME为要下载的ogbn的数据集名称
    2. 修改pre.sh中LEN参数为需要feature长度(0表示不做处理，直接使用原本的feature)
    3. 检查sh脚本的可执行权限：chmod u+x pre.sh ；执行./pre.sh 

## 运行：
## 分布式 (git-branch: distributed_version)
### 0. 安装环境，准备数据集（参考Setup）
### 1. 安装golang和rpc相关的库
```bash
# install golang
wget https://go.dev/dl/go1.19.3.linux-amd64.tar.gz
sudo bash -c "rm -rf /usr/local/go && tar -C /usr/local -xzf go1.19.3.linux-amd64.tar.gz"
export PATH=/usr/local/go/bin:$PATH

# grpc-related
go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.28
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.2
sudo apt-get remove protobuf-compiler # remove already installed protobuf-compiler and re-install it from source 
PB_REL="https://github.com/protocolbuffers/protobuf/releases"
curl -LO $PB_REL/download/v3.12.1/protoc-3.12.1-linux-x86_64.zip
unzip protoc-3.12.1-linux-x86_64.zip -d $HOME/.local
export PATH="$PATH:$HOME/.local/bin"
```
### 2. 挂载数据集，启动go server
```bash
ln -sf src_data_folder_path/cwj/repgnn ./dist/repgnn_data # 在dist下创建软连接repgnn_data（代码中通过该软连接的路径进行数据访问）
cd repgnn/dist
go run server.go # 在多个节点都启动，然后等待服务端启动完毕
```

### 3. 运行client
```bash
cd repgnn
conda activate repgnn
# 可能需要指定 NCCL_SOCKET_IFNAME 和 GLOO_SOCKET_IFNAME 的网卡，如export GLOO_SOCKET_IFNAME=ens5f0 export GLOO_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=ens5f0 && time python3 dgl_default.py -bs 8000 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 0 --grpc-port 10.214.243.19:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ --log  # yq2 a100: export NCCL_SOCKET_IFNAME=ens5f0 ; export GLOO_SOCKET_IFNAME=ens5f0
export GLOO_SOCKET_IFNAME=ens9f0 && time python3 dgl_default.py -bs 8000 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.78.18.230:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ --log # zjg1 a100: export NCCL_SOCKET_IFNAME=ens9f0 ; export GLOO_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=eno3 && time python3 dgl_default.py -bs 8000 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.214.242.140:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ --log # yq1/zju 2080ti: export NCCL_SOCKET_IFNAME=eno3 ; export GLOO_SOCKET_IFNAME=eno3
export GLOO_SOCKET_IFNAME=ens17f1 && time python3 dgl_default.py -bs 8000 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.214.243.20:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ --log # yq3 a100: export NCCL_SOCKET_IFNAME=ens17f1 ; export GLOO_SOCKET_IFNAME=ens17f1
```
## Cross_test
用来测试跨节点通信带宽负载均衡，在之前单机多卡的代码上修改，使用时需要运行cpu_graph_server，使用参数和之前单机多卡一致

## Simulate
存放模拟运行的代码。
- dgl_default_sampling_simulate.py 在`dgl_default.py`代码基础上修改，用于模拟大规模分布式训练GNN，探究训练数据的采样分布规律。日志文件存放在simulate/logs目录下。
    + 使用方法：如模拟两个node：`python3 simulate/dgl_default_sampling_simulate.py -bs 8000 -ep 1 --world-size 2 --sampling 2-2-2 -d ./dist/repgnn_data/ogbn_arxiv128/`
- dgl_jpgnn_sampling_simulate.py 模拟micro-batch采样的结果，展示模型转移前后的数据命中分布规律。日志文件存放在simulate/logs目录下。
    + 使用方法：如模拟两个node：`python3 simulate/dgl_jpgnn_sampling_simulate.py -bs 8000 -ep 1 --world-size 2 --sampling 2-2-2 -d ./dist/repgnn_data/ogbn_arxiv128/`
- analyze_logs.py 分析上述两个程序跑出来的日志文件。
    + 使用方法：`python3 simulate/analyze_logs.py` 按提示输入文件名（从vscode中可以直接拖拽文件到终端）

#### Backup: 
+ cache server.go related golang libraries:
    ```bash
    go get github.com/jinzhu/configor
    go get github.com/sbinet/npyio
    go get github.com/sirupsen/logrus
    go get google.golang.org/grpc
    go get google.golang.org/grpc/codes
    go get google.golang.org/grpc/status
    go get google.golang.org/protobuf/reflect/protoreflect
    go get google.golang.org/protobuf/runtime/protoimpl
    ```
+ kill remaining processing.  `ps -ef | grep weijian | grep python3 | awk '{print $2}' | xargs kill -9`
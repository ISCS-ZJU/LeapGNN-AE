## RepGNN

## Setup
### 0. 创建conda python环境，在python环境中安装相关包
```bash
1. conda create -n repgnn python==3.9 -y
2. conda activate repgnn
3. pip3 install torch torchvision # cuda version, like: pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
4. pip3 install psutil tqdm pymetis grpcio grpcio-tools
5. install dgl==0.4.1 from source:
    ```
    git clone --recurse-submodules https://github.com/dmlc/dgl.git
    git checkout 263656f89348d882478da2f993d96293e9603a22
    git submodule update --init --recursive
    sudo apt-get update
    sudo apt-get install -y build-essential python3-dev make cmake
    rm -rf build && mkdir build && cd build
    cmake -DUSE_CUDA=ON ..  # CUDA build, 如果提示gcc版本太高，-DCMAKE_CXX_COMPILER=/usr/bin/gcc-4.8
    make -j4
    cd ../python && python setup.py install
    ```
    
    在A100上无法编译dgl的这个版本，需要进行CUDA.cmake和binary_reduce_sum.cu的修改，参见dgl的提交 5cff2f1cb2e3e307617bfa5b225df05555effb4b 和 715b3b167d707e397f41881e481408b18eed22cd (array下的cuda目录不用管)

6. 如果要使用模拟生成的图，可以安装PaRMAT
    ```
    git clone https://github.com/farkhor/PaRMAT.git
    cd PaRMAT/Release
    make
    ```
```
### 1. 生成数据集（2种选择）
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
export PATH=$PATH:/usr/local/go/bin

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
ln -sf src_data_folder_path/cwj/repgnn ./dist/repgnn_data/ # 在dist下创建软连接repgnn_data（代码中通过该软连接的路径进行数据访问）
cd repgnn/dist
go run server.go # 在多个节点都启动，然后等待服务端启动完毕
```

### 3. 运行client
```bash
cd repgnn
conda activate repgnn
# 可能需要指定 NCCL_SOCKET_IFNAME 和 GLOO_SOCKET_IFNAME 的网卡，如export GLOO_SOCKET_IFNAME=ens5f0 export GLOO_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=ens5f0 && time python3 dgl_default.py -bs 8000 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 0 --grpc-port 10.214.243.19:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ --log  # yq2 a100: export NCCL_SOCKET_IFNAME=ens5f0 ; export GLOO_SOCKET_IFNAME=ens5f0
export GLOO_SOCKET_IFNAME=ens9f0 && time python3 dgl_default.py -bs 8000 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.78.18.230:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ # zjg1 a100: export NCCL_SOCKET_IFNAME=ens9f0 ; export GLOO_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=eno3 && time python3 dgl_default.py -bs 8000 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.214.242.140:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ --log # yq1/zju 2080ti: export NCCL_SOCKET_IFNAME=eno3 ; export GLOO_SOCKET_IFNAME=eno3
```

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
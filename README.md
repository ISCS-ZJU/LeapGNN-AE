## RepGNN

## Setup
### 0. 创建conda python环境，在python环境中安装相关包
```bash
1. conda create -n repgnn python==3.9 -y
2. conda activate repgnn
3. pip3 install torch torchvision # cuda version, like: pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
4. pip3 install psutil tqdm pymetis grpcio grpcio-tools ogb h5py numpy==1.23.4 netifaces PyYAML asyncio GitPython openpyxl protobuf==3.20.3
5. clone repgnn库以及三方库
    ```
    # clone repgnn库
    git clone https://gitee.com/nustart/repgnn.git
    # 切换到分布式分支
    cd repgnn; git checkout distributed_version
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
7. 使用deepgcn
    先安装torch_scatter torch_cluster（需要指定torch，cuda和python版本）
    https://pytorch-geometric.com/whl/torch-1.10.1%2Bcu113.html 下载
    torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
    torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
    torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
    进行安装
    完成后再安装torch_geometric
    pip3 install torch_geometric==2.2.0
    如果出现error: metadata-generation-failed报错
    pip install setuptools==50.3.2
```
### 1. 数据集（2种选择）
+ 产生模拟小图数据，10个点，25条边，没有重复，无方向; 图文件保存在pp.txt中；
    ```bash
    ./PaRMAT/Release/PaRMAT -noDuplicateEdges -undirected -threads 16 -nVertices 20 -nEdges 50 -output ./dist/repgnn_data/gendemo/pp.txt
    python data/preprocess.py --ppfile pp.txt --gen-feature --gen-label --gen-set --dataset ./dist/repgnn_data/gendemo
    ```
+ 使用OGBN中的开源数据集（ Transfer Ogbn Dataset Format )
    1. 修改pre.sh中的SETPATH为数据要存储的文件夹路径(不包括文件名), NAME为要下载的ogbn的数据集名称
    2. 修改pre.sh中LEN参数为需要feature长度(0表示不做处理，直接使用原本的feature)
    3. 检查sh脚本的可执行权限：chmod u+x pre.sh ；执行./pre.sh 

+ 使用citeseer和pubmed数据集
    ```bash
    cd 3rdparties
    git clone https://github.com/tkipf/gcn.git # or git@github.com:tkipf/gcn.git
    拷贝 gcn/gcn/data/ 中数据集对应的文件到需要的目录，例如 cp 3rdparties/gcn/gcn/data/ind.citeseer.* /data2/cwj/repgnn/citeseer/
    修改 pre.sh 中ORINAME, 然后执行 ./pre.sh  # maybe should in repgnn conda env
    ```

## 运行：
## 手动分布式 (git-branch: distributed_version)
### 0. 安装环境，准备数据集（参考Setup）
### 1. 安装golang和rpc相关的库
```bash
# install golang
wget https://go.dev/dl/go1.19.3.linux-amd64.tar.gz
sudo bash -c "rm -rf /usr/local/go && tar -C /usr/local -xzf go1.19.3.linux-amd64.tar.gz"
export PATH=/usr/local/go/bin:$PATH
# 增加golang的下载镜像，打开go module
go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.cn,direct

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
# dgl_default.py 表示默认的方法，跑的命令如下
# dgl_jpgnn_trans.py 表示使用jpgnn策略，移动模型后的方法，跑的命令和dgl_default.py的一样，只是py文件名换一下
# dgl_jpgnn_trans_multiplenfs 表示在trans的基础上将多个nfs合并的方法，默认会去重，如果不去重需添加 --nodedup 的参数选项
cd repgnn
conda activate repgnn
# 可能需要指定 NCCL_SOCKET_IFNAME 和 GLOO_SOCKET_IFNAME 的网卡，如export GLOO_SOCKET_IFNAME=ens5f0 export GLOO_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=ens5f0 && time python3 dgl_default.py -mn gcn -bs 8000 -s 10-10-10 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 0 --grpc-port 10.214.243.19:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ --log  # yq2 a100: export NCCL_SOCKET_IFNAME=ens5f0 ; export GLOO_SOCKET_IFNAME=ens5f0
export GLOO_SOCKET_IFNAME=ens9f0 && time python3 dgl_default.py -mn gcn -bs 8000 -s 10-10-10 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.78.18.230:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ --log # zjg1 a100: export NCCL_SOCKET_IFNAME=ens9f0 ; export GLOO_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=eno3 && time python3 dgl_default.py -mn gcn -bs 8000 -s 10-10-10 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.214.242.140:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ --log # yq1/zju 2080ti: export NCCL_SOCKET_IFNAME=eno3 ; export GLOO_SOCKET_IFNAME=eno3
export GLOO_SOCKET_IFNAME=ens17f1 && time python3 dgl_default.py -mn gcn -bs 8000 -s 10-10-10 -ep 1 --dist-url 'tcp://10.214.243.19:23456' --world-size 2 --rank 1 --grpc-port 10.214.243.20:18110 -d ./dist/repgnn_data/ogbn_arxiv128/ --log # yq3 a100: export NCCL_SOCKET_IFNAME=ens17f1 ; export GLOO_SOCKET_IFNAME=ens17f1
# 跑 dgl_jpgnn_trans.py 和 dgl_jpgnn_trans_multiplenfs.py 只需要替换上的 dgl_default.py 即可，其他保持不变
```

## 自动分布式运行
相关脚本放在`auto_test`目录下, 首先进入 auto_test 目录：
1. 修改 `test_config.yaml` 文件，配置集群信息和要运行的文件信息；
2. 执行 `python3 servers_start.py` 会自动在各节点启动server脚本；查看其中一个节点的 `logs/server_output*.log`，等待server启动完毕；
3. 执行 `python3 clients_start.py` 会自动在各节点启动client脚本；日志写入`logs`目录下；在运行 lessjp 时，需要添加未减少跳跃次数的单 epoch 运行时间：`python3 clients_start.py --default-time xxx`
4. 执行 `python3 servers_kill.py` 和 `python3 clients_kill.py` 可以自动kill掉所有结点的server和client进程。
5. 如果不想手动频繁修改test_config.yaml，也可以以其为基础模板，启动客户端程序时传入自定义参数，如`python3 client_start.py --run_client_idx 3 --model_name gat --sampling 10-10 --batch-size 2048 --n-epochs 5 --hidden-size 256`


## Cross_test
用来测试跨节点通信带宽负载均衡，在之前单机多卡的代码上修改，使用时需要运行cpu_graph_server，使用参数和之前单机多卡一致

## Simulate
存放模拟运行的代码。
- motv:
    - motv/dgl_defaultsimulate.py 在`dgl_default.py`代码基础上修改，用于模拟大规模分布式训练GNN，探究训练数据的采样分布规律。日志文件存放在simulate/motv/logs目录下。
        + 使用方法：如模拟两个node：`python3 simulate/motv/dgl_defaultsimulate.py -bs 8000 -ep 1 --world-size 2 --sampling 2-2 -d ./dist/repgnn_data/ogbn_arxiv128/`
    - motv/dgl_jpgnnsimulate.py 模拟micro-batch采样的结果，展示模型转移前后的数据命中分布规律。日志文件存放在simulate/motv/logs目录下。
        + 使用方法：如模拟两个node：`python3 simulate/motv/dgl_jpgnnsimulate.py -bs 8000 -ep 1 --world-size 2 --sampling 2-2 -d ./dist/repgnn_data/ogbn_arxiv128/`
    - motv/analyze_logs.py 分析上述两个程序跑出来的日志文件。
        + 使用方法：`python3 simulate/motv/analyze_logs.py` 按提示输入文件名（从vscode中可以直接拖拽文件到终端）
- cha1
    - cha1/dgl_defaultsimulate.py 在`dgl_default.py`代码基础上修改，用于模拟大规模分布式训练GNN，模拟计算出naive迁移方式下的数据迁移量。日志文件存放在simulate/cha1/logs目录下。
        + 使用方法：如模拟两个node：`python3 simulate/cha1/dgl_defaultsimulate.py -bs 8000 -ep 1 --world-size 2 --sampling 2-2 -d ./dist/repgnn_data/ogbn_arxiv128/`
- obsv1
    - obsv1/dgl_defaultsimulate.py 在 `dgl_default.py`代码基础上修改，用于模拟大规模分布式训练GNN，探究target node=1时生产的子树中和target node属于同一机器的比例：
        + 使用方法：`python3 simulate/cha1/dgl_defaultsimulate.py -bs 1 -ep 1 --world-size 2 --sampling 2 -d ./dist/repgnn_data/ogbn_arxiv128/`

## 日志分析
在utils目录下运行log_analys.py，需要分析的文件填写在里面的log_analys.yaml中 (支持填写文件夹的路径)
生成的结果在data.xlsx中，下载到本地复制粘贴表格数据即可

## 分布式parmetis

先安装metis和GKlib

+ ```bash
  git clone https://github.com/KarypisLab/METIS.git
  cd METIS/
  make config shared=1 cc=gcc prefix=~/local i64=1
  make install
  ```
+ ```bash
  git clone https://github.com/KarypisLab/GKlib.git
  cd GKlib/
  make config 
  make
  make install
  ```

最后安装Parmetis

+ ```bash
  git clone --branch dgl https://github.com/KarypisLab/ParMETIS.git
  make config cc=mpicc prefix=~/local
  make install
  ```

如果编译过程中出现找不到x86_64-conda_cos6-linux-gnu-cc的问题：

+ 查找本地的编译文件，修改报错中mpicc文件中39行的CC为新的文件，如我的为CC="x86_64-conda_cos7-linux-gnu-cc"

如果没有，先进行安装

+ ```bash
  conda install  gcc_linux-64 
  conda install  gxx_linux-64 
  conda install  gfortran_linux-64
  ```

如果在编译过程中出现error: unknown type name 'siginfo_t'的问题：

+ 在报错的signal.h文件中注释掉78行和81行的的if和endif（包版本不同位置可能不同，重点是让#include `<bits/siginfo.h>`能够执行）

运行需要先建立密钥，保证各机器间无密码可ssh登录。

+ ```bash
  mpirun -hostfile xxx -np a pm_dglpart yyy b
  ```

其中xxx表示各服务器ip的配置文件，每行写每台服务器的ip
a表示总服务器的数量，b表示每台服务器上分几个区域
yyy表示图结构文件名的前缀，总共需要三个文件yyy_edges.txt,yyy_nodes.txt,yyy_stats.txt

如果运行时出现Open MPI failed an OFI Libfabric library call (fi_endpoint).  This is highly
unusual; your job may behave unpredictably (and/or abort) after this.

+ 运行mpirun -- version检查版本，不合适的话可以安装3.3.2版本并重新编译ParMetis
+ ```bash
  conda install mpich==3.3.2
  ```
+ 如果出现其它问题，检查kernel-headers_linux-64版本为3.10.0

## P3
client端源文件dgl_p3.py，参数与其它相同，可以直接利用自动运行脚本执行
server端源文件p3_cache.go，运行时修改static_cache.yaml中cache_type项为p3 （或者在auto_test中修改cache_type）
p3需要单独的模型结构，目前仅支持gcn，对应版本为gcn_p3.py
运行时在cache处打印frame[name].data[-1]，有时会出现全0项，使用非流访问的版本不会出现

## 大数据集preprocess更新
由于部分数据集较大，无法通过直接装入内存再选取需要的特征供cache保存，需要在生成时将feature直接分割为多个文件，直接读取对应区域的特征文件来进行装载，因此修改preprocess文件的特征生成部分
新增一些参数来控制是否保存为多份文件（--feat-multi-file），按分图结果进行分割（需要先完成分图操作）或按p3方式进行分割（--p3-feature），基于的分图方式和分图数量（--part-num --part-type）
以生成p3类型的2分图特征为例
+ ```bash
  python ./data/preprocess.py --dataset ./dist/repgnn_data/ogbn_arxiv00/ --directed --gen-feature --feat-multi-file --feat-size 101 --part-num 2 --part-type pagraph --p3-feature
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
+ kill remaining processing.  `ps -ef | grep weijian | grep python3 | grep repgnn | awk '{print $2}' | xargs kill -9`
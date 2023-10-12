## Note

#### 添加dgl子模块步骤总结：
```bash
# git添加dgl子模块
git submodule add https://github.com/dmlc/dgl.git
git commit -m 'add dgl submodule'
# 子模块内的代码完整clone，这样后续才能安装dgl
cd dgl
git submodule init
git submodule update
# 子模块切换到目的分支，并修改代码
git checkout 263656f893
#-> 对于cuda11,需要参见dgl的提交 5cff2f1cb2e3e307617bfa5b225df05555effb4b 和 715b3b167d707e397f41881e481408b18eed22cd (array下的cuda目录不用管)
# 在主项目中提交对submodule的更新
cd ..
git add -A
git commit -m 'update submodule'
```
#### install dgl==0.4.1 from source:
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


#### golang端添加/修改grpc接口步骤:
example: 实现`storage_dist.py`中的`self.get_statistic()`接口
1. 由于client不需要传递参数，因此`distcache.proto`文件中的`OpType`中添加`get_statistic = 4;`接口；`DCReply`中添加返回值`bool statistic = 7;`；
2. 根据distcache.proto中的笔记，对client和server端grpc通信需要的代码进行更新；执行后可以通过`git status`看到`dist/rpc/cache/distcache.pb.go`、`rpc_client/distcache_pb2.py` 以及 `rpc_client/distcache_pb2_grpc.py`发生了修改。（注意：为保证后续运行路径不出错，`rpc_client/distcache_pb2_grpc.py`文件中的`import distcache_pb2 as distcache__pb2`要再次手动改为`import rpc_client.distcache_pb2 as distcache__pb2`）
3. 在`rpc/distcache_rpc_imple.go`中实现新的函数；
4. 在`rpc/distcache_rpc.go`中的`DCSubmit`中注册新的grpc函数；


### 文件说明
dgl_default.py 原始的dgl代码；
dgl_jpgnn_trans.py 在dgl_default.py的基础上通过传输模型参数和梯度减少feat的通信开销；
src_modified 需要修改的源代码目录。比如实现NeighborSamplerWithDiffBatchSz类；
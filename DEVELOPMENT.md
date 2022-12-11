### Note

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
#-> 参见dgl的提交 715b3b167d707e397f41881e481408b18eed22cd 和 5cff2f1cb2e3e307617bfa5b225df05555effb4b
# 在主项目中提交对submodule的更新
cd ..
git add -A
git commit -m 'update submodule'
```

### 文件说明
dgl_default.py 原始的dgl代码；
dgl_jpgnn_trans.py 在dgl_default.py的基础上通过传输模型参数和梯度减少feat的通信开销；
src_modified 需要修改的源代码目录。比如实现NeighborSamplerWithDiffBatchSz类；
## RepGNN

### Setup
1. conda create -n repgnn python==3.9 -y
2. conda activate repgnn
3. conda install -c dglteam dgl-cuda10.1
4. pip3 install torch torchvision
5. pip3 install psutil tqdm pymetis

dgl==0.4.1 install from source:

git clone --recurse-submodules https://github.com/dmlc/dgl.git
git submodule update --init --recursive
sudo apt-get install -y build-essential python3-dev make cmake
git checkout 263656f89348d882478da2f993d96293e9603a22
mkdir build
cd build
cmake -DUSE_CUDA=ON .. # CUDA build
make -j4
cd ../python
python setup.py install

安装PaRMAT
git clone https://github.com/farkhor/PaRMAT.git
cd Release
make

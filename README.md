# Artifact Evaluation for LeapGNN

This repository contains the artifacts for evaluating the methods proposed in our paper *LeapGNN: Accelerating Distributed GNN Training Leveraging Feature-Centric Model Migration*, which was accepted at FAST'25.

<!-- TOC -->
- [Getting Started](#getting-started)
  - [Step 1: Set Up the Environment](#step-1-set-up-the-environment)
  - [Step 2: Prepare the Dataset](#step-2-prepare-the-dataset)
  - [Step 3: Run a Basic Example ("Hello World")](#step-3-run-a-basic-example-hello-world)
- [Detailed Instructions](#detailed-instructions)
- [Contact](#contact)
<!-- /TOC -->

## Getting Started

Follow the steps below to quickly set up and run the experiments for a basic "Hello World" experience. 
We used four machines, each with an A100 GPU for our most exepriments. The CUDA driver version is 550.127.08 and CUDA Version 12.4. The machines are interconnected with 10Gbps networking. 

### Step 1: Set Up the Environment

**Note**: The environment setup can be a bit complex as it involves multiple modules. If you encounter any issues, please contact me at [weijianchen@zju.edu.cn](mailto:weijianchen@zju.edu.cn), and we can provide assistance with our prepared lab environment and datasets.

#### 1. **Create a Conda Python Environment**

```bash
conda create -n repgnn python==3.9 -y
conda activate repgnn
```

#### 2. **Install Required Packages:**

```bash
pip install torch torchvision
pip install psutil tqdm pymetis grpcio grpcio-tools ogb h5py numpy==1.23.4 netifaces PyYAML asyncio gputil GitPython openpyxl protobuf==3.20.3
```
The software versions we use are:

<details>
  <summary>Click to see full software versions</summary>
  
  - `torch==1.10.1+cu113`
  - `torchvision==0.11.2+cu113`
  - `psutil==5.9.4`
  - `tqdm==4.65.0`
  - `pymetis==2023.1`
  - `grpcio==1.53.0`
  - `grpcio-tools==1.53.0`
  - `ogb==1.3.6`
  - `h5py==3.8.0`
  - `numpy==1.23.4`
  - `netifaces==0.11.0`
  - `PyYAML==6.0`
  - `asyncio==3.4.3`
  - `gputil==1.4.0`
  - `GitPython==3.1.31`
  - `openpyxl==3.1.2`
  - `protobuf==3.20.3`

</details>

#### 3. **Clone the Repository and Install Submodules**

```bash
git clone https://github.com/ISCS-ZJU/LeapGNN-AE.git
cd LeapGNN-AE
git checkout distributed_version
git submodule init
git submodule update
# Go to the dgl submodule
cd 3rdparties/dgl
git submodule init
git submodule update
# Compile and install dgl from source
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit  # Optional
rm -rf build && mkdir build && cd build
cmake -DUSE_CUDA=ON ..  # CUDA build, if gcc version is too high, use: -DCMAKE_CXX_COMPILER=/usr/bin/gcc-4.8
make -j4
cd ../python && python setup.py install
```



#### 4. **Install Go and gRPC dependencies for our distributed feature cache server**

Since our distributed cache is implemented in Go, you’ll need to install Go and gRPC-related libraries:

```bash
# Install Go
wget https://go.dev/dl/go1.19.3.linux-amd64.tar.gz
sudo bash -c "rm -rf /usr/local/go && tar -C /usr/local -xzf go1.19.3.linux-amd64.tar.gz"
export PATH=/usr/local/go/bin:$PATH

# Set up Go module
go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.cn,direct # Optional: proxy for China

# Install gRPC tools
go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.28
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.2

# Remove existing protobuf compiler and install from source
sudo apt-get remove protobuf-compiler
PB_REL="https://github.com/protocolbuffers/protobuf/releases"
curl -LO $PB_REL/download/v3.12.1/protoc-3.12.1-linux-x86_64.zip
unzip protoc-3.12.1-linux-x86_64.zip -d $HOME/.local
export PATH="$PATH:$HOME/.local/bin"
```

#### 5. **Install Dependencies for running deep model: DeepGCN**

First, install `torch_scatter` and `torch_cluster` (be sure to match your torch, cuda, and python versions).

Go to [PyTorch Geometric](https://pytorch-geometric.com/whl/torch-1.10.1%2Bcu113.html) and download:

- `torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl`
- `torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl`
- `torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl`

Then install them using pip:

```bash
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
```

Finally, install `torch_geometric`:

```bash
pip install torch_geometric==2.2.0
```

If you encounter the error `metadata-generation-failed`, you can resolve it by downgrading `setuptools`:

```bash
pip install setuptools==50.3.2
```

### Step 2: Prepare the Dataset

Here’s how to prepare the dataset. To avoid any potential issues, we have made the dataset available on our Cloud [link], and we recommend downloading the data directly from there.

**Important**: Make sure that the datasets downloaded from the cloud are copied to each machine. 
All data should be placed in the `LeapGNN-AE/dist/repgnn_data` directory, for example, `LeapGNN-AE/dist/repgnn_data/ogbn_products50`. If your disk space is limited, please place the data on a data disk and create a symbolic link to `LeapGNN-AE/dist/repgnn_data` using `ln -s`.

(Optional)
Below is our process for preparing the datasets.
We use the open-source dataset from OGBN (Transfer Ogbn Dataset Format):
1. `cd data`; Modify `pre.sh` by setting `SETPATH` to the directory where the data will be stored (excluding the filename), and `NAME` to the name of the OGBN dataset you wish to download.
2. Adjust the `LEN` parameter in `pre.sh` for the feature length (set to 0 to use the original features without modification).
3. Ensure the script is executable: `chmod u+x pre.sh`, then run `./pre.sh`.

### Step 3: Run a Basic Example ("Hello World")

We provide the `auto_test` feature to automatically start the distributed GNN training across multiple machines (ensure all machines can SSH into each other).

To run a simple GNN training on the small arxiv dataset within 5 minutes.

```bash
# Simple example
1. modify the 'cluster_servers' variable in auto_test/test_config.yaml file
2. cd test
3. ./hello_world.sh
4. The program exited without any exceptions, and a `.log` file was generated in the current directory. This indicates that the program has successfully completed its execution.
```

The command first invokes `servers_start.py` located in the `auto_test` directory to automatically launch `dist/server.go` on each node, thereby setting up the distributed feature caching system. Subsequently, the command automatically calls `clients_start.py` to initiate the corresponding client GNN training system on each node.

**Note**:
- Use `python3 servers_kill.py` and `python3 clients_kill.py` to terminate all server and client processes across the nodes.
- If you don’t want to manually modify `test_config.yaml`, you can use it as a template and pass custom arguments when starting the client, for example:


---

## Detailed Instructions


With the assurance that the basic example in Step 3 can run successfully (to confirm that there are no issues with your environment), you can proceed to execute the scripts in the order of increasing figure numbers to reproduce the results.
Note that some scripts take longer to run because they need to generate the .log files for the first time; others run more quickly as they reuse the results analyzed from previously generated .log files. Therefore, **it is essential to execute the scripts in the specified order**, as some scripts depend on the results generated by the previous ones.

```bash
cd test
# execute the scripts in the order of increasing figure numbers to reproduce the results.
Below is the estimated runtime for each .sh script:
./figure11.sh # 3h10m
./figure12.sh # 24m
./figure13.sh # 3h10m
./figure14.sh # <1min
./figure15.sh # <1min
./figure16.sh # <1min
./figure17.sh # 10min
./figure18.sh # 52min
./figure20.sh # 8min
./figure21.sh # 2h5m
./figure22-a.sh # 55min
./figure22-b.sh # 1h25m
./figure23-a.sh # 1h35m
./figure23-b.sh # 36m
./test_acc.sh # <1min for table 2
```

Each experiment corresponds to a specific figure in the paper and has a dedicated script. These scripts follow a similar structure:
(1) Run the model with the appropriate command.
(2) Generate logs containing relevant metrics.
(3) Extract target data from the logs automatically.


Upon completion, each script generates a **CSV file that contains the necessary data for the figures in our paper**. Additionally, some scripts include declarations and instructions, such as how to switch between different datasets.



**NOTE:** If an accident occurs during the run that causes the program to interrupt, you need to delete the .log files in the `logs` top-level directory (not in subfolders) before rerunning.





## Contact

For any further questions or if you encounter issues, feel free to open an issue or contact me via email at [weijianchen@zju.edu.cn](mailto:weijianchen@zju.edu.cn).

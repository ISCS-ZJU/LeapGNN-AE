repgnndir="/root/cwj/repgnn"
resultdir="${repgnndir}/results"
cmd=(
    "go run dcserver.go -config ${DCdir}/deepcache-go/dcserver-go/conf/conf-hpca/dcserver_mntcifar10_lru_1.yaml"
    "go run dcserver.go -config ${DCdir}/deepcache-go/dcserver-go/conf/conf-hpca/dcserver_mntcifar10_lru_0.2.yaml"
    "go run dcserver.go -config ${DCdir}/deepcache-go/dcserver-go/conf/conf-hpca/dcserver_mntcifar10_lru_1.yaml"
    "go run dcserver.go -config ${DCdir}/deepcache-go/dcserver-go/conf/conf-hpca/dcserver_mntcifar10_lru_0.2.yaml"
    "go run dcserver.go -config ${DCdir}/deepcache-go/dcserver-go/conf/conf-hpca/dcserver_mntcifar10_lru_1.yaml"
    "go run dcserver.go -config ${DCdir}/deepcache-go/dcserver-go/conf/conf-hpca/dcserver_mntcifar10_lru_0.2.yaml"
    "go run dcserver.go -config ${DCdir}/deepcache-go/dcserver-go/conf/conf-hpca/dcserver_mntcifar10_lru_1.yaml"
    "go run dcserver.go -config ${DCdir}/deepcache-go/dcserver-go/conf/conf-hpca/dcserver_mntcifar10_lru_0.2.yaml"
)

# 无根据文件是否存在判断是否执行下条命令
for c in ${cmd[@]}
do
    echo "Start Command $c ..."
    eval "$c"
    echo "Command $c Finished!"
done

# IFS=$'\n'
# # -d参数判断etcdFile是否存在
# for c in ${cmd[@]}; do
#     while true
#     do
#         if [ ! -f "$etcdFile" ]; then
#             sleep 5;
#         else
#             sleep 5 # 等待etcd完全运行起来
#             rm "$etcdFile" # 删除文件
#             echo "Start Command $c ..."
#             eval "$c"    # 执行命令
#             echo "Command $c Finished!"
#             break
#         fi
#     done
# done
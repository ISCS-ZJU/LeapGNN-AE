package peerclient

import (
	"main/common"
	"main/rpc/cache"
	"net"
	"strconv"
	"strings"

	log "github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

var GrpcClients map[string]cache.OperatorClient

func Build() {
	// 获取当前机器节点的ip地址, 确定当前ip是第几个part
	Curaddr := ""
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		log.Fatal("[distcache.go] Get IP addr err" + err.Error())
	}
	log.Info("[distcache.go] addrs:", addrs)

	for _, address := range addrs {
		// 检查ip地址判断是否回环地址
		if ipnet, ok := address.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				Curaddr = ipnet.IP.String()
				break
			}
		}
	}
	// 建立针对其他server的rpc client
	GrpcClients = make(map[string]cache.OperatorClient)
	ips_slice := strings.Split(common.Config.Cache_group, ",")
	for _, addr := range ips_slice {
		// log.Infof("[peerrpc.go] %v %T %v %T", addr, addr, Curaddr, Curaddr)
		if addr != Curaddr {
			for {
				log.Infoln("[peerrpc.go] start to connect to grpc server", addr+":"+strconv.Itoa(common.Config.Rpcport))
				// "google.golang.org/grpc/credentials/insecure"
				// conn, err := grpc.Dial(addr+":"+strconv.Itoa(common.Config.Rpcport), grpc.WithTransportCredentials(insecure.NewCredentials()))
				conn, err := grpc.Dial(addr+":"+strconv.Itoa(common.Config.Rpcport), grpc.WithInsecure(), grpc.WithBlock())
				if err != nil {
					log.Fatalf("[peerrpc.go] Failed to connect: %v", err)
				} else {
					log.Infof("[peerrpc.go] Connected to Grpc server %v", addr)
					GrpcClients[addr] = cache.NewOperatorClient(conn)
					break
				}
			}
		}
	}
	for k, v := range GrpcClients {
		log.Debugln(k, "'s value is", v)
	}
}

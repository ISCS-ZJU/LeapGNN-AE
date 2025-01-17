package rpc

import (
	"net"
	"strconv"
	"strings"

	"main/common"

	log "github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

func Start() {
	go run_grpc_server()
}

func run_grpc_server() {
	// 获取当前机器节点的ip地址
	var curaddr = ""
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		log.Fatal("[rpc.go] Get IP addr err" + err.Error())
	}

	for _, address := range addrs {
		// 检查ip地址判断是否回环地址
		if ipnet, ok := address.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				curaddr = ipnet.IP.String()
				if strings.HasPrefix(curaddr, "12.") {
					break
				}
			}
		}
	}
	// lis, err := net.Listen("tcp", common.Config.Node+":"+strconv.Itoa(common.Config.Rpcport))
	lis, err := net.Listen("tcp", curaddr+":"+strconv.Itoa(common.Config.Rpcport))
	if err != nil {
		log.Fatalf("[rpc.go] 监听RPC端口失败: %v", err)
	} else {
		log.Infoln("[rpc.go] GRPC服务启动成功") //, "grpc://"+common.Config.Node+":"+strconv.FormatInt(int64(common.Config.Rpcport), 10))
	}
	/*
		// 如果发送大小超出4MB，进行下面的设置
		recvSize := 20 * 1024 * 1024
		sendSize := 20 * 1024 * 1024
		var options = []grpc.ServerOption{
			grpc.MaxRecvMsgSize(recvSize),
			grpc.MaxSendMsgSize(sendSize),
		}
		s := grpc.NewServer(options...)
	*/
	recvSize := 2000 * 1024 * 1024
	sendSize := 2000 * 1024 * 1024
	var options = []grpc.ServerOption{
		grpc.MaxRecvMsgSize(recvSize),
		grpc.MaxSendMsgSize(sendSize),
	}
	s := grpc.NewServer(options...)
	// s := grpc.NewServer()
	// hello.Register(s) // 注册 hello grpc 模块
	Register(s) // 注册
	if err := s.Serve(lis); err != nil {
		log.Fatalf("[rpc.go] GRPC服务启动失败: %v", err)
	}
}

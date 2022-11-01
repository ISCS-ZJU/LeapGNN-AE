package rpc

import (
	context "context"
	"main/rpc/cache"

	log "github.com/sirupsen/logrus"
	grpc "google.golang.org/grpc"
)

type dcrpcserver struct {
	cache.UnimplementedOperatorServer
}

// Op func imple
func (s *dcrpcserver) DCSubmit(ctx context.Context, request *cache.DCRequest) (*cache.DCReply, error) {
	var reply *cache.DCReply
	reply, _ = Grpc_op_imple_get_features(request)
	return reply, nil
}

func Register(s *grpc.Server) {
	log.Info("[distcache_rpc.go] grpc通信代理服务注册")
	log.Infoln("服务端启动完成. √")
	cache.RegisterOperatorServer(s, &dcrpcserver{}) // 进行注册 grpc 通信代理服务
}

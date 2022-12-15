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
	switch request.Type {
	case cache.OpType_get_features_by_peer_server:
		reply, _ = Grpc_op_imple_get_features_by_peer_server(request)
	case cache.OpType_get_features_by_client:
		reply, _ = Grpc_op_imple_get_features_by_client(request)
	case cache.OpType_get_feature_dim:
		reply, _ = Grpc_op_imple_get_feature_dim(request)
	case cache.OpType_get_cache_info:
		reply, _ = Grpc_op_imple_get_cache_info(request)
	case cache.OpType_get_statistic:
		reply, _ = Grpc_op_imple_get_statistic(request)
	}

	return reply, nil
}

func Register(s *grpc.Server) {
	log.Info("[distcache_rpc.go] grpc通信代理服务注册")
	cache.RegisterOperatorServer(s, &dcrpcserver{}) // 进行注册 grpc 通信代理服务
}

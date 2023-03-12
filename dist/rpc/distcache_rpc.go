package rpc

import (
	context "context"
	"main/rpc/cache"
	"main/services"

	log "github.com/sirupsen/logrus"
	grpc "google.golang.org/grpc"
)

type dcrpcserver struct {
	cache.UnimplementedOperatorServer
	cache.UnimplementedOperatorFeaturesServer
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

func ChunkBytes(b []byte, chunkSize int) [][]byte {
	var chunks [][]byte
	for len(b) > 0 {
		// log.Infof("len(b)=%v", len(b))
		if len(b) < chunkSize {
			chunkSize = len(b)
		}
		chunks = append(chunks, b[:chunkSize])
		b = b[chunkSize:]
	}
	return chunks
}

// Op func imple
func (s *dcrpcserver) DCSubmitFeatures(request *cache.DCRequest, stream cache.OperatorFeatures_DCSubmitFeaturesServer) error {
	var reply *cache.DCReply
	switch request.Type {
	case cache.OpType_get_stream_features_by_client:
		reply, _ = Grpc_op_imple_get_stream_features_by_client(request)
	}
	// log.Info("len of reply.Features:", len(reply.Features))
	// ensure max chunk byte size considering feat_dim
	reply_chunks := ChunkBytes(reply.Features, int(services.DCRuntime.CacheMng.Get_MaxChunkSize()))

	for i := 0; i < len(reply_chunks); i++ {
		resp := &cache.DCReply{Features: reply_chunks[i]}
		if err := stream.Send(resp); err != nil {
			return err
		}
	}
	return nil
}

func Register(s *grpc.Server) {
	log.Info("[distcache_rpc.go] grpc通信代理服务注册")
	cache.RegisterOperatorServer(s, &dcrpcserver{}) // 进行注册 grpc 通信代理服务
	cache.RegisterOperatorFeaturesServer(s, &dcrpcserver{})
}

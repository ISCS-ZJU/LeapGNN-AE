package rpc

import (
	"main/rpc/cache"
	"main/services"

	log "github.com/sirupsen/logrus"
)

func Grpc_op_imple_get_features_by_client(request *cache.DCRequest) (*cache.DCReply, error) {
	log.Info("[distcache_rpc_imple.go] get_features_by_client 操作被调用")
	var reply cache.DCReply
	feature, err := services.DCRuntime.CacheMng.Get(request.Idx)
	if err != nil {
		log.Fatal("Get error")
	} else {
		reply.Feature = feature
	}
	return &reply, nil
}

func Grpc_op_imple_get_features_by_peer_server(request *cache.DCRequest) (*cache.DCReply, error) {
	log.Info("[distcache_rpc_imple.go] get_features_by_peer_server 操作被调用")
	var reply cache.DCReply
	feature, err := services.DCRuntime.CacheMng.PeerServerGet(request.Idx)
	if err != nil {
		log.Fatal("Get error")
	} else {
		reply.Feature = feature
	}
	return &reply, nil
}

func Grpc_op_imple_get_feature_dim(request *cache.DCRequest) (*cache.DCReply, error) {
	log.Info("[distcache_rpc_imple.go] get_feature_dim 操作被调用")
	var reply cache.DCReply
	feature := services.DCRuntime.CacheMng.Get_feat_dim()
	reply.Featdim = feature
	return &reply, nil
}

func Grpc_op_imple_get_cache_info(request *cache.DCRequest) (*cache.DCReply, error) {
	log.Info("[distcache_rpc_imple.go] get_cache_info 操作被调用")
	var reply cache.DCReply
	partidx, curaddr, request_num, local_hit_num := services.DCRuntime.CacheMng.Get_cache_info()
	reply.Partidx = partidx
	reply.Curaddr = curaddr
	reply.Requestnum = request_num
	reply.Localhitnum = local_hit_num
	return &reply, nil
}

package rpc

import (
	"main/common"
	"main/rpc/cache"
	"main/services"

	// "time"

	log "github.com/sirupsen/logrus"
)

func Grpc_op_imple_get_features_by_client(request *cache.DCRequest) (*cache.DCReply, error) {
	// log.Infof("[distcache_rpc_imple.go] get_features_by_client 操作被调用 for %v gnids.", len(request.Ids))
	var reply cache.DCReply
	features, err := services.DCRuntime.CacheMng.Get(request.Ids)
	if err != nil {
		log.Fatal("Get error")
	} else {
		reply.Features = features
	}
	return &reply, nil
}

func Grpc_op_imple_reset(request *cache.DCRequest) (*cache.DCReply, error) {
	// log.Infof("[distcache_rpc_imple.go] get_features_by_client 操作被调用 for %v gnids.", len(request.Ids))
	var reply cache.DCReply
	services.DCRuntime.CacheMng.Reset()
	return &reply, nil
}

func Grpc_op_imple_get_stream_features_by_client(request *cache.DCRequest) (*cache.DCReply, error) {
	// log.Infof("[distcache_rpc_imple.go] get_features_by_client 操作被调用 for %v gnids.", len(request.Ids))
	// st := time.Now()
	var reply cache.DCReply
	var features []byte
	var err error
	if services.DCRuntime.CacheMng.Get_type() == "static_cache"{
		features, err = services.DCRuntime.CacheMng.FastGet(request.Serids, request.Seplen)
	}else{
		features, err = services.DCRuntime.CacheMng.Get(request.Ids)
	}
	
	// totalTime += float64(time.Since(st) / time.Millisecond)
	if err != nil {
		log.Fatal("Get error")
	} else {
		reply.Features = features
	}
	// totalTime += float64(time.Since(st) / time.Millisecond)
	return &reply, nil
}

func Grpc_op_imple_get_features_by_peer_server(request *cache.DCRequest) (*cache.DCReply, error) {
	// log.Infof("[distcache_rpc_imple.go] get_features_by_peer_server 操作被调用 for %v gnids.", len(request.Ids))
	var reply cache.DCReply
	features, err := services.DCRuntime.CacheMng.PeerServerGet(request.Ids)
	if err != nil {
		log.Fatal("Get error")
	} else {
		reply.Rfeatures = features
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
	partidx, curaddr, request_num, local_hit_num, local_feats_gather_time, remote_feats_gather_time := services.DCRuntime.CacheMng.Get_cache_info()
	reply.Partidx = partidx
	reply.Curaddr = curaddr
	reply.Requestnum = request_num
	reply.Localhitnum = local_hit_num
	total_local_feats_gather_time, total_remote_feats_gather_time := float32(0.0), float32(0.0)
	for _, t := range local_feats_gather_time {
		total_local_feats_gather_time += t
	}
	for _, t := range remote_feats_gather_time {
		total_remote_feats_gather_time += t
	}
	reply.LocalFeatsGatherTime = total_local_feats_gather_time
	reply.RemoteFeatsGatherTime = total_remote_feats_gather_time
	// reply.RemoteFeatsGatherTime = float32(totalTime)
	return &reply, nil
}

func Grpc_op_imple_get_statistic(request *cache.DCRequest) (*cache.DCReply, error) {
	log.Info("[distcache_rpc_imple.go] get_statistic 操作被调用")
	var reply cache.DCReply
	statistic := common.Config.Statistic
	reply.Statistic = statistic
	return &reply, nil
}

package rpc

import (
	"main/rpc/cache"
	"main/services"

	log "github.com/sirupsen/logrus"
)

func Grpc_op_imple_get_features(request *cache.DCRequest) (*cache.DCReply, error) {
	log.Info("[distcache_rpc_imple.go] get_features操作被调用")
	var reply cache.DCReply
	feature, err := services.DCRuntime.CacheMng.Get(request.Idx)
	if err != nil {
		log.Fatal("Get error")
	} else {
		reply.Feature = feature
	}
	return &reply, nil
}

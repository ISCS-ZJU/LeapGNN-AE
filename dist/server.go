package main

import (
	"main/common"
	"main/rpc"
	"main/services"

	log "github.com/sirupsen/logrus"
)

func init() {
	common.Parser()  // parse cmdline and config
	services.Start() // cache services
	rpc.Start()      // rpc init for client to request
	// distkv.Start()   // distributed etcd kv, indicate samples cached in which node
}

func main() {
	log.Info("[dist_feat_cache_server.go] server run succeeded. √")
	done := make(chan bool)
	<-done
}

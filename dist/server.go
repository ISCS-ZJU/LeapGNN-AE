package main

import (
	"main/common"
	"main/peerclient"
	"main/rpc"
	"main/services"
	"os"
	"fmt"

	log "github.com/sirupsen/logrus"
)

func init() {
	common.Parser()  // parse cmdline and config
	services.Start() // cache services
	rpc.Start()      // rpc init for client to request
	// distkv.Start()   // distributed etcd kv, indicate samples cached in which node
	peerclient.Build() // build clients of other servers on each server
}

func checkAndDeleteFile(fileName string) error {
	if _, err := os.Stat(fileName); err == nil {
		err = os.Remove(fileName)
		if err != nil {
			return err
		}
		fmt.Printf("Deleted file: %s\n", fileName)
	}
	return nil
}

func main() {
	log.Info("[dist_feat_cache_server.go] server run succeeded. √")
	log.Infoln("服务端启动完成. √")
	fileName := "server_done.txt"
	err := checkAndDeleteFile(fileName)
	if err != nil {
		fmt.Println("Error deleting file:", err)
		return
	}
	file, err := os.Create(fileName)
	if err!= nil{
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()
	fmt.Println("File", fileName, "created successfully.")
	done := make(chan bool)
	<-done
}

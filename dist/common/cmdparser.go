package common

import (
	"flag"
	"fmt"

	log "github.com/sirupsen/logrus"
)

var (
	flagConfigPath *string
	rpcPort        *int
	dataSet        *string
	cacheGroup     *string
)

func Parser() {
	flagConfigPath = flag.String("config", "./conf/static_cache.yaml", "配置文件地址(建议使用全局路径)")
	rpcPort = flag.Int("rpcport", 0, "rpc服务使用的端口")
	// 允许用户通过命令行修改 flagConfigPath 文件中的设置
	dataSet = flag.String("dataset", "./repgnn_data/citeseer0", "Dataset path.")
	cacheGroup = flag.String("cachegroup", "10.214.243.19,10.214.241.227", "Distributed Cache Group")

	flag.Parse() // 解析命令行

	yamlparser(*flagConfigPath)

	// modify items in flagConfigPath
	if *dataSet != "" {
		Config.Dataset = *dataSet
	}
	if *cacheGroup != "" {
		Config.Cache_group = *cacheGroup
	}

	// 打印信息进行确认
	fmt.Println("Config.Dataset:", Config.Dataset)
	fmt.Println("Config.Partition:", Config.Partition)
	fmt.Println("Config.Cache_type:", Config.Cache_type)
	fmt.Println("Config.Cache_group:", Config.Cache_group)
	fmt.Println("Config.Statistic:", Config.Statistic)
	if Config.Statistic {
		log.Warnf("You turned on Statistic, thus the training time is not accurate.")
	}
}

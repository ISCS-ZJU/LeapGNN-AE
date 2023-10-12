package common

import (
	"flag"
	"fmt"
	"strings"

	log "github.com/sirupsen/logrus"
)

var (
	flagConfigPath *string
	rpcPort        *int
	dataSet        *string
	cacheGroup     *string
	cacheType      *string
	statistic      *bool
	partition_type      *string
	multi_feat_file      *bool
)

func Parser() {
	flagConfigPath = flag.String("config", "./conf/static_cache.yaml", "配置文件地址(建议使用全局路径)")
	rpcPort = flag.Int("rpcport", 0, "rpc服务使用的端口")
	// 允许用户通过命令行修改 flagConfigPath 文件中的设置
	dataSet = flag.String("dataset", "./repgnn_data/citeseer0", "Dataset path.")
	cacheGroup = flag.String("cachegroup", "10.214.243.19,10.214.241.227", "Distributed Cache Group")
	cacheType = flag.String("cachetype", "static", "cache server type")
	partition_type = flag.String("partition_type", "metis", "graph partition type")
	multi_feat_file = flag.Bool("multi_feat_file", false, "each part feat in one file or all feat in one file")
	statistic = flag.Bool("statistic", false, "whether gathering server time information")
	

	flag.Parse() // 解析命令行

	yamlparser(*flagConfigPath)

	// modify items in flagConfigPath
	if *partition_type != "" {
		Config.Partition_type = *partition_type
	}
	if *dataSet != "" {
		Config.Dataset = *dataSet
	}
	if *cacheGroup != "" {
		Config.Cache_group = *cacheGroup
	}
	if *cacheType != "" {
		Config.Cache_type = *cacheType
	}
	if *statistic {
		Config.Statistic = *statistic
	}else{
		Config.Statistic = false
	}
	if *multi_feat_file {
		Config.Multi_feat_file = *multi_feat_file
	}else{
		Config.Multi_feat_file = false
	}
	if *rpcPort != 0 {
		Config.Rpcport = *rpcPort
	}

	// 根据cache_group解析出partition的个数
	ips_slice := strings.Split(Config.Cache_group, ",")
	Config.Partition = len(ips_slice)
	// 去掉Cache_group中的空格
	Config.Cache_group = strings.Replace(Config.Cache_group, " ", "", -1)

	// 打印信息进行确认
	fmt.Println("Config.Dataset:", Config.Dataset)
	fmt.Println("Config.Partition:", Config.Partition)
	fmt.Println("Config.Cache_type:", Config.Cache_type)
	fmt.Println("Config.Cache_group:", Config.Cache_group)
	fmt.Println("Config.Partition_type:", Config.Partition_type)
	fmt.Println("Config.Multi_feat_file:", Config.Multi_feat_file)
	fmt.Println("Config.rpcPort:", Config.Rpcport)
	fmt.Println("Config.Statistic:", Config.Statistic)
	if Config.Statistic {
		log.Warnf("You turned on Statistic, thus the training time is not accurate.")
	}
}

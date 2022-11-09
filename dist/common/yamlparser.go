package common

import (
	"fmt"
	"strings"

	configor "github.com/jinzhu/configor"
	log "github.com/sirupsen/logrus"
)

var Config = Configmodel{}

func yamlparser(configpath string) {
	fmt.Println("config file yaml path:", configpath)
	configor.Load(&Config, configpath) // 解析yaml文件并写入Config变量中

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
	fmt.Println("Config.Statistic:", Config.Statistic)
	if Config.Statistic {
		log.Warnf("You turned on Statistic, thus the training time is not accurate.")
	}

}

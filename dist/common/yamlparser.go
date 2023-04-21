package common

import (
	"fmt"
	"strings"

	configor "github.com/jinzhu/configor"
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

}

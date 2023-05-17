package common

import (
	"fmt"

	configor "github.com/jinzhu/configor"
)

var Config = Configmodel{}

func yamlparser(configpath string) {
	fmt.Println("config file yaml path:", configpath)
	configor.Load(&Config, configpath) // 解析yaml文件并写入Config变量中

}

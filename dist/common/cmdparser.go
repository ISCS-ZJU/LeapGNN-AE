package common

import "flag"

var (
	flagConfigPath *string
	rpcPort        *int
)

func Parser() {
	flagConfigPath = flag.String("config", "./conf/static_cache.yaml", "配置文件地址(建议使用全局路径)")
	rpcPort = flag.Int("rpcport", 0, "rpc服务使用的端口")
	flag.Parse() // 解析命令行

	yamlparser(*flagConfigPath)
}

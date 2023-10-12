package services

import (
	"os/exec"
	"runtime"

	"main/common"

	log "github.com/sirupsen/logrus"
)

var DCRuntime *DistCache

func Start() {

	// 初始化本地的缓存，读取本地part对应的node feat到缓存中
	DCRuntime = &DistCache{
		Cache_type:  common.Config.Cache_type,
		Cache_group: common.Config.Cache_group,
	}
	initDistCache(DCRuntime)
	log.Info("[services.go] services模块启动成功")
}

func Command(arg ...string) (result string) {
	name := "/bin/bash"
	c := "-c"
	// 根据系统设定不同的命令name
	if runtime.GOOS == "windows" {
		name = "cmd"
		c = "/C"
	}
	arg = append([]string{c}, arg...)
	cmd := exec.Command(name, arg...)

	//执行命令
	if err := cmd.Run(); err != nil {
		log.Error("Error:The command is err,", err)
		return "error"
	}
	return ""
}

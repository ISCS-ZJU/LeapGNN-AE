package services

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"time"

	"main/common"
	"path/filepath"

	log "github.com/sirupsen/logrus"
)

var DCRuntime *DistCache

func Start() {
	// 调用python的pymetis进行metis分割，程序执行完毕会在本地文件系统中得到每个part的gnid
	partgnid_npy_filepath := fmt.Sprintf("%v/dist_True/%v_%v", common.Config.Dataset, common.Config.Partition,common.Config.Partition_type)
	log.Infof("%v", partgnid_npy_filepath)
	_, err := os.Stat(partgnid_npy_filepath)
	
	if err != nil {
		if !os.IsExist(err) {
			dataset_abspath, _ := filepath.Abs(common.Config.Dataset)
			exepath := filepath.Dir(filepath.Dir(filepath.Dir(dataset_abspath)))
			log.Infof(exepath)
			cmd := ""
			if common.Config.Partition_type == "pagraph"{
				cmd = fmt.Sprintf("source `which conda | xargs readlink -f | xargs dirname | xargs dirname`/bin/activate && conda activate repgnn && cd %v && python3 prepartition/dg.py --partition %v --dataset ./dist/%v && cd dist", exepath, common.Config.Partition, common.Config.Dataset)
			}else if common.Config.Partition_type == "metis"{
				cmd = fmt.Sprintf("source `which conda | xargs readlink -f | xargs dirname | xargs dirname`/bin/activate && conda activate repgnn && cd %v && python3 prepartition/metis.py --partition %v --dataset ./dist/%v && cd dist", exepath, common.Config.Partition, common.Config.Dataset)
			}
			log.Infof("Start %v ...", cmd)
			start := time.Now()
			result := Command(cmd)
			if result != "" {
				log.Fatal("partition failed")
				os.Exit(-1)
			} else {
				elapsed := time.Since(start)
				log.Infof("Time takes for %v algorithm:%v", common.Config.Partition_type, elapsed)
			}
		}
	}

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

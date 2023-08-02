package services

import (
	"errors"
	"fmt"
	"main/common"
	"net"
	"os"
	"reflect"
	// "sort"
	"strings"
	"time"

	npyio "github.com/sbinet/npyio"
	log "github.com/sirupsen/logrus"
)

type cache_mng_interface interface {
	Get_type() string
	Get([]int64) ([]byte, error)
	Put(int64, []float32) error
	PeerServerGet([]int64) ([]float32, error)                            // 为了区别client端和peer server端的get请求，从而方便统计本地和远程命中率
	Get_feat_dim() int64                                                 // 返回feature dim
	Get_cache_info() (int64, string, int64, int64, []float32, []float32) // 返回partidx, curaddr, request_num, local_hit_num, local_feats_gather_time, remote_feats_gather_time
	Get_MaxChunkSize() int64                                             // for stream transfer
	Reset()
}

type DistCache struct {
	Nid2Pid                  map[int64]int64        // graph node id 2 part id
	PartIdx                  int64                  // curaddr's graph part idx
	Curaddr                  string                 // current compute node's ip address
	Cache_type               string                 // cache type
	CacheMng                 cache_mng_interface    // cache management
	Cache_group              string                 // 所有节点的node id
	Ip_slice                 []string               // 按ip字符排序后的分布式cache中每个node的id
	all_cache_mng_init_funcs map[string]interface{} // 记录所有类型cache的初始函数
}

func initDistCache(dc *DistCache) {
	// 初始化
	dc.all_cache_mng_init_funcs = map[string]interface{}{
		"static": init_static_cache_mng,
		"p3": init_P3_cache_mng,
	}
	// 确定当前node对应的part id
	ips_slice := strings.Split(dc.Cache_group, ",")
	// sort.Slice(ips_slice, func(i, j int) bool {
	// 	return ips_slice[i] < ips_slice[j]
	// })
	log.Infof("[distcache.go] Sorted ips_slice: %v", ips_slice)
	dc.Ip_slice = ips_slice
	// 获取当前机器节点的ip地址, 确定当前ip是第几个part
	dc.Curaddr = ""
	dc.PartIdx = -1
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		log.Fatal("[distcache.go] Get IP addr err" + err.Error())
	}
	log.Info("[distcache.go] addrs:", addrs)

	for _, address := range addrs {
		// 检查ip地址判断是否回环地址
		if ipnet, ok := address.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				dc.Curaddr = ipnet.IP.String()
				break
			}
		}
	}
	for idx, addr := range ips_slice {
		if addr == dc.Curaddr {
			dc.PartIdx = int64(idx)
			break
		}
	}
	if dc.PartIdx == -1 {
		log.Errorf("PartIdx %v error.", dc.PartIdx)
		os.Exit(-1)
	} else {
		log.Infof("[distcache.go] This machine will cache graph partition %v", dc.PartIdx)
	}

	// 记录每个gnid对应所在的part id
	start := time.Now()
	dc.Nid2Pid = make(map[int64]int64)
	for pid := 0; pid < len(ips_slice); pid++ {
		partgnid_npy_filepath := fmt.Sprintf("%v/dist_True/%v_%v/%v.npy", common.Config.Dataset, common.Config.Partition,common.Config.Partition_type, pid)

		var gnid []int64 // 当前part需要读取的node feature数量
		f, _ := os.Open(partgnid_npy_filepath)
		defer f.Close()
		err := npyio.Read(f, &gnid)
		if err != nil {
			log.Fatal(err)
		}
		for _, graph_nid := range gnid {
			dc.Nid2Pid[graph_nid] = int64(pid)
		}

	}
	log.Infof("[distcache.go] It takes %v time to record Nid2Pid.", time.Since(start))

	// 根据cache type初始化对应的cache
	dc.CacheMng, _ = dc.Call(dc.Cache_type, dc)
	log.Info("[distcache.go] *** ", dc.CacheMng.Get_type(), " *** 构造完成，未初始化")
}

func (dc *DistCache) Call(funcName string, params ...interface{}) (result cache_mng_interface, err error) {
	f := reflect.ValueOf(dc.all_cache_mng_init_funcs[funcName])
	if len(params) != f.Type().NumIn() {
		err = errors.New("the number of params is out of index")
		return
	}
	in := make([]reflect.Value, len(params))
	for k, param := range params {
		in[k] = reflect.ValueOf(param)
	}
	// var res []reflect.Value
	res := f.Call(in)
	result = res[0].Interface().(cache_mng_interface)
	return
}

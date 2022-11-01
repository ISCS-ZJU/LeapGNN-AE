package services

import (
	"errors"
	"fmt"
	"main/common"
	"os"
	"sync"
	"time"

	npyio "github.com/sbinet/npyio"
	log "github.com/sirupsen/logrus"
)

type Static_cache_mng struct {
	sync.RWMutex
	cache         map[int64][]float32 // real cache, key: int64, value: []float64
	Cached_nitems int64               // cached number of data
}

func init_static_cache_mng(dc *DistCache) *Static_cache_mng {
	var static_cache Static_cache_mng
	static_cache.cache = make(map[int64][]float32)
	// 根据dc.PartIdx加载对应的node feature到dc.cache中
	// TODO: 存储的路径可以由python的metis分图后返回
	partgnid_npy_filepath := fmt.Sprintf("%v/dist_True/%v_metis/%v.npy", common.Config.Dataset, common.Config.Partition, dc.PartIdx)
	var gnid []int64 // 当前part需要读取的node feature数量
	f, _ := os.Open(partgnid_npy_filepath)
	defer f.Close()
	err := npyio.Read(f, &gnid)
	if err != nil {
		log.Fatal(err)
	}
	log.Infof("[static_cache.go] %v graph node ids are asigned to current compute node part (%v).", len(gnid), dc.Curaddr)
	// fmt.Printf("data = %v %v\n", data[:3], len(data))

	// 从并行文件系统中读取nid对应的feature数据
	// TODO: 当前实现是1次加载全部的features到内存
	feat_npy_filepath := fmt.Sprintf("%v/feat.npy", common.Config.Dataset)
	featf, _ := os.Open(feat_npy_filepath)
	defer featf.Close()
	r, err := npyio.NewReader(featf)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("npy-header: %v\n", r.Header)
	shape := r.Header.Descr.Shape // shape[0]-# of nodes, shape[1]-node feat dim

	features := make([]float32, shape[0]*shape[1])

	err = r.Read(&features)
	if err != nil {
		log.Fatal(err)
	}

	// 初始化填充点的feature到缓存
	start := time.Now()
	for _, nid := range gnid {
		start_index, end_index := nid*int64(shape[1]), (nid+1)*int64(shape[1])
		static_cache.Put(nid, features[start_index:end_index])
	}
	log.Infof("[static_cache.go] successfully cached %v nodes features.", len(gnid))
	log.Infof("[static_cache.go] It takes %v to cache these nodes' features.", time.Since(start))

	return &static_cache
}

func (static_cache *Static_cache_mng) Put(idx int64, feature []float32) error {
	static_cache.RLock()
	defer static_cache.RUnlock()
	static_cache.cache[idx] = feature // put feature into cache
	static_cache.Cached_nitems++
	return nil
}

func (static_cache *Static_cache_mng) Get(idx int64) ([]float32, error) {
	static_cache.RLock()
	defer static_cache.RUnlock()
	feature, exist := static_cache.cache[idx]
	if exist {
		return feature, nil
	} else {
		return nil, errors.New(string(idx) + "not exists.")
	}
}

func (static_cache *Static_cache_mng) Get_type() string {
	return "static_cache"
}

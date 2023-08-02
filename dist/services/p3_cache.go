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

type P3_cache_mng struct {
	sync.RWMutex
	cache         map[int64][]float32 // real cache, key: int64, value: []float64
	Cached_nitems int64               // cached number of data
	// client requests
	Feature_dim              int64     //feature dimension
	Get_request_num          int64     // # of client read requests
	Local_hit_num            int64     // # of local hit requests
	Local_feats_gather_time  []float32 // time of local feats gathering
	Remote_feats_gather_time []float32 // time of remote feats gathering
	MaxChunkSize             int64     // for stream transfer
	offset             int64     
	feat_len             int64     
}

func init_P3_cache_mng(dc *DistCache) *P3_cache_mng {
	var p3_cache P3_cache_mng
	p3_cache.cache = make(map[int64][]float32)
	// 根据dc.PartIdx加载对应的node feature到dc.cache中
	// TODO: 存储的路径可以由python的metis分图后返回
	// partgnid_npy_filepath := fmt.Sprintf("%v/dist_True/%v_metis/%v.npy", common.Config.Dataset, common.Config.Partition, dc.PartIdx)
	// var gnid []int64 // 当前part需要读取的node feature id
	// f, _ := os.Open(partgnid_npy_filepath)
	// defer f.Close()
	// err := npyio.Read(f, &gnid)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// log.Infof("[p3_cache.go] %v graph node ids are asigned to current compute node part (%v).", len(gnid), dc.Curaddr)
	// fmt.Printf("data = %v %v\n", data[:3], len(data))

	// 从并行文件系统中读取nid对应的feature数据
	// TODO: 当前实现是1次加载全部的features到内存
	if common.Config.Multi_feat_file == true{
		feat_npy_filepath := fmt.Sprintf("%v/p3_feat%v.npy", common.Config.Dataset, dc.PartIdx)
		log.Infof("%v\n", feat_npy_filepath)
		featf, _ := os.Open(feat_npy_filepath)
		// defer featf.Close()
		r, err := npyio.NewReader(featf)
		if err != nil {
			log.Fatal(err)
		}
		log.Infof("[p3_cache.go] npy-header: %v\n", r.Header)
		shape := r.Header.Descr.Shape // shape[0]-# of nodes, shape[1]-node feat dim
	
		features := make([]float32, shape[0]*shape[1])
	
		err = r.Read(&features)
		if err != nil {
			log.Fatal(err)
		}
	
		// 初始化填充点的feature到缓存
		cur_cache_id := DCRuntime.PartIdx
		tol_node := int64(len(DCRuntime.Ip_slice))
		p3_cache.feat_len = int64(shape[1])
		p3_cache.offset = (int64(shape[1]) / tol_node) * cur_cache_id
		
		start := time.Now()
		for nid := int64(0); nid < int64(shape[0]); nid++ {
			start_index, end_index := nid*int64(shape[1]), (nid+1)*int64(shape[1])
			p3_cache.Put(nid, features[start_index:end_index])
		}
		// log.Infof("[p3_cache.go] successfully cached %v nodes features.", len(gnid))
		log.Infof("[p3_cache.go] It takes %v to cache these nodes' features.", time.Since(start))
		featf.Close()
		p3_cache.Feature_dim = int64(0)
		for i:= 0; i < common.Config.Partition;i++{
			feat_npy_filepath = fmt.Sprintf("%v/p3_feat%v.npy", common.Config.Dataset, i)
			featf, _ = os.Open(feat_npy_filepath)
			r, err = npyio.NewReader(featf)
			if err != nil {
				log.Fatal(err)
			}
			shape := r.Header.Descr.Shape // shape[0]-# of nodes, shape[1]-node feat dim
			p3_cache.Feature_dim = p3_cache.Feature_dim + int64(shape[1])
			featf.Close()
		}
	}else{
		feat_npy_filepath := fmt.Sprintf("%v/feat.npy", common.Config.Dataset)
		featf, _ := os.Open(feat_npy_filepath)
		defer featf.Close()
		r, err := npyio.NewReader(featf)
		if err != nil {
			log.Fatal(err)
		}
		log.Infof("[p3_cache.go] npy-header: %v\n", r.Header)
		shape := r.Header.Descr.Shape // shape[0]-# of nodes, shape[1]-node feat dim
	
		features := make([]float32, shape[0]*shape[1])
	
		err = r.Read(&features)
		if err != nil {
			log.Fatal(err)
		}
	
		// 初始化填充点的feature到缓存
		cur_cache_id := DCRuntime.PartIdx
		tol_node := int64(len(DCRuntime.Ip_slice))
		if cur_cache_id == tol_node - 1{
			p3_cache.feat_len = (int64(shape[1]) / tol_node) + int64(shape[1]) % tol_node
		}else{
			p3_cache.feat_len = (int64(shape[1]) / tol_node)
		}
		p3_cache.offset = (int64(shape[1]) / tol_node) * cur_cache_id
	
		start := time.Now()
		for nid := int64(0); nid < int64(shape[0]); nid++ {
			start_index, end_index := nid*int64(shape[1]) + p3_cache.offset, (nid)*int64(shape[1]) + p3_cache.offset + p3_cache.feat_len
			p3_cache.Put(nid, features[start_index:end_index])
		}
		// log.Infof("[p3_cache.go] successfully cached %v nodes features.", len(gnid))
		log.Infof("[p3_cache.go] It takes %v to cache these nodes' features.", time.Since(start))
		p3_cache.Feature_dim = int64(shape[1])
	}

	p3_cache.Get_request_num = 0
	p3_cache.Local_hit_num = 0

	p3_cache.Local_feats_gather_time = []float32{}
	p3_cache.Remote_feats_gather_time = []float32{}

	// 计算grpc maxchunksize within 4MB
	p3_cache.MaxChunkSize = getMaxNumDivisibleByXYAnd1024(p3_cache.feat_len, 4)
	log.Infof("[p3_cache.go] max chunk size %v B.", p3_cache.MaxChunkSize)

	return &p3_cache
}

func (p3_cache *P3_cache_mng) Put(idx int64, feature []float32) error {
	p3_cache.Lock()
	defer p3_cache.Unlock()
	p3_cache.cache[idx] = feature // put feature into cache
	p3_cache.Cached_nitems++
	return nil
}

func (p3_cache *P3_cache_mng) Get(ids []int64) ([]byte, error) {
	if !common.Config.Statistic {
		p3_cache.RLock()
		defer p3_cache.RUnlock()

		ret_features := make([]float32, len(ids)*int(p3_cache.feat_len)) // return features

		// 读取本地缓存的数据
		st_local_total := time.Now()
		// server_node_id = DCRuntime.PartIdx
		st_idx, ed_idx := int64(-1), int64(-1)
		var feats []float32
		for st, local_hit_nid := range ids {
			// ret_id = gnid2retid[local_hit_nid]
			st_idx, ed_idx = int64(st)*p3_cache.feat_len,  int64(st+1)*p3_cache.feat_len
			// st := time.Now()
			feats = p3_cache.cache[local_hit_nid]
			// log.Infof("[p3_cache.go] read features from local: %v", time.Since(st)) // 200ns
			// st = time.Now()
			copy(ret_features[st_idx:ed_idx], feats)
			// log.Infof("[p3_cache.go] copy to dest array: %v", time.Since(st)) // copy的时间大概是read local features时间的3倍；450ns-1us
		}
		log.Infof("[p3_cache.go] read features from local and copy to dest array: %v", time.Since(st_local_total)) // 135ms
		byte_ret_features := encodeUnsafe(ret_features) // 113ns
		// log.Infof("[p3_cache.go] convert []float to []byte: %v", time.Since(st))
		return byte_ret_features, nil
	} else {
		// 读写锁
		p3_cache.Lock()
		defer p3_cache.Unlock()
		p3_cache.Get_request_num += int64(len(ids)) // 总请求数增加

		ret_features := make([]float32, len(ids)*int(p3_cache.feat_len)) // return features
		// 将ids中的点根据所在server cache的位置分类

		// 读取本地缓存的数据
		st_local_time := time.Now()
		st_idx, ed_idx:= int64(-1), int64(-1)
		// p3_cache.Local_hit_num += int64(len(ip2ids[server_node_id])) // 本地命中数增加
		// log.Infof("[p3_cache.go] len of ids: %v", len(ids))
		for st, local_hit_nid := range ids {
			st_idx, ed_idx = int64(st)*p3_cache.feat_len, int64(st+1)*p3_cache.feat_len
			copy(ret_features[st_idx:ed_idx], p3_cache.cache[local_hit_nid])
		}
		p3_cache.Local_feats_gather_time = append(p3_cache.Local_feats_gather_time, float32(time.Since(st_local_time)/time.Millisecond))
		// log.Infof("len of ret_features %v after Get", len(ret_features))
		return encodeUnsafe(ret_features), nil
	}
}

func (p3_cache *P3_cache_mng) PeerServerGet(ids []int64) ([]float32, error) {
	// p3_cache.RLock()
	// defer p3_cache.RUnlock()
	ret_features := make([]float32, len(ids)*int(p3_cache.Feature_dim)) // return features
	st_idx, ed_idx := int64(-1), int64(-1)
	i := int64(0)
	st := time.Now()
	for _, local_hit_nid := range ids {
		feature, exist := p3_cache.cache[local_hit_nid]
		if exist {
			st_idx = i * p3_cache.Feature_dim
			ed_idx = (i + 1) * p3_cache.Feature_dim
			copy(ret_features[st_idx:ed_idx], feature)
			i++
		} else {
			log.Fatalf("[p3_cache.go] %v not in cur cache", local_hit_nid)
			return nil, errors.New(string(local_hit_nid) + "not exists in peerserverget.")
		}
	}
	log.Infof("[p3_cache.go] gather features for remote requests: %v", time.Since(st))
	return ret_features, nil
}

func (p3_cache *P3_cache_mng) Get_type() string {
	return "p3_cache"
}

func (p3_cache *P3_cache_mng) Get_feat_dim() int64 {
	return p3_cache.Feature_dim
}

func (p3_cache *P3_cache_mng) Reset() {
	p3_cache.Get_request_num = 0
	p3_cache.Local_hit_num = 0
	p3_cache.Local_feats_gather_time = []float32{}
	p3_cache.Remote_feats_gather_time = []float32{}
}

func (p3_cache *P3_cache_mng) Get_cache_info() (int64, string, int64, int64, []float32, []float32) {
	return DCRuntime.PartIdx, DCRuntime.Curaddr, p3_cache.Get_request_num, p3_cache.Local_hit_num, p3_cache.Local_feats_gather_time, p3_cache.Remote_feats_gather_time
}

func (p3_cache *P3_cache_mng) Get_MaxChunkSize() int64 {
	return p3_cache.MaxChunkSize
}

// func encodeUnsafe(fs []float32) []byte {
// 	return unsafe.Slice((*byte)(unsafe.Pointer(&fs[0])), len(fs)*4)
// }

// func getMaxNumDivisibleByXYAnd1024(x int64, y int64) int64 {
// 	maxNum := int64(4 * 1024 * 1024) // 4MB
// 	for i := maxNum; i > 0; i-- {
// 		if i%x == 0 && i%y == 0 && i%1024 == 0 {
// 			return i
// 		}
// 	}
// 	return -1 // If no number found
// }

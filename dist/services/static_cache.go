package services

import (
	"context"
	"errors"
	"fmt"
	"main/common"
	"main/peerclient"
	"main/rpc/cache"
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
	// client requests
	Feature_dim     int64 //feature dimension
	Get_request_num int64 // # of client read requests
	Local_hit_num   int64 // # of local hit requests
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
	log.Infof("[static_cache.go] npy-header: %v\n", r.Header)
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

	static_cache.Feature_dim = int64(shape[1])
	static_cache.Get_request_num = 0
	static_cache.Local_hit_num = 0

	return &static_cache
}

func (static_cache *Static_cache_mng) Put(idx int64, feature []float32) error {
	static_cache.Lock()
	defer static_cache.Unlock()
	static_cache.cache[idx] = feature // put feature into cache
	static_cache.Cached_nitems++
	return nil
}

func (static_cache *Static_cache_mng) Get(ids []int64) ([]float32, error) {
	if !common.Config.Statistic {
		static_cache.RLock()
		defer static_cache.RUnlock()

		ret_features := make([]float32, len(ids)*int(static_cache.Feature_dim)) // return features
		// 将ids中的点根据所在server cache的位置分类
		ip2ids := make(map[int64][]int64)   // key: server node id; value: requests ids
		gnid2retid := make(map[int64]int64) // key: graph node id; value: ret_features location
		server_node_id := int64(-1)

		for retid, nid := range ids {
			server_node_id = DCRuntime.Nid2Pid[nid]
			ip2ids[server_node_id] = append(ip2ids[server_node_id], nid)
			gnid2retid[nid] = int64(retid)
		}
		// 读取本地缓存的数据
		st_local_total := time.Now()
		server_node_id = DCRuntime.PartIdx
		st_idx, ed_idx, ret_id := int64(-1), int64(-1), int64(-1)
		var feats []float32
		for _, local_hit_nid := range ip2ids[server_node_id] {
			ret_id = gnid2retid[local_hit_nid]
			st_idx, ed_idx = ret_id*static_cache.Feature_dim, (ret_id+1)*static_cache.Feature_dim
			// st := time.Now()
			feats = static_cache.cache[local_hit_nid]
			// log.Infof("[static_cache.go] read features from local: %v", time.Since(st)) // 200ns
			// st = time.Now()
			copy(ret_features[st_idx:ed_idx], feats)
			// log.Infof("[static_cache.go] copy to dest array: %v", time.Since(st)) // copy的时间大概是read local features时间的3倍；450ns-1us
		}
		log.Infof("[static_cache.go] read features from local and copy to dest array: %v", time.Since(st_local_total)) // 135ms

		// 发起peerServerGet请求(从本地id开始++并循回)，将收到的结果进行整理，得到ids的feature，返回结果
		total_server := int64(len(DCRuntime.Ip_slice))
		gnid := int64(-1)
		src_st_idx, src_ed_idx, dst_st_idx, dst_ed_idx := int64(-1), int64(-1), int64(-1), int64(-1)
		for i := int64(0); i < total_server-1; i++ {
			server_node_id = (server_node_id + 1) % total_server
			remote_addr := DCRuntime.Ip_slice[server_node_id]
			ctx, channel := context.WithTimeout(context.Background(), time.Hour)
			defer channel()
			st := time.Now()
			ret, err := peerclient.GrpcClients[remote_addr].DCSubmit(ctx, &cache.DCRequest{Type: cache.OpType_get_features_by_peer_server, Ids: ip2ids[server_node_id]})
			if err != nil {
				fmt.Println(err)
				log.Fatalf("[static_cache.go] Calling OpType_get_features_by_peer_server failed.")
			}
			fetched_featurs := ret.GetFeatures()
			log.Infof("[static_cache.go] rpc calling until get ret: %v for %v number of float data, total size:%v MB.", time.Since(st), len(fetched_featurs), 4*len(fetched_featurs)/1024/1024)
			// st = time.Now()

			// log.Infof("[static_cache.go] extract features from ret: %v", time.Since(st))
			// 从ret.GetFeatures()中按static_cache.Feature_dim为单位读取features到ret_features中
			// st = time.Now()
			for j := int64(0); j < int64(len(ip2ids[server_node_id])); j++ {
				src_st_idx = j * static_cache.Feature_dim
				src_ed_idx = (j + 1) * static_cache.Feature_dim
				gnid = ip2ids[server_node_id][j] // cur graph node id
				ret_id = gnid2retid[gnid]
				dst_st_idx = ret_id * static_cache.Feature_dim
				dst_ed_idx = (ret_id + 1) * static_cache.Feature_dim

				copy(ret_features[dst_st_idx:dst_ed_idx], fetched_featurs[src_st_idx:src_ed_idx])
			}
			// log.Infof("[static_cache.go] copy features to local dest array: %v", time.Since(st))
		}
		return ret_features, nil
	} else {
		// 读写锁
		static_cache.Lock()
		defer static_cache.Unlock()
		static_cache.Get_request_num += int64(len(ids)) // 总请求数增加

		ret_features := make([]float32, len(ids)*int(static_cache.Feature_dim)) // return features
		// 将ids中的点根据所在server cache的位置分类
		ip2ids := make(map[int64][]int64)   // key: server node id; value: requests ids
		gnid2retid := make(map[int64]int64) // key: graph node id; value: ret_features location
		server_node_id := int64(-1)

		for retid, nid := range ids {
			server_node_id = DCRuntime.Nid2Pid[nid]
			ip2ids[server_node_id] = append(ip2ids[server_node_id], nid)
			gnid2retid[nid] = int64(retid)
		}
		// 读取本地缓存的数据
		server_node_id = DCRuntime.PartIdx
		st_idx, ed_idx, ret_id := int64(-1), int64(-1), int64(-1)
		static_cache.Local_hit_num += int64(len(ip2ids[server_node_id])) // 本地命中数增加
		for _, local_hit_nid := range ip2ids[server_node_id] {
			ret_id = gnid2retid[local_hit_nid]
			st_idx, ed_idx = ret_id*static_cache.Feature_dim, (ret_id+1)*static_cache.Feature_dim
			copy(ret_features[st_idx:ed_idx], static_cache.cache[local_hit_nid])
		}

		// 发起peerServerGet请求(从本地id开始++并循回)，将收到的结果进行整理，得到ids的feature，返回结果
		total_server := int64(len(DCRuntime.Ip_slice))
		gnid := int64(-1)
		src_st_idx, src_ed_idx, dst_st_idx, dst_ed_idx := int64(-1), int64(-1), int64(-1), int64(-1)
		for i := int64(0); i < total_server-1; i++ {
			server_node_id = (server_node_id + 1) % total_server
			remote_addr := DCRuntime.Ip_slice[server_node_id]
			ctx, channel := context.WithTimeout(context.Background(), time.Hour)
			defer channel()
			ret, err := peerclient.GrpcClients[remote_addr].DCSubmit(ctx, &cache.DCRequest{Type: cache.OpType_get_features_by_peer_server, Ids: ip2ids[server_node_id]})
			if err != nil {
				fmt.Println(err)
				log.Fatalf("[static_cache.go] Calling OpType_get_features_by_peer_server failed.")
			}
			fetched_featurs := ret.GetFeatures()
			// 从ret.GetFeatures()中按static_cache.Feature_dim为单位读取features到ret_features中
			for j := int64(0); j < int64(len(ip2ids[server_node_id])); j++ {
				src_st_idx = j * static_cache.Feature_dim
				src_ed_idx = (j + 1) * static_cache.Feature_dim
				gnid = ip2ids[server_node_id][j] // cur graph node id
				ret_id = gnid2retid[gnid]
				dst_st_idx = ret_id * static_cache.Feature_dim
				dst_ed_idx = (ret_id + 1) * static_cache.Feature_dim

				copy(ret_features[dst_st_idx:dst_ed_idx], fetched_featurs[src_st_idx:src_ed_idx])
			}
		}
		return ret_features, nil
	}
}

func (static_cache *Static_cache_mng) PeerServerGet(ids []int64) ([]float32, error) {
	// static_cache.RLock()
	// defer static_cache.RUnlock()
	ret_features := make([]float32, len(ids)*int(static_cache.Feature_dim)) // return features
	st_idx, ed_idx := int64(-1), int64(-1)
	i := int64(0)
	st := time.Now()
	for _, local_hit_nid := range ids {
		feature, exist := static_cache.cache[local_hit_nid]
		if exist {
			st_idx = i * static_cache.Feature_dim
			ed_idx = (i + 1) * static_cache.Feature_dim
			copy(ret_features[st_idx:ed_idx], feature)
		} else {
			log.Fatalf("%v not in cur cache", local_hit_nid)
			return nil, errors.New(string(local_hit_nid) + "not exists in peerserverget.")
		}
	}
	log.Infof("[static_cache.go] gather features for remote requests: %v", time.Since(st))
	return ret_features, nil
}

func (static_cache *Static_cache_mng) Get_type() string {
	return "static_cache"
}

func (static_cache *Static_cache_mng) Get_feat_dim() int64 {
	return static_cache.Feature_dim
}

func (static_cache *Static_cache_mng) Get_cache_info() (int64, string, int64, int64) {
	return DCRuntime.PartIdx, DCRuntime.Curaddr, static_cache.Get_request_num, static_cache.Local_hit_num
}
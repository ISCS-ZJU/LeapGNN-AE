package rpc

import (
	context "context"
	"fmt"
	"main/rpc/cache"
	"main/services"
	"os"
	"strconv"
	"sync"
	"syscall"
	"time"
	"unsafe"

	log "github.com/sirupsen/logrus"
	grpc "google.golang.org/grpc"
)

const MAX_CHUNK_SIZE int = 4 * 1024 * 1024

const FILENAME_BASE string = "/dev/shm/repgnn_shm"

var cnt int = 0
var mu_cnt sync.Mutex

type dcrpcserver struct {
	cache.UnimplementedOperatorServer
	cache.UnimplementedOperatorFeaturesServer
}

// Op func imple
func (s *dcrpcserver) DCSubmit(ctx context.Context, request *cache.DCRequest) (*cache.DCReply, error) {
	var reply *cache.DCReply
	switch request.Type {
	case cache.OpType_get_features_by_peer_server:
		reply, _ = Grpc_op_imple_get_features_by_peer_server(request)
	case cache.OpType_get_features_by_client:
		reply, _ = Grpc_op_imple_get_features_by_client(request)
	case cache.OpType_get_feature_dim:
		reply, _ = Grpc_op_imple_get_feature_dim(request)
	case cache.OpType_get_cache_info:
		reply, _ = Grpc_op_imple_get_cache_info(request)
	case cache.OpType_get_statistic:
		reply, _ = Grpc_op_imple_get_statistic(request)
	case cache.OpType_reset:
		reply, _ = Grpc_op_imple_reset(request)
		cnt = 0
		log.Infof("Reset")
	}

	return reply, nil
}

func ChunkBytes(b []byte, chunkSize int) [][]byte {
	var chunks [][]byte
	for len(b) > 0 {
		// log.Infof("len(b)=%v", len(b))
		if len(b) < chunkSize {
			chunkSize = len(b)
		}
		chunks = append(chunks, b[:chunkSize])
		b = b[chunkSize:]
	}
	return chunks
}

func writeShmFiles(reply_chunks [][]byte, CHUNK_SIZE int) error {
	mu_cnt.Lock()
	defer mu_cnt.Unlock()
	for ckid := 0; ckid < len(reply_chunks); ckid++ {
		FILENAME := FILENAME_BASE + strconv.Itoa(cnt)
		// log.Infof("-> to generate file: %v", FILENAME)
		// 打开共享内存
		var err error
		for {
			_, err = os.Stat(FILENAME)
			if err == nil {
				time.Sleep(time.Second)
			} else {
				break
			}
		}
		fd, err := syscall.Open(FILENAME, syscall.O_CREAT|syscall.O_RDWR, 0600)
		if err != nil {
			fmt.Println(err)
			return err
		}
		defer syscall.Close(fd)

		// Truncate shared memory to the desired size
		shm_size := int64(len(reply_chunks[ckid]))
		err = syscall.Ftruncate(fd, shm_size)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error truncating shared memory: %v\n", err)
			return err
		}

		// 映射到用户空间
		addr, err := syscall.Mmap(fd, 0, int(shm_size), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
		if err != nil {
			panic(err)
		}
		defer syscall.Munmap(addr)

		blockPtr := unsafe.Pointer(uintptr(unsafe.Pointer(&addr[0])))
		copy((*[MAX_CHUNK_SIZE]byte)(blockPtr)[:], reply_chunks[ckid])
		// log.Infof("-> Written file: %v", FILENAME)
		cnt += 1
		// log.Infof("-> cnt+1, now cnt=: %v", cnt)
	}

	return nil
}

// Op func imple
func (s *dcrpcserver) DCSubmitFeatures(request *cache.DCRequest, stream cache.OperatorFeatures_DCSubmitFeaturesServer) error {
	var reply *cache.DCReply
	// fmt.Printf("enter dcsubmit server with cnt: %v\n", cnt)
	switch request.Type {
	case cache.OpType_get_stream_features_by_client:
		reply, _ = Grpc_op_imple_get_stream_features_by_client(request)
	}
	// log.Info("len of reply.Features:", len(reply.Features))
	// ensure max chunk byte size considering feat_dim
	CHUNK_SIZE := int(services.DCRuntime.CacheMng.Get_MaxChunkSize())
	reply_chunks := ChunkBytes(reply.Features, CHUNK_SIZE)
	// fmt.Printf("len(chunk): %v\n", len(reply_chunks))
	// shm_size := int64(len(reply.Features))
	// a go routine to write features into shm
	go writeShmFiles(reply_chunks, CHUNK_SIZE)
	// fmt.Printf("cnt = %v\n", cnt)
	// fmt.Printf("shm_size: %v \n", shm_size)
	return nil
}

func Register(s *grpc.Server) {
	log.Info("[distcache_rpc.go] grpc通信代理服务注册")
	cache.RegisterOperatorServer(s, &dcrpcserver{}) // 进行注册 grpc 通信代理服务
	cache.RegisterOperatorFeaturesServer(s, &dcrpcserver{})
}

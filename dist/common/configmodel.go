package common

type Configmodel struct {
	Rpcport     int // grpc port
	Debug       bool
	Dataset     string // raw dataset path
	Cache_type  string // cache type name
	Cache_group string
	Partition   int // number of partition to cache features
}

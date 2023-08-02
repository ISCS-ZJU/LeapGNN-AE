package common

type Configmodel struct {
	Rpcport     int // grpc port
	Debug       bool
	Dataset     string // raw dataset path
	Cache_type  string // cache type name
	Cache_group string
	Partition   int  // number of partition to cache features
	Statistic   bool // whether statistic hit ratio or not
	Multi_feat_file   bool // feat in one file or each part one feat file
	Partition_type string
}

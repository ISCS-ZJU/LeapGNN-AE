// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        v3.12.1
// source: distcache.proto

package cache

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type OpType int32

const (
	OpType_get_feature_dim             OpType = 0 // client gets feature dim to construct network
	OpType_get_features_by_client      OpType = 1 // client gets feature by many graph node ids
	OpType_get_features_by_peer_server OpType = 2 // peer server gets feature by many graph node ids
	OpType_get_cache_info              OpType = 3 // client gets cache basic info
	OpType_get_statistic               OpType = 4 // get cache server's statistic flag value
)

// Enum value maps for OpType.
var (
	OpType_name = map[int32]string{
		0: "get_feature_dim",
		1: "get_features_by_client",
		2: "get_features_by_peer_server",
		3: "get_cache_info",
		4: "get_statistic",
	}
	OpType_value = map[string]int32{
		"get_feature_dim":             0,
		"get_features_by_client":      1,
		"get_features_by_peer_server": 2,
		"get_cache_info":              3,
		"get_statistic":               4,
	}
)

func (x OpType) Enum() *OpType {
	p := new(OpType)
	*p = x
	return p
}

func (x OpType) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (OpType) Descriptor() protoreflect.EnumDescriptor {
	return file_distcache_proto_enumTypes[0].Descriptor()
}

func (OpType) Type() protoreflect.EnumType {
	return &file_distcache_proto_enumTypes[0]
}

func (x OpType) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use OpType.Descriptor instead.
func (OpType) EnumDescriptor() ([]byte, []int) {
	return file_distcache_proto_rawDescGZIP(), []int{0}
}

type DCRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Type OpType  `protobuf:"varint,1,opt,name=type,proto3,enum=rpc.OpType" json:"type,omitempty"` // request type
	Idx  int64   `protobuf:"varint,2,opt,name=idx,proto3" json:"idx,omitempty"`                   // graph node idx
	Ids  []int64 `protobuf:"varint,3,rep,packed,name=ids,proto3" json:"ids,omitempty"`            // multiple graph node ids
}

func (x *DCRequest) Reset() {
	*x = DCRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distcache_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DCRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DCRequest) ProtoMessage() {}

func (x *DCRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distcache_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DCRequest.ProtoReflect.Descriptor instead.
func (*DCRequest) Descriptor() ([]byte, []int) {
	return file_distcache_proto_rawDescGZIP(), []int{0}
}

func (x *DCRequest) GetType() OpType {
	if x != nil {
		return x.Type
	}
	return OpType_get_feature_dim
}

func (x *DCRequest) GetIdx() int64 {
	if x != nil {
		return x.Idx
	}
	return 0
}

func (x *DCRequest) GetIds() []int64 {
	if x != nil {
		return x.Ids
	}
	return nil
}

type DCReply struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Features    []float32 `protobuf:"fixed32,1,rep,packed,name=features,proto3" json:"features,omitempty"` // feature of one idx
	Featdim     int64     `protobuf:"varint,2,opt,name=featdim,proto3" json:"featdim,omitempty"`           // feature dim
	Partidx     int64     `protobuf:"varint,3,opt,name=partidx,proto3" json:"partidx,omitempty"`           // part id of gnid cached in this server
	Curaddr     string    `protobuf:"bytes,4,opt,name=curaddr,proto3" json:"curaddr,omitempty"`            // server addr
	Requestnum  int64     `protobuf:"varint,5,opt,name=requestnum,proto3" json:"requestnum,omitempty"`     // client requests number
	Localhitnum int64     `protobuf:"varint,6,opt,name=localhitnum,proto3" json:"localhitnum,omitempty"`   // local hit requests number
	Statistic   bool      `protobuf:"varint,7,opt,name=statistic,proto3" json:"statistic,omitempty"`       // whether cache server counts cache miss/hit times
}

func (x *DCReply) Reset() {
	*x = DCReply{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distcache_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DCReply) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DCReply) ProtoMessage() {}

func (x *DCReply) ProtoReflect() protoreflect.Message {
	mi := &file_distcache_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DCReply.ProtoReflect.Descriptor instead.
func (*DCReply) Descriptor() ([]byte, []int) {
	return file_distcache_proto_rawDescGZIP(), []int{1}
}

func (x *DCReply) GetFeatures() []float32 {
	if x != nil {
		return x.Features
	}
	return nil
}

func (x *DCReply) GetFeatdim() int64 {
	if x != nil {
		return x.Featdim
	}
	return 0
}

func (x *DCReply) GetPartidx() int64 {
	if x != nil {
		return x.Partidx
	}
	return 0
}

func (x *DCReply) GetCuraddr() string {
	if x != nil {
		return x.Curaddr
	}
	return ""
}

func (x *DCReply) GetRequestnum() int64 {
	if x != nil {
		return x.Requestnum
	}
	return 0
}

func (x *DCReply) GetLocalhitnum() int64 {
	if x != nil {
		return x.Localhitnum
	}
	return 0
}

func (x *DCReply) GetStatistic() bool {
	if x != nil {
		return x.Statistic
	}
	return false
}

var File_distcache_proto protoreflect.FileDescriptor

var file_distcache_proto_rawDesc = []byte{
	0x0a, 0x0f, 0x64, 0x69, 0x73, 0x74, 0x63, 0x61, 0x63, 0x68, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x12, 0x03, 0x72, 0x70, 0x63, 0x22, 0x50, 0x0a, 0x09, 0x44, 0x43, 0x52, 0x65, 0x71, 0x75,
	0x65, 0x73, 0x74, 0x12, 0x1f, 0x0a, 0x04, 0x74, 0x79, 0x70, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x0e, 0x32, 0x0b, 0x2e, 0x72, 0x70, 0x63, 0x2e, 0x4f, 0x70, 0x54, 0x79, 0x70, 0x65, 0x52, 0x04,
	0x74, 0x79, 0x70, 0x65, 0x12, 0x10, 0x0a, 0x03, 0x69, 0x64, 0x78, 0x18, 0x02, 0x20, 0x01, 0x28,
	0x03, 0x52, 0x03, 0x69, 0x64, 0x78, 0x12, 0x10, 0x0a, 0x03, 0x69, 0x64, 0x73, 0x18, 0x03, 0x20,
	0x03, 0x28, 0x03, 0x52, 0x03, 0x69, 0x64, 0x73, 0x22, 0xd3, 0x01, 0x0a, 0x07, 0x44, 0x43, 0x52,
	0x65, 0x70, 0x6c, 0x79, 0x12, 0x1a, 0x0a, 0x08, 0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x73,
	0x18, 0x01, 0x20, 0x03, 0x28, 0x02, 0x52, 0x08, 0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x73,
	0x12, 0x18, 0x0a, 0x07, 0x66, 0x65, 0x61, 0x74, 0x64, 0x69, 0x6d, 0x18, 0x02, 0x20, 0x01, 0x28,
	0x03, 0x52, 0x07, 0x66, 0x65, 0x61, 0x74, 0x64, 0x69, 0x6d, 0x12, 0x18, 0x0a, 0x07, 0x70, 0x61,
	0x72, 0x74, 0x69, 0x64, 0x78, 0x18, 0x03, 0x20, 0x01, 0x28, 0x03, 0x52, 0x07, 0x70, 0x61, 0x72,
	0x74, 0x69, 0x64, 0x78, 0x12, 0x18, 0x0a, 0x07, 0x63, 0x75, 0x72, 0x61, 0x64, 0x64, 0x72, 0x18,
	0x04, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x63, 0x75, 0x72, 0x61, 0x64, 0x64, 0x72, 0x12, 0x1e,
	0x0a, 0x0a, 0x72, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x6e, 0x75, 0x6d, 0x18, 0x05, 0x20, 0x01,
	0x28, 0x03, 0x52, 0x0a, 0x72, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x6e, 0x75, 0x6d, 0x12, 0x20,
	0x0a, 0x0b, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x68, 0x69, 0x74, 0x6e, 0x75, 0x6d, 0x18, 0x06, 0x20,
	0x01, 0x28, 0x03, 0x52, 0x0b, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x68, 0x69, 0x74, 0x6e, 0x75, 0x6d,
	0x12, 0x1c, 0x0a, 0x09, 0x73, 0x74, 0x61, 0x74, 0x69, 0x73, 0x74, 0x69, 0x63, 0x18, 0x07, 0x20,
	0x01, 0x28, 0x08, 0x52, 0x09, 0x73, 0x74, 0x61, 0x74, 0x69, 0x73, 0x74, 0x69, 0x63, 0x2a, 0x81,
	0x01, 0x0a, 0x06, 0x4f, 0x70, 0x54, 0x79, 0x70, 0x65, 0x12, 0x13, 0x0a, 0x0f, 0x67, 0x65, 0x74,
	0x5f, 0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x5f, 0x64, 0x69, 0x6d, 0x10, 0x00, 0x12, 0x1a,
	0x0a, 0x16, 0x67, 0x65, 0x74, 0x5f, 0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x73, 0x5f, 0x62,
	0x79, 0x5f, 0x63, 0x6c, 0x69, 0x65, 0x6e, 0x74, 0x10, 0x01, 0x12, 0x1f, 0x0a, 0x1b, 0x67, 0x65,
	0x74, 0x5f, 0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x73, 0x5f, 0x62, 0x79, 0x5f, 0x70, 0x65,
	0x65, 0x72, 0x5f, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x10, 0x02, 0x12, 0x12, 0x0a, 0x0e, 0x67,
	0x65, 0x74, 0x5f, 0x63, 0x61, 0x63, 0x68, 0x65, 0x5f, 0x69, 0x6e, 0x66, 0x6f, 0x10, 0x03, 0x12,
	0x11, 0x0a, 0x0d, 0x67, 0x65, 0x74, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x69, 0x73, 0x74, 0x69, 0x63,
	0x10, 0x04, 0x32, 0x36, 0x0a, 0x08, 0x4f, 0x70, 0x65, 0x72, 0x61, 0x74, 0x6f, 0x72, 0x12, 0x2a,
	0x0a, 0x08, 0x44, 0x43, 0x53, 0x75, 0x62, 0x6d, 0x69, 0x74, 0x12, 0x0e, 0x2e, 0x72, 0x70, 0x63,
	0x2e, 0x44, 0x43, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x0c, 0x2e, 0x72, 0x70, 0x63,
	0x2e, 0x44, 0x43, 0x52, 0x65, 0x70, 0x6c, 0x79, 0x22, 0x00, 0x42, 0x0a, 0x5a, 0x08, 0x2e, 0x2f,
	0x3b, 0x63, 0x61, 0x63, 0x68, 0x65, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_distcache_proto_rawDescOnce sync.Once
	file_distcache_proto_rawDescData = file_distcache_proto_rawDesc
)

func file_distcache_proto_rawDescGZIP() []byte {
	file_distcache_proto_rawDescOnce.Do(func() {
		file_distcache_proto_rawDescData = protoimpl.X.CompressGZIP(file_distcache_proto_rawDescData)
	})
	return file_distcache_proto_rawDescData
}

var file_distcache_proto_enumTypes = make([]protoimpl.EnumInfo, 1)
var file_distcache_proto_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_distcache_proto_goTypes = []interface{}{
	(OpType)(0),       // 0: rpc.OpType
	(*DCRequest)(nil), // 1: rpc.DCRequest
	(*DCReply)(nil),   // 2: rpc.DCReply
}
var file_distcache_proto_depIdxs = []int32{
	0, // 0: rpc.DCRequest.type:type_name -> rpc.OpType
	1, // 1: rpc.Operator.DCSubmit:input_type -> rpc.DCRequest
	2, // 2: rpc.Operator.DCSubmit:output_type -> rpc.DCReply
	2, // [2:3] is the sub-list for method output_type
	1, // [1:2] is the sub-list for method input_type
	1, // [1:1] is the sub-list for extension type_name
	1, // [1:1] is the sub-list for extension extendee
	0, // [0:1] is the sub-list for field type_name
}

func init() { file_distcache_proto_init() }
func file_distcache_proto_init() {
	if File_distcache_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_distcache_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*DCRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distcache_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*DCReply); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_distcache_proto_rawDesc,
			NumEnums:      1,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_distcache_proto_goTypes,
		DependencyIndexes: file_distcache_proto_depIdxs,
		EnumInfos:         file_distcache_proto_enumTypes,
		MessageInfos:      file_distcache_proto_msgTypes,
	}.Build()
	File_distcache_proto = out.File
	file_distcache_proto_rawDesc = nil
	file_distcache_proto_goTypes = nil
	file_distcache_proto_depIdxs = nil
}

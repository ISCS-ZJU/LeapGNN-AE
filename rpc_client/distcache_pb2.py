# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: distcache.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x64istcache.proto\x12\x03rpc\"@\n\tDCRequest\x12\x19\n\x04type\x18\x01 \x01(\x0e\x32\x0b.rpc.OpType\x12\x0b\n\x03idx\x18\x02 \x01(\x03\x12\x0b\n\x03ids\x18\x03 \x03(\x03\"w\n\x07\x44\x43Reply\x12\x10\n\x08\x66\x65\x61tures\x18\x01 \x03(\x02\x12\x0f\n\x07\x66\x65\x61tdim\x18\x02 \x01(\x03\x12\x0f\n\x07partidx\x18\x03 \x01(\x03\x12\x0f\n\x07\x63uraddr\x18\x04 \x01(\t\x12\x12\n\nrequestnum\x18\x05 \x01(\x03\x12\x13\n\x0blocalhitnum\x18\x06 \x01(\x03*n\n\x06OpType\x12\x13\n\x0fget_feature_dim\x10\x00\x12\x1a\n\x16get_features_by_client\x10\x01\x12\x1f\n\x1bget_features_by_peer_server\x10\x02\x12\x12\n\x0eget_cache_info\x10\x03\x32\x36\n\x08Operator\x12*\n\x08\x44\x43Submit\x12\x0e.rpc.DCRequest\x1a\x0c.rpc.DCReply\"\x00\x42\nZ\x08./;cacheb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'distcache_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\010./;cache'
  _OPTYPE._serialized_start=211
  _OPTYPE._serialized_end=321
  _DCREQUEST._serialized_start=24
  _DCREQUEST._serialized_end=88
  _DCREPLY._serialized_start=90
  _DCREPLY._serialized_end=209
  _OPERATOR._serialized_start=323
  _OPERATOR._serialized_end=377
# @@protoc_insertion_point(module_scope)
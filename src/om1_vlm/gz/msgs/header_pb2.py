# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: gz/msgs/header.proto
# Protobuf Python Version: 5.29.3
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    3,
    '',
    'gz/msgs/header.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import time_pb2 as gz_dot_msgs_dot_time__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14gz/msgs/header.proto\x12\x07gz.msgs\x1a\x12gz/msgs/time.proto\"l\n\x06Header\x12\x1c\n\x05stamp\x18\x01 \x01(\x0b\x32\r.gz.msgs.Time\x12!\n\x04\x64\x61ta\x18\x02 \x03(\x0b\x32\x13.gz.msgs.Header.Map\x1a!\n\x03Map\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x03(\tB\x1b\n\x0b\x63om.gz.msgsB\x0cHeaderProtosb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'gz.msgs.header_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\013com.gz.msgsB\014HeaderProtos'
  _globals['_HEADER']._serialized_start=53
  _globals['_HEADER']._serialized_end=161
  _globals['_HEADER_MAP']._serialized_start=128
  _globals['_HEADER_MAP']._serialized_end=161
# @@protoc_insertion_point(module_scope)

import math
import time
from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import float32, int32, uint32


@dataclass
class Time(IdlStruct, typename="Time"):
    sec: int32
    nanosec: uint32


@dataclass
class Duration(IdlStruct, typename="Duration"):
    sec: int32
    nanosec: uint32


@dataclass
class Header(IdlStruct, typename="Header"):
    stamp: Time
    frame_id: str


@dataclass
class ColorRGBA(IdlStruct, typename="ColorRGBA"):
    r: float32
    g: float32
    b: float32
    a: float32


@dataclass
class String(IdlStruct, typename="String"):
    data: str


def prepare_header(self) -> Header:
    ts = time.time()
    remainder, seconds = math.modf(ts)
    timestamp = Time(sec=int32(seconds), nanosec=uint32(remainder * 1000000000))
    header = Header(stamp=timestamp, frame_id=str(self.sentence_counter))
    return header

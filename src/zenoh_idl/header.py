import math
import time

from pycdr2.types import int32, uint32

from .std_msgs import Header, Time


def prepare_header(self) -> Header:
    ts = time.time()
    remainder, seconds = math.modf(ts)
    timestamp = Time(sec=int32(seconds), nanosec=uint32(remainder * 1000000000))
    header = Header(stamp=timestamp, frame_id=str(self.sentence_counter))
    return header

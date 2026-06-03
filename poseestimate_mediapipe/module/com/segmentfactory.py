from enum import Enum, auto
from typing import Union
from poseestimate_mediapipe.module.com.segment import BothEndSegment, HandSegment, HeadSegment, FootSegment


class SegmentKind(Enum):
    head = auto()
    hand = auto()
    foot = auto()
    bothend = auto()


class SegmentFactory:
    def create(self,
               segmentname: str,
               length: float,
               ratio: float,
               root: tuple[float, float, float],
               end: tuple[float, float, float]) -> Union[BothEndSegment, HandSegment, HeadSegment, FootSegment]:

        segmentkindid = self._segment_judge(segmentname)

        if SegmentKind.bothend == segmentkindid:
            return BothEndSegment(length, ratio, root, end)
        elif SegmentKind.head == segmentkindid:
            return HeadSegment(length, ratio, root, end)
        elif SegmentKind.hand == segmentkindid:
            return HandSegment(length, ratio, root, end)
        elif SegmentKind.foot == segmentkindid:
            return FootSegment(length, ratio, root, end)

    def _segment_judge(self, name) -> SegmentKind:
        _name = self._segment_name_parse(name)

        if _name == "head":
            return SegmentKind.head
        elif _name == "hand":
            return SegmentKind.hand
        elif _name == "foot":
            return SegmentKind.foot
        else:
            return SegmentKind.bothend

    def _segment_name_parse(self, name) -> str:
        if len(name.split('_')) >= 2:
            return name[2:]
        else:
            return name

from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path
from csv import reader


class Baseframedefine(IntEnum):
    FILENAME = 0
    START = auto()
    END = auto()
    BASE = auto()


@dataclass
class Baseframe():
    dataid: int
    filename: str
    startframe: int
    endframe: int
    baseframe: int


def make_baseframe(dataid: int, csvrow: list[str]) -> Baseframe:
    return Baseframe(dataid,
                     csvrow[Baseframedefine.FILENAME.value],
                     int(csvrow[Baseframedefine.START.value]),
                     int(csvrow[Baseframedefine.END.value]),
                     int(csvrow[Baseframedefine.BASE.value]))


class BaseframeList():

    def __init__(self) -> None:
        self.datalist = []

    def append(self, baseframe: Baseframe) -> None:
        self.datalist.append(baseframe)

    def show(self) -> None:
        """baseframeが格納されたリストの中身をすべて表示する
        """
        for baseframe in self.datalist:
            print(baseframe)

    def extract_baseframe(self, dataid: int) -> Baseframe:
        """baseframeが格納されたリストから, dataidを元にしてbaseframeを抽出する.

        Args:
            dataid (int): baseframeのdataid

        Returns:
            Baseframe: bagファイルRGB動画から見た基準フレームオブジェクト
        """
        return self.datalist[dataid]


def make_baseframelist_from_csv(filepath: str) -> BaseframeList:
    csvpathobj = Path(filepath)

    if not csvpathobj.exists():
        raise FileExistsError

    baseframe_list = BaseframeList()

    with csvpathobj.open(newline='', encoding="utf_8", mode='r') as f_table:
        csvreader = reader(f_table, delimiter=',')
        baseframearrs = [row for row in csvreader]

    for index, baseframerow in enumerate(baseframearrs):
        if index == 0:
            continue

        baseframe_list.append(make_baseframe(index, baseframerow))

    return baseframe_list

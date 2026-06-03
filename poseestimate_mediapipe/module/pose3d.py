from copy import deepcopy
import itertools
import math

from poseestimate_mediapipe.config.poseinfo import Joint
from poseestimate_mediapipe.module.vec2 import vec2

# Pose3D, EmptyPose3Dは同名メソッドを持ち, 同一コンテナ内から一つのコードで呼び出し可能.
# BasePose3Dを継承させ、ダックタイピングではなくポリモーフィズムっぽく
# -> ABCを使用するのはあまり良くないそう・


class Metadata:
    # realsenseのmetadataを操作するクラス.
    # update methodで
    def __init__(self) -> None:
        self.count: float = 0
        self.framecount: float = 0
        self.timestamp: float = 0
        self.backendtime: float = 0

    def update(self, framecount, timestamp, backendtime, count) -> None:
        self.count = count
        self.framecount = framecount
        self.timestamp = timestamp
        self.backendtime = backendtime

    def show(self) -> None:
        print("----  metadata  ----")
        for key in self.metadict.keys():
            print(f"metadata || {key} : {self.metadict[key]}")
        print("--------------------")

    @property
    def metadict(self) -> dict[str, float]:
        return {'count': self.count,
                'framecount': self.framecount,
                'timestamp': self.timestamp,
                'backendtime': self.backendtime,
                }

    def datalist_without_count(self) -> list[float]:
        return [float(self.framecount), self.timestamp, self.backendtime]

    @staticmethod
    def endcolnames() -> list[str]:
        return ['framecount', 'timestamp', 'backendtime']

    @staticmethod
    def startcolname() -> str:
        return 'count'


class BasePose3D:
    # Pythonは抽象クラスを意識しないのでこれを消すか, abcを使うかは考えよう

    def __init__(self) -> None:
        pass

    def get_flat(self) -> list[float]:
        pass

    def set_metadata(self, metadata: Metadata) -> None:
        pass

    def get_flat_data_only(self):
        pass

    def update(self, pose: list[float], markernum: int) -> None:
        pass

    def updatejoint(self, jointnumber: int, location: tuple[float, float, float]) -> None:
        pass


class Pose3D(BasePose3D):
    # TODO 部位とidの対応を記載。intEnumで表現.
    def __init__(self, initpose: list[tuple[float, float, float]]) -> None:
        # initposeがfalse同等であった場合valueErrorを吐き出す.
        if initpose:
            self.pose3d = initpose
        else:
            raise ValueError(
                "Pose3D need list[tuple[float float float]] type data, not None.")

    def get_flat(self) -> list[float]:
        # TODO 要リファクタリング
        data = list(itertools.chain.from_iterable(self.pose3d))
        ret = [self.metadata.count]
        ret.extend(data)
        ret.extend(self.metadata.datalist_without_count())
        return ret

    def get_flat_data_only(self) -> list[float]:
        return list(itertools.chain.from_iterable(self.pose3d))

    def set_metadata(self, metadata: Metadata) -> None:
        self.metadata = metadata

    def update(self, dataonlylist: list[float], markernum: int) -> None:
        self.pose3d = [tuple(dataonlylist[markerindex*3:markerindex*3+3])
                       for markerindex in range(markernum)]

    def joint(self, jointnumber: int) -> tuple[float, float, float]:
        """関節座標のXYZ座標を取得するメソッド

        Args:
            jointnumber (int): Joint Enumのvalueプロパティを使用して引数を決めることを想定している

        Returns:
            tuple[float, float, float]: 推定点のXYZ座標
        """
        return self.pose3d[jointnumber - 1]

    def jointyz(self, jointnumber: int) -> tuple[float, float]:
        """関節座標のXYZ座標を取得するメソッド

        Args:
            jointnumber (int): Joint Enumのvalueプロパティを使用して引数を決めることを想定している

        Returns:
            tuple[float, float, float]: 推定点のXYZ座標
        """
        joint = self.pose3d[jointnumber - 1]
        return (joint[1], joint[2])

    def updatejoint(self, jointnumber: int, location: tuple[float, float, float]) -> None:
        """関節座標のXYZ座標を更新するメソッド

        Args:
            jointnumber (int): Joint Enumのvalueプロパティを使用して引数を決める
            location (tuple[float, float, float]): 更新する推定点のXYZ座標
        """

        self.pose3d[jointnumber - 1] = deepcopy(location)

    def _two_joint_length(self, firstnum: int, secondnum: int) -> float:
        """二つの関節間の長さを計算するメソッド

        Args:
            firstnum (int): intEnum Jointで指定する
            secondnum (int): intEnum Jointで指定する

        Returns:
            float: 関節間の長さを返す
        """
        each_axis_sub = ((firstjoint - secondjoint)
                         for firstjoint, secondjoint in zip(self.joint(firstnum), self.joint(secondnum)))
        square = (value * value for value in each_axis_sub)
        return math.sqrt(sum(square))

    @property
    def nose2shoulderxy_length(self) -> float:
        nose = self.joint(Joint.NOSE.value)
        neck = self.neck
        nose_xy = (nose[0], nose[1])
        neck_xy = (neck[0], neck[1])
        return vec2.distance(nose_xy, neck_xy)
        

    @property
    def bodylength(self) -> float:
        """XY平面身体長さ

        Returns:
            float: 奥行方向は同じ値を持つと仮定した三次元長さの計算
        """
        each_axis_sub = [(shoulder_center - waist_center)
                         for shoulder_center, waist_center in zip(self.neck, self.waist_front)]
        square = (value * value for value in each_axis_sub[:2])
        return math.sqrt(sum(square))

    @property
    def thighlength(self) -> float:
        r_thigh = self._two_joint_length(
            Joint.RIGHT_HIP.value, Joint.RIGHT_KNEE.value)
        l_thigh = self._two_joint_length(
            Joint.LEFT_HIP.value, Joint.LEFT_KNEE.value)
        return (r_thigh + l_thigh) / 2.0

    @property
    def waistlength(self) -> float:
        return self._two_joint_length(Joint.RIGHT_HIP.value, Joint.LEFT_HIP.value)

    @property
    def neck(self) -> tuple[float, float, float]:
        return tuple([(r_shoulder + l_shoulder) / 2.0 for r_shoulder, l_shoulder in zip(self.joint(Joint.RIGHT_SHOULDER.value), self.joint(Joint.LEFT_SHOULDER.value))])

    @property
    def waist_front(self) -> tuple[float, float, float]:
        return tuple([(r_waist + l_waist) / 2.0 for r_waist, l_waist in zip(self.joint(Joint.RIGHT_HIP.value), self.joint(Joint.LEFT_HIP.value))])


class EmptyPose3D(BasePose3D):
    def __init__(self, posenumber: int) -> None:
        self.pose3d = [tuple([-1.0, -1.0, -1.0]) for _ in range(posenumber)]

    def get_flat(self) -> list[float]:
        data = list(itertools.chain.from_iterable(self.pose3d))
        ret = [self.metadata.count]
        ret.extend(data)
        ret.extend(self.metadata.datalist_without_count())
        return ret

    def set_metadata(self, metadata: Metadata) -> None:
        self.metadata = metadata

    def get_flat_data_only(self) -> list[float]:
        return list(itertools.chain.from_iterable(self.pose3d))

    def updatejoint(self, jointnumber: int, location: tuple[float, float, float]) -> None:
        """関節座標のXYZ座標を更新するメソッド

        Args:
            jointnumber (int): Joint Enumのvalueプロパティを使用して引数を決める
            location (tuple[float, float, float]): 更新する推定点のXYZ座標
        """

        self.pose3d[jointnumber - 1] = deepcopy(location)

    def joint(self, jointnumber: int) -> tuple[float, float, float]:
        """関節座標のXYZ座標を取得するメソッド

        Args:
            jointnumber (int): Joint Enumのvalueプロパティを使用して引数を決めることを想定している

        Returns:
            tuple[float, float, float]: 推定点のXYZ座標
        """
        return (-1.0, -1.0, -1.0)

    def jointyz(self, jointnumber: int) -> tuple[float, float]:
        """関節座標のXYZ座標を取得するメソッド

        Args:
            jointnumber (int): Joint Enumのvalueプロパティを使用して引数を決めることを想定している

        Returns:
            tuple[float, float, float]: 推定点のXYZ座標
        """
        return (-1.0, -1.0)

    # 首の推定点を肩から取得するgetter

    @property
    def neck(self) -> tuple[float, float, float]:
        return (-1.0, -1.0, -1.0)

    # 股付近の推定点を腰から取得するgetter
    @property
    def waist_front(self) -> tuple[float, float, float]:
        return (-1.0, -1.0, -1.0)

    @property
    def bodylength(self) -> float:
        print('[Warning] : now calling Emptypose bodylength property. this property absolutely return -1.0.')
        return -1.0

    @property
    def thighlength(self) -> float:
        print('[Warning] : now calling Emptypose thighlength property. this property absolutely return -1.0.')
        return -1.0

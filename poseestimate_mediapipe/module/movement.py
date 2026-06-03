from copy import deepcopy
import csv
from datetime import datetime
import os
from poseestimate_mediapipe.module.pose3d import Metadata, Pose3D, EmptyPose3D
import mediapipe as mp
from itertools import chain
from typing import Union

# 三次元姿勢の時系列データを表現するクラス
# 三次元姿勢の追加や変更、各種情報、変換に責任を持つ


class Movement():
    def __init__(self) -> None:
        self.datastore: list[Union[Pose3D, EmptyPose3D]] = []
        self.landmarknames: list[str] = [
            landmark.name for landmark in mp.solutions.pose.PoseLandmark]
        # self.metadatastore : list[Metadata] = []

    def append_pose(self, pose: Union[Pose3D, EmptyPose3D]) -> None:
        if isinstance(pose, Pose3D) or isinstance(pose, EmptyPose3D):
            self.datastore.append(pose)
        else:
            raise TypeError(
                "can't append posedata to Movement without Pose3D object")

    def get_poselist(self) -> list[list[float]]:
        return [pose3d.get_flat() for pose3d in self.datastore]

    def addmarker(self, newmarkername: str) -> None:
        self.landmarknames.append(newmarkername)

    def updatedatastore(self, index: int, poseobj: Union[Pose3D, EmptyPose3D]) -> None:
        self.datastore[index] = deepcopy(poseobj)

    def fetchframepose(self, framenum: int) -> Union[Pose3D, EmptyPose3D]:
        return self.datastore[framenum]

    # movement writerクラスを作成して、分離する
    def to_csv(self, outdir: str, filename: str, subjectname: str, FPS: str = '30') -> None:

        framenum = len(self.datastore)
        dtnow = datetime.now()
        inforow = [filename,
                   'create_time', dtnow.strftime('%Y_%m_%d_%H_%M_%S'),
                   'FPS', FPS,
                   'Mean FPS', '',
                   'subject', subjectname,
                   'framenum', str(framenum)]

        with open(os.path.join(outdir, f'{filename}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(inforow)
            writer.writerow([])
            writer.writerow(self.poseheader)

            datarows = [pose3d.get_flat() for pose3d in self.datastore]
            writer.writerows(datarows)

    @property
    def poseheader(self) -> list[str]:
        header = [[f'{name}_{axis}' for axis in 'xyz']
                  for name in self.landmarknames]
        headerfront = Metadata.startcolname()
        headerlast = Metadata.endcolnames()
        ret = [headerfront]
        ret.extend(list(chain.from_iterable(header)))
        ret.extend(headerlast)
        return ret

    @property
    def framenumber(self) -> int:
        return len(self.datastore)

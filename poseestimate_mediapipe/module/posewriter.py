from csv import writer
import os
import datetime
import cv2
from tqdm import tqdm
import numpy as np

from poseestimate_mediapipe.module.movement import Movement
from poseestimate_mediapipe.module.poseconverter import Pose2ImgConverter
# movementの記載内容に影響しない様に記述する.


class MovementWriter:
    def __init__(self,
                 movement: Movement,
                 filename: str = '',
                 dirpath: str = '') -> None:
        if not isinstance(movement, Movement):
            raise TypeError("arg 'movement' must be Movement type.")
        self._outfilepth = os.path.join(dirpath, filename)
        self._movement = movement
        dtnow = datetime.datetime.now()
        self._info: list[str] = [filename,
                                 'create time', dtnow.strftime(
                                     '%Y_%m_%d_%H_%M_%S'),
                                 'FPS', '',
                                 'Mean FPS', '',
                                 'subject', '',
                                 'framenum', str(self._movement.framenumber)]

# self._info[7] = str(self._movement.calcmeanfps())

    def set_info(self, subject: str = '', FPS: str = '') -> None:
        self._info[4] = FPS
        self._info[8] = subject

    def output(self) -> None:
        print("pose csv file now writing...")
        with open(self._outfilepth, 'w', newline='') as f:
            posewriter = writer(f)
            posewriter.writerow(self._info)
            posewriter.writerow([])
            posewriter.writerow(self._movement.poseheader)
            posewriter.writerows(self._movement.get_poselist())
        print("end pose csv file writing...")


class Video3DWriter:
    def __init__(self,
                 movement: Movement,
                 filename: str = '',
                 dirpath: str = '',
                 elev_angle: float = 30.0,
                 yaw_angle: float = 30.0) -> None:
        if not isinstance(movement, Movement):
            raise TypeError("arg 'movement' must be Movement type.")
        self._outfilepth = os.path.join(dirpath, filename)
        self._movement = movement
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.converter = Pose2ImgConverter(elev_angle, yaw_angle)

    def videosetting(self, width: int, height: int, fps: float = 30.0):
        self.width = width
        self.height = height

        self.video = cv2.VideoWriter(self._outfilepth,
                                     self.fourcc,
                                     fps, (width, height))

    def _write(self, img) -> None:
        self.video.write(img)

    def write3dpose(self) -> None:
        print("Video writing...")
        for pose3d in tqdm(self._movement.datastore):
            img = self.converter.convert(pose3d)
            img = cv2.resize(img, (self.width, self.height))
            self._write(img)
        print("End writing.")

    def release(self):
        self.video.release()

    def write3dposewithcom(self, comdata: list[tuple[float, float, float]]) -> None:
        print("Video writing...")
        for index, pose3d in tqdm(enumerate(self._movement.datastore)):
            img = self.converter.convert_with_com(pose3d, comdata[index])
            img = cv2.resize(img, (self.width, self.height))
            self._write(img)
        print("End writing.")


class MultiVideoWriter(Video3DWriter):
    def __init__(self,
                 movement: Movement,
                 filename: str = '',
                 dirpath: str = '') -> None:
        if not isinstance(movement, Movement):
            raise TypeError("arg 'movement' must be Movement type.")
        self._outfilepth = os.path.join(dirpath, filename)
        self._movement = movement
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.converter = [Pose2ImgConverter(), Pose2ImgConverter(
            0.0, 0.0), Pose2ImgConverter(0.0, 90.0), Pose2ImgConverter(90)]

    def write3dpose(self) -> None:
        print("Video writing...")
        for pose3d in tqdm(self._movement.datastore):
            imgs = [converter.convert(pose3d) for converter in self.converter]
            resized_imgs = [cv2.resize(
                img, (self.width, self.height)) for img in imgs]

            # 4つの写真を組み合わせる
            img_up = np.hstack((resized_imgs[0], resized_imgs[1]))
            img_down = np.hstack((resized_imgs[2], resized_imgs[3]))
            img = np.vstack((img_up, img_down))

            self._write(img)
        print("End writing.")

    def videosetting(self, width: int, height: int, fps: float = 30.0):
        self.width = width
        self.height = height

        self.video = cv2.VideoWriter(self._outfilepth,
                                     self.fourcc,
                                     fps, (width * 2, height * 2))

from copy import deepcopy
from typing import Union
from poseestimate_mediapipe.module.pose3d import EmptyPose3D, Pose3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3D
import numpy as np
import cv2
import mediapipe as mp


class Pose2ImgConverter:
    def __init__(self, elev_angle: float = 30.0, yaw_angle: float = 30.0) -> None:
        self._fig = plt.figure(figsize=(6, 6))
        self.ax = self._fig.add_subplot(111, projection='3d')
        # self.ax = Axes3D(self._fig) の方が良いか???
        self.connections = mp.solutions.pose_connections.POSE_CONNECTIONS
        self._elev_angle = elev_angle
        self._yaw_angle = yaw_angle

    def _set_lim(self, xlim: tuple[float, float] = (-1.0, 1.0),
                 ylim: tuple[float, float] = (1.2, -0.8),
                 zlim: tuple[float, float] = (1.5, 3.5)):
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(zlim[0], zlim[1])
        self.ax.set_zlim(ylim[0], ylim[1])
        self.ax.set_xlabel("Left - Right Location [m]")
        self.ax.set_ylabel("Front - Back Location [m]")
        self.ax.set_zlabel("Up - Down Location [m]")
        
    # TODO #44 setting colors to each segment

    def set_title(self, graphtitle: str):
        self.ax.set_title(graphtitle, size=20)
    
    # TODO #32 任意の角度で画像を生成できるようにする
    def _set_angle(self) -> None:
        # https://qiita.com/knakajima3027/items/6d3ea41cdef96352cfc2
        # おそらくこれで実装できるはず
        self.ax.view_init(elev=self._elev_angle, azim=self._yaw_angle)

    def convert(self, pose: Union[Pose3D, EmptyPose3D]):
        posedata = pose.pose3d

        self._set_lim()
        self._set_angle()
        self._drawpose(posedata)

        self._fig.canvas.draw()
        im = np.array(self._fig.canvas.renderer.buffer_rgba())
        im = deepcopy(cv2.cvtColor(im, cv2.COLOR_RGBA2BGR))
        plt.cla()
        return im

    def _drawpose(self, posedata) -> None:
        for conn in self.connections:
            line = self._factory3dline(conn[0], conn[1], posedata)
            self.ax.add_line(line)

    def _factory3dline(self, start: int, end: int, posedata) -> art3D.Line3D:
        startpoint = posedata[start]
        endpoint = posedata[end]
        x_list = [startpoint[0], endpoint[0]]
        y_list = [startpoint[1], endpoint[1]]
        z_list = [startpoint[2], endpoint[2]]
        return art3D.Line3D(x_list, z_list, y_list, color='g')

    def convert_with_com(self, pose: Union[Pose3D, EmptyPose3D], com: tuple[float, float, float]):
        posedata = pose.pose3d

        self._set_lim()
        self._drawpose(posedata)
        self._drawcom(com)

        self._fig.canvas.draw()
        im = np.array(self._fig.canvas.renderer.buffer_rgba())
        im = deepcopy(cv2.cvtColor(im, cv2.COLOR_RGBA2BGR))
        plt.cla()
        return im

    def _drawcom(self, com) -> None:
        u = com[0]
        v = com[2]
        r = 0.015
        x = [u - r, u - r, u + r, u + r, u - r]
        y = [com[1] for _ in range(5)]
        z = [v + r, v - r, v - r, v + r, v + r]
        poly = list(zip(x, z, y))
        self.ax.add_collection3d(art3D.Poly3DCollection([poly], color='m'))
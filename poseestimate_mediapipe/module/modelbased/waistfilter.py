import math
import numpy as np
from poseestimate_mediapipe.config.poseinfo import Joint

from poseestimate_mediapipe.module.movement import Movement
from poseestimate_mediapipe.module.pose3d import Pose3D
from poseestimate_mediapipe.module.vec2 import vec2

import traceback


class Waistfilter:
    def __init__(self, basepose: Pose3D, subjectinfo: dict[str, str], movement: Movement, startframe: int, endframe: int) -> None:
        self.basepose = basepose
        self.subjectinfo = subjectinfo
        self.movement = movement
        self.startframe = startframe
        self.endframe = endframe
        # TODO baseframeからの取得と比較し, どちらを使用するか考える.(おそらくbaseframeの方が良いかも). 同時に, 腰の横幅を持っておく.
        self.thighlength = (float(subjectinfo['r_thigh'])/1000.0 + float(subjectinfo['l_thigh'])/1000.0) / 2.0
        self.bodylength = float(subjectinfo['body'])/1000.0 - self.basepose.nose2shoulderxy_length / 2.0

    def _yz_thigh(self, knee_x: float, waist_x: float) -> float:
        return math.sqrt(np.square(self.thighlength) - np.square(knee_x - waist_x))

    def _internal_div_left(self, thigh_yz_len: float, knee_shoulder_len: float) -> float:
        return (knee_shoulder_len*knee_shoulder_len + self.bodylength*self.bodylength - thigh_yz_len*thigh_yz_len) / (2 * knee_shoulder_len)

    def _perpend_length(self, knee_shoulder_left: float) -> float:
        ####return math.sqrt(self.bodylength * self.bodylength - knee_shoulder_left*knee_shoulder_left)
        value = self.bodylength * self.bodylength - knee_shoulder_left * knee_shoulder_left
        ####0未満にならないように丸める
        value = max(value, 0)
        return math.sqrt(value)

    def _judge_triangle(self, knee_shoulder: float, thigh: float) -> bool:
        judge = self.bodylength*self.bodylength - knee_shoulder * \
            knee_shoulder - thigh * thigh
        if judge > 0:
            return True
        else:
            return False

    def process(self) -> None:

        for frame, pose3d in enumerate(self.movement.datastore):

            if not (self.startframe <= frame <= self.endframe):
                continue

            try:
                # jointの抽出
                r_hip_vec3 = pose3d.joint(Joint.RIGHT_HIP.value)
                l_hip_vec3 = pose3d.joint(Joint.LEFT_HIP.value)
                r_knee_vec3 = pose3d.joint(Joint.RIGHT_KNEE.value)
                l_knee_vec3 = pose3d.joint(Joint.LEFT_KNEE.value)

                # process1 大腿骨YZ変換 右と左を計算して、平均を取る
                r_yz_thigh = self._yz_thigh(r_knee_vec3[0], r_hip_vec3[0])
                l_yz_thigh = self._yz_thigh(l_knee_vec3[0], l_hip_vec3[0])
                yz_thigh = (r_yz_thigh + l_yz_thigh) / 2.0
                #print('yz thigh ', yz_thigh)

                # process2 膝 - 肩間長さ計算 膝も肩も平均から
                # vector2の計算
                r_shoulder_vec2 = pose3d.jointyz(Joint.RIGHT_SHOULDER.value)
                l_shoulder_vec2 = pose3d.jointyz(Joint.LEFT_SHOULDER.value)
                r_knee_vec2 = pose3d.jointyz(Joint.RIGHT_KNEE.value)
                l_knee_vec2 = pose3d.jointyz(Joint.LEFT_KNEE.value)

                shoulder_vec2 = vec2.average(r_shoulder_vec2, l_shoulder_vec2)
                knee_vec2 = vec2.average(r_knee_vec2, l_knee_vec2)
                knee_shoulder_distance = vec2.distance(
                    shoulder_vec2, knee_vec2)

                # 一度この状態でmodelbased correctして, 動画化してどうなるかチェックする
                
                if yz_thigh < knee_shoulder_distance - self.bodylength <= self.thighlength:
                    yz_thigh = self.thighlength
                
                elif knee_shoulder_distance >= (yz_thigh + self.bodylength):
                    waist_vec2 = vec2.inner_divine(
                        shoulder_vec2, knee_vec2, self.bodylength, yz_thigh)
                    inner_divine_waist_r = (
                        r_hip_vec3[0], waist_vec2[0], waist_vec2[1])
                    inner_divine_waist_l = (
                        l_hip_vec3[0], waist_vec2[0], waist_vec2[1])
                    pose3d.updatejoint(Joint.RIGHT_HIP.value, inner_divine_waist_r)
                    pose3d.updatejoint(Joint.LEFT_HIP.value, inner_divine_waist_l)
                    continue

                # process3 内分長さ計算
                # 左長さ, 右長さ用の関数
                left_knee_shoulder_dist = self._internal_div_left(
                    yz_thigh, knee_shoulder_distance)
                right_knee_shoulder_dist = knee_shoulder_distance - left_knee_shoulder_dist
                #print('body', self.bodylength, 'knee shoulder', knee_shoulder_distance,
                #      'L knee shoulder', left_knee_shoulder_dist, 'thigh', yz_thigh)

                # process4 垂線の長さ計算
                perpend_length = self._perpend_length(
                    left_knee_shoulder_dist)
                # print(perpend_length)

                # process5 内分点の計算
                perpend_cross_point = vec2.inner_divine(
                    shoulder_vec2, knee_vec2, left_knee_shoulder_dist, right_knee_shoulder_dist)

                # 法線ベクトルの計算
                knee_shoulder_direction = vec2.direc(
                    shoulder_vec2, knee_vec2)
                normal_knee_shoulder = vec2.normal(knee_shoulder_direction)

                # 単位ベクトルの計算
                unit_normal_kn2sh = vec2.unit(normal_knee_shoulder)

                # 垂線の足 - 腰のベクトル計算
                perpend_waist_one = vec2.multiple(
                    unit_normal_kn2sh, perpend_length)
                perpend_waist_two = vec2.multiple(
                    unit_normal_kn2sh, -perpend_length)

                # 候補点の計算
                waist_1 = vec2.add(perpend_cross_point, perpend_waist_one)
                waist_2 = vec2.add(perpend_cross_point, perpend_waist_two)

                # データ訂正
                correct_waist = waist_1 if waist_1[1] > waist_2[1] else waist_2
                correct_waist_r_vec3 = (
                    r_hip_vec3[0], correct_waist[0], correct_waist[1])
                correct_waist_l_vec3 = (
                    l_hip_vec3[0], correct_waist[0], correct_waist[1])

                pose3d.updatejoint(Joint.RIGHT_HIP.value, correct_waist_r_vec3)
                pose3d.updatejoint(Joint.LEFT_HIP.value, correct_waist_l_vec3)

                #print('Right', r_hip_vec3[2], correct_waist_r_vec3[2])
                #print('Left', l_hip_vec3[2], correct_waist_l_vec3[2])

            except Exception as e:
                print(f'Error {e.with_traceback}, {e.args}')
                print(f'Traceback {traceback.format_exc()}')

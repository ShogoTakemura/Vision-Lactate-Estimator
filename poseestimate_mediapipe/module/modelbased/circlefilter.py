from math import pi
import math
from poseestimate_mediapipe.config.poseinfo import Joint
from poseestimate_mediapipe.module.movement import Movement
from poseestimate_mediapipe.module.pose3d import Pose3D


class Circlefilter:

    def __init__(self, subjectinfo: dict[str, str], movement: Movement, basepose: Pose3D) -> None:
        self.subject_info = subjectinfo
        self.movement = movement
        self.basepose = basepose

        # 半径情報のタプルを保持.
        self.head = self._calc_radius('head_circum')
        self.r_shoulder = self._calc_radius(
            'r_shoulder_circum')
        self.l_shoulder = self._calc_radius(
            'l_shoulder_circum')
        self.r_elbow = self._calc_radius('r_elbow_circum')
        self.l_elbow = self._calc_radius('r_elbow_circum')
        self.r_wrist = self._calc_radius('r_hand_circum')
        self.l_wrist = self._calc_radius('l_hand_circum')
        self.r_knee = self._calc_radius('r_knee_circum')
        self.l_knee = self._calc_radius('l_knee_circum')
        #self.r_hip = self._calc_radius('r_thigh_circum')
        #self.l_hip = self._calc_radius('l_thigh_circum')
        self.r_hip = self._calc_hip_depth()
        self.l_hip = self._calc_hip_depth()

        self.radius_index_tuples = [
            (self.head, Joint.NOSE.value),
            (self.r_shoulder, Joint.RIGHT_SHOULDER.value),
            (self.l_shoulder, Joint.LEFT_SHOULDER.value),
            (self.r_elbow, Joint.RIGHT_ELBOW.value),
            (self.l_elbow, Joint.LEFT_ELBOW.value),
            (self.r_wrist, Joint.RIGHT_WRIST.value),
            (self.l_wrist, Joint.LEFT_WRIST.value),
            (self.r_knee, Joint.RIGHT_KNEE.value),
            (self.l_knee, Joint.LEFT_KNEE.value),
            (self.r_hip, Joint.RIGHT_HIP.value),
            (self.l_hip, Joint.LEFT_HIP.value)
        ]

    def _calc_radius(self, jointname: str) -> float:
        return float(self.subject_info[jointname]) / 2 / pi / 1000.0

    def _calc_hip_depth(self) -> float:
        return (float(self.subject_info['body_circum']) / 1000.0 - 2 * self.basepose.waistlength) / 2.0 / math.pi

    def process(self):

        for index, pose3d in enumerate(self.movement.datastore):

            # インデックスをタプルにして,コンストラクタに移動
            # 半径と, インデックス情報を同時に持つタプルをコンストラクタに
            # head_vec3 = pose3d.joint(Joint.NOSE.value)
            for depth_value, joint_index in self.radius_index_tuples:
                origin_vec3 = pose3d.joint(joint_index)
                correct_vec3 = (
                    origin_vec3[0], origin_vec3[1], origin_vec3[2] + depth_value)
                pose3d.updatejoint(joint_index, correct_vec3)
            # print(pose3d.get_flat_data_only()[:9])

            self.movement.updatedatastore(index, pose3d)

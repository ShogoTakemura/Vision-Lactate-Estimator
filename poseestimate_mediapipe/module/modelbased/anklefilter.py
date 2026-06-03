from poseestimate_mediapipe.module.pose3d import Pose3D
from poseestimate_mediapipe.module.movement import Movement
from poseestimate_mediapipe.config.poseinfo import Joint

from poseestimate_mediapipe.module.vec3 import vec3


class Anklefilter:

    def __init__(self, startframe: int, endframe: int, basepose: Pose3D, subjectinfo: dict[str, str]) -> None:
        self.start = startframe
        self.end = endframe
        self.basepose = basepose
        self.subject = subjectinfo

        self.rightankle = basepose.joint(Joint.RIGHT_ANKLE.value)
        self.leftankle = basepose.joint(Joint.LEFT_ANKLE.value)
        self.shoes = float(subjectinfo['shoes']) / 1000.0

    def set_movement(self, movementobj: Movement) -> None:
        self.movement = movementobj
    

    def process(self, endoskeleton_flag: bool) -> None:

        for frame in range(self.start, self.end+1):

            # movementからpose3dフェッチ
            pose3d = self.movement.fetchframepose(frame)

            # pose1個に対してankleの位置を固定
            pose3d.updatejoint(Joint.RIGHT_ANKLE.value, self.rightankle)
            pose3d.updatejoint(Joint.LEFT_ANKLE.value, self.leftankle)

            # 内骨格計算フラグがTrueであれば, 以降も計算する
            if not endoskeleton_flag:
                # 二重に更新作業をしている可能性あり
                self.movement.updatedatastore(frame, pose3d)
                continue

            # かかとの位置計算
            r_toe = pose3d.joint(Joint.RIGHT_FOOT_INDEX.value)
            l_toe = pose3d.joint(Joint.LEFT_FOOT_INDEX.value)

            # xzだけ抽出した足首座標
            r_ankle_xz = (self.rightankle[0], r_toe[1], self.rightankle[2])
            l_ankle_xz = (self.leftankle[0], l_toe[1], self.leftankle[2])

            # 足先 -> 足首への方向ベクトル
            r_toe2ankle = vec3.direc_vec3(r_toe, r_ankle_xz)
            l_toe2ankle = vec3.direc_vec3(l_toe, l_ankle_xz)

            # 単位ベクトル計算
            r_unit_toe2ankle = vec3.unit_vec3(r_toe2ankle)
            l_unit_toe2ankle = vec3.unit_vec3(l_toe2ankle)

            # 足先 -> かかとベクトル
            r_toe2heel = vec3.multi_vec3(r_unit_toe2ankle, self.shoes)
            l_toe2heel = vec3.multi_vec3(l_unit_toe2ankle, self.shoes)

            # かかと位置ベクトル
            r_heel = vec3.add_vec3(r_toe, r_toe2heel)
            l_heel = vec3.add_vec3(l_toe, l_toe2heel)

            # 内骨格位置計算
            r_ankle_inside = vec3.ave_vec3(r_heel, self.rightankle)
            l_ankle_inside = vec3.ave_vec3(l_heel, self.leftankle)

            # pose3d くるぶしデータupdate処理
            pose3d.updatejoint(Joint.RIGHT_ANKLE.value, r_ankle_inside)
            pose3d.updatejoint(Joint.LEFT_ANKLE.value, l_ankle_inside)
            
            # pose3d かかとデータUpdate処理
            pose3d.updatejoint(Joint.RIGHT_HEEL.value, r_heel)
            pose3d.updatejoint(Joint.LEFT_HEEL.value, l_heel)

            # movement updatedatastoreで更新
            # 二重に更新作業をしている可能性あり
            self.movement.updatedatastore(frame, pose3d)

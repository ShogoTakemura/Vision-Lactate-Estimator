import copy
from enum import IntEnum, auto
from poseestimate_mediapipe.module.com.aeestimate import AEEstimate
from poseestimate_mediapipe.module.movement import Movement
from poseestimate_mediapipe.module.com.segmentfactory import SegmentFactory

# Type alias
vec3d = tuple[float, float, float]


class MP_Pose(IntEnum):
    nose = 0
    l_eye_inner = auto()
    l_eye = auto()
    l_eye_outer = auto()
    r_eye_inner = auto()
    r_eye = auto()
    r_eye_outer = auto()
    l_ear = auto()
    r_ear = auto()
    l_mouth = auto()
    r_mouth = auto()
    l_shoulder = auto()
    r_shoulder = auto()
    l_elbow = auto()
    r_elbow = auto()
    l_wrist = auto()
    r_wrist = auto()
    l_pinky = auto()
    r_pinky = auto()
    l_index = auto()
    r_index = auto()
    l_thumb = auto()
    r_thumb = auto()
    l_hip = auto()
    r_hip = auto()
    l_knee = auto()
    r_knee = auto()
    l_ankle = auto()
    r_ankle = auto()
    l_heel = auto()
    r_heel = auto()
    l_foot = auto()
    r_foot = auto()
    neck = auto()
    waist_front = auto()


class Comcalclator:

    # name, root id(intEnum), end id(intEnum), -> nameをファクトリに入れて、segmentの判別
    SEGMENT_RELATIONS = (
        ('head', int(MP_Pose.neck), int(MP_Pose.nose)),
        # TODO #34 bodyの方向をチェックする、どっちがrootかわからない。
        ('body', int(MP_Pose.waist_front), int(MP_Pose.neck)),
        ('r_upper_arm', int(MP_Pose.r_shoulder), int(MP_Pose.r_elbow)),
        ('r_fore_arm', int(MP_Pose.r_elbow), int(MP_Pose.r_wrist)),
        ('r_hand', int(MP_Pose.r_wrist), int(MP_Pose.r_index)),
        ('l_upper_arm', int(MP_Pose.l_shoulder), int(MP_Pose.l_elbow)),
        ('l_fore_arm', int(MP_Pose.l_elbow), int(MP_Pose.l_wrist)),
        ('l_hand', int(MP_Pose.l_wrist), int(MP_Pose.l_index)),
        ('r_thigh', int(MP_Pose.r_hip), int(MP_Pose.r_knee)),
        ('r_crus',  int(MP_Pose.r_knee), int(MP_Pose.r_ankle)),
        ('r_foot', int(MP_Pose.r_ankle), int(MP_Pose.r_foot)),
        ('l_thigh', int(MP_Pose.l_hip), int(MP_Pose.l_knee)),
        ('l_crus', int(MP_Pose.l_knee), int(MP_Pose.l_ankle)),
        ('l_foot', int(MP_Pose.l_ankle), int(MP_Pose.l_foot))
    )

    segment_map = {segment[0]: index for index,
                   segment in enumerate(SEGMENT_RELATIONS)}

    def __init__(self, aeobj: AEEstimate, movement: Movement) -> None:
        self.AECoeff = aeobj
        self.movement = movement

    def translate(self):
        # この関数内で行わず、segmentに入れる形で実装をする可能性が高い.
        # 測定された周囲計測値を使用して、奥行の変更を行う
        # subjectdata内に周囲径を実装.
        # subjectdataとともにComCalclatorに挿入
        # トランザクションデータに入れる.
        # 平行移動を行うか否かをトランザクションデータに格納.
        # 部分質量中心の計算と一緒に平行移動を行う.
        pass

    def run(self):
        # TODO #33 segment, Pose3Dを元にして重心の計算を行う

        # nameからSegmentを作成するFactoryクラスを生成
        factory_segment = SegmentFactory()
        
        # TODO 事前確保 出力用リスト
        self.partcoms: list[list[vec3d]] = []
        self.bodycom: list[vec3d] = []

        partmass_total = sum(self.AECoeff.partmass.values())
        total_mass = partmass_total + self.AECoeff.load  # 推定部分質量合計 + 荷重

        # 可視化用でcomのクラスも作成する.

        # 姿勢データの繰り返し処理
        for _, pose3d in enumerate(self.movement.datastore):
            posedata: list[vec3d] = copy.deepcopy(pose3d.pose3d)
            posedata.append(pose3d.neck)
            posedata.append(pose3d.waist_front)

            partcom: list[vec3d] = []

            # 重み付きcom値が欲しい
            weighted_com = (0.0, 0.0, 0.0)

            
            
            for segmentname, rootindex, endindex in Comcalclator.SEGMENT_RELATIONS:
                # TODO pose3dのデータの平行移動処理

                
                try:
                    # 部位のオブジェクト生成
                    segment = factory_segment.create(
                        segmentname=segmentname,
                        length=float(self.AECoeff.subjectinfo[segmentname]),
                        ratio=self.AECoeff.ex_ratio[segmentname],
                        root=posedata[rootindex],
                        end=posedata[endindex])

                    segment_com = segment.segment_com()
                    weighted_com = self._add(weighted_com, self._weighting(
                        self.AECoeff.partmass[segmentname], segment_com))
                    partcom.append(segment_com)
                
                except Exception as e:
                    partcom.append( (-1.0, -1.0, -1.0) )
                    print('Segment generate error occurs.')
                

            # 荷重の重心位置
            # TODO adjustの値を測定値から持ってくるように
            r_load_com = self.calc_loadcom(posedata[int(MP_Pose.r_wrist)],
                                           posedata[int(MP_Pose.r_shoulder)],
                                           adjust=0.035)
            l_load_com = self.calc_loadcom(posedata[int(MP_Pose.l_wrist)],
                                           posedata[int(MP_Pose.l_shoulder)],
                                           adjust=0.035)
            partcom.append(r_load_com)
            partcom.append(l_load_com)

            # 荷重を重心計算に入れる
            weighted_com = self._add(weighted_com, self._weighting(
                self.AECoeff.load / 2.0, r_load_com))
            weighted_com = self._add(weighted_com, self._weighting(
                self.AECoeff.load / 2.0, l_load_com))

            # 部分重心、身体重心の追加
            self.partcoms.append(copy.deepcopy(partcom))
            self.bodycom.append(self.calc_bodycom(
                weighted_com, total_mass))

    # comの座標重み付け
    def _weighting(self, partweight: float, com: vec3d) -> vec3d:
        return tuple([partweight * com for com in com])

    # comの和を計算
    def _add(self, origin, adding: tuple) -> vec3d:
        return tuple([origin_val + add_val for origin_val, add_val in zip(origin, adding)])

    # 全体重心の計算
    def calc_bodycom(self, weightedCom: vec3d, bodymass: float):
        return tuple([com / bodymass for com in weightedCom])

    def calc_loadcom(self, hand: vec3d, shoulder: vec3d, adjust: float = 0.035) -> vec3d:
        """荷重(バーベル)重心位置の推定

        バーベルはグリップ(手首)の真上にあるため X は hand[x] を使用する。
        Y は手首高さ、Z は手首奥行き + adjust(バーベル半径オフセット)。

        FIX Bug4: 元コードは shoulder[x] を使用していた。
        """
        x, y, z = 0, 1, 2
        return (hand[x], hand[y], hand[z] + adjust)   # FIX: shoulder[x] → hand[x]

    def partcom(self, segment_id: int, axis: int) -> tuple[float, ...]:
        return tuple([self.partcoms[row_ind][segment_id][axis] for row_ind in range(len(self.partcoms))])

    @property
    def comlist(self) -> list[vec3d]:
        return copy.deepcopy(self.bodycom)

    @property
    def partscom(self) -> list[list[vec3d]]:
        return copy.deepcopy(self.partcoms)

    @property
    def partscomheader(self) -> list[str]:
        loadheader = [f'{name}_{axis}' for name in [
            'r_load', 'l_load'] for axis in 'xyz']
        poseheader = [f'{name}_{axis}' for name, _,
                      _ in Comcalclator.SEGMENT_RELATIONS for axis in 'xyz']
        poseheader.extend(loadheader)
        return copy.deepcopy(poseheader)

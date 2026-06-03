import copy
from poseestimate_mediapipe.module.com.aeparam import AEParam


class AEEstimate:
    segment_names = (
        'head',
        'body',
        'r_upper_arm',
        'r_fore_arm',
        'r_hand',
        'l_upper_arm',
        'l_fore_arm',
        'l_hand',
        'r_thigh',
        'r_crus',
        'r_foot',
        'l_thigh',
        'l_crus',
        'l_foot',
    )

    def __init__(self, load: float, mass: float, subjectid: int) -> None:
        # load, massをメンバに
        self.load = load
        self.mass = mass
        self.id = subjectid

    def set_subjectinfo(self, subject_dict: dict[str, str]) -> None:
        self.subjectinfo = copy.deepcopy(subject_dict)

    def calc(self) -> None:
        self.set_ratio()
        self.set_partmass()

    def set_ratio(self) -> None:
        # 阿江の係数選択用性別値の取得
        self.sex = self.subjectinfo['sex']

        # 性別のバリデーション
        if self.sex not in ('F', 'M'):
            print('性別の入力値が間違っています。(F or M)')
            exit(0)

        # 質量中心比をメンバに登録
        self.ratio = AEParam.COM_RATIO[self.sex]
        
        # R, Lに拡張した質量中心比をメンバに登録
        self.expand_ratio()

    def expand_ratio(self) -> None:
        segmentnames_no_RL = map(
            lambda s: s if s in ('head', 'body') else s[2:], AEEstimate.segment_names)
        expand_ratio = (self.ratio[name] for name in segmentnames_no_RL)
        self.ex_ratio = {name: ratio for name, ratio in zip(
            AEEstimate.segment_names, expand_ratio)}

    def set_partmass(self) -> None:
        # 部分長の取得
        segment_length = tuple([float(self.subjectinfo[seg_name])
                               for seg_name in AEEstimate.segment_names])

        # 係数の取得
        segment_namesplit = list(map(lambda s: s if s in (
            'head', 'body') else s[2:], AEEstimate.segment_names))
        a0_generator = (AEParam.PARTMASS_A0[self.sex][name]
                        for name in segment_namesplit)
        a1_generator = (AEParam.PARTMASS_A1[self.sex][name]
                        for name in segment_namesplit)
        a2_generator = (AEParam.PARTMASS_A2[self.sex][name]
                        for name in segment_namesplit)

        # 部分質量を計算
        segment_mass = [self._estimatemass(partlength=length, mass=self.mass, a0=a0, a1=a1, a2=a2) for (
            length, a0, a1, a2) in zip(segment_length, a0_generator, a1_generator, a2_generator)]

        # 部分質量を辞書型で保存
        self.partmass = {name: mass for (name, mass) in zip(
            AEEstimate.segment_names, segment_mass)}

    def _estimatemass(self, partlength: float, mass: float, a0: float, a1: float, a2: float) -> float:
        # 阿江の部分質量推定
        return a0 + a1 * partlength / 1000.0 + a2 * mass

    def show_info(self) -> None:
        print("[ae estimate] load[kg] : ", self.load)
        print("[ae estimate] mass[kg] : ", self.mass)
        print("[ae estimate] id[int] : ", self.id)
        print("[ae estimate] sex[M/F] : ", self.sex)
        print("[ae estimate] ratio : ", self.ratio)
        print("[ae estimate] partmass[kg] : ", self.partmass)
        print("[ae estimate] expand ratio : ", self.ex_ratio)
        print('')

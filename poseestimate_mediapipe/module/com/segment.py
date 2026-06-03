from math import sqrt


class Segment:
    def __init__(self,
                 length: float,
                 ratio: float,
                 root: tuple[float, float, float],
                 end: tuple[float, float, float]) -> None:
        self.length = length
        self.ratio = ratio
        self.root = root
        self.end = end
        self.unit_vec = self._div_vector(self._vector_r2e, self.vec_r2e_magni)

    def trans_plus_root(self,
                        mv_root: tuple[float, float, float]) -> None:
        self.root = tuple(
            [origin + add for origin, add in zip(self.root, mv_root)])

    def trans_plus_end(self,
                       mv_end: tuple[float, float, float]) -> None:
        self.end = tuple(
            [origin + add for origin, add in zip(self.root, mv_end)])

    def _div_vector(self, vector: tuple[float, float, float], divider: float) -> tuple[float, float, float]:
        return tuple([val / divider for val in vector])

    def _magnitude(self, vector: tuple[float, float, float]) -> float:
        sq = (val * val for val in vector)
        return sqrt(sum(sq))

    def _add(self, vector1: tuple[float, float, float], vector2: tuple[float, float, float]) -> tuple[float, float, float]:
        return (vector1[0] + vector2[0], vector1[1] + vector2[1], vector1[2] + vector2[2])

    def _multi_vector(self, vector: tuple[float, float, float], multipled: float) -> tuple[float, float, float]:
        return tuple([val * multipled for val in vector])

    @property
    def _vector_r2e(self) -> tuple[float, float, float]:
        return tuple([end - root for root, end in zip(self.root, self.end)])

    @property
    def root_magni(self) -> float:
        return self._magnitude(self.root)

    @property
    def end_magni(self) -> float:
        return self._magnitude(self.end)

    @property
    def vec_r2e_magni(self) -> float:
        return self._magnitude(self._vector_r2e)


class BothEndSegment(Segment):
    # upper_arm, fore_arm, thigh, crus, body
    # upper_arm, fore_arm, thigh, crus : both R and L

    # 末端、先端の座標を使用して内分する
    def segment_com(self) -> tuple[float, float, float]:
        return tuple([(end - root)*(1 - self.ratio) + root for root, end in zip(self.root, self.end)])


class HeadSegment(Segment):

    # root : neck
    # end : nose
    
    # TODO neckの値をshoulderから計算する。

    def segment_com(self) -> tuple[float, float, float]:

        # TODO 位置の補正(頭部に関しては必須か)

        # 単位ベクトルと長さを使用して頭先端位置を取得
        # 貼るベクトルに関してもできるだけ奥行きを入れたくない(今は入れているが)
        # mm -> m換算する
        headtip = self._multi_vector(self.unit_vec, self.length / 1000.0)

        # 内分する
        head_location = tuple([(end - root)*(1 - self.ratio) +
                               root for root, end in zip(self.root, headtip)])

        x_ind = 0
        y_ind = 1
        z_ind = 2

        return (self.root[x_ind], head_location[y_ind], self.root[z_ind])


class HandSegment(Segment):

    # root : hand
    # end : index finger
    # lengthと人差し指-手首の単位ベクトルから仮想指先ベクトルを作成、ratioを使用して内分する.

    def segment_com(self) -> tuple[float, float, float]:

        handtip = self._multi_vector(self.unit_vec, self.length / 1000.0)

        return tuple([(end - root)*(1 - self.ratio) + root for root, end in zip(self.root, handtip)])


class FootSegment(Segment):

    # root : ankle
    # end  : foot index tip

    # Ankle - foot index tip間のベクトルを作成
    # 作成するベクトルはTip -> Ankle方向,
    # 内分の方向が逆なので注意. Tipから奥にベクトルを張る, Heelのベクトル取得.
    # Z-X位置はHeelのベクトルとfoot indexから決定する.
    # 高さはfoot indexのyを使用

    # TODO 靴を考慮したい.

    def segment_com(self) -> tuple[float, float, float]:
        # tupleの座標を参照するためのローカル変数
        x_ind = 0
        # y_ind = 1
        z_ind = 2

        # 足先から足首への単位ベクトルの生成を行う(Y軸は使用しない)
        foot_xz_vector = (
            self.root[x_ind] - self.end[x_ind], 0, self.root[z_ind] - self.end[z_ind])

        # 大きさで割り、単位ベクトルを計算
        foot_xz_unit = self._div_vector(
            foot_xz_vector, self._magnitude(foot_xz_vector))

        # 足の長さを掛け、足先からかかとまでのベクトルを作成
        # 長さが mm単位のため mに換算
        tip_to_heel = self._multi_vector(foot_xz_unit, self.length / 1000.0)

        # かかとのベクトルを方向ベクトル+位置ベクトルで計算
        heel = self._add(tip_to_heel, self.end)

        # つま先 -> かかとベクトルに対して質量中心比を掛けた値を計算
        return tuple([(heel - tip) * self.ratio + tip for heel, tip in zip(heel, self.end)])

"""
COP (Center of Pressure: 圧力中心) 算出モジュール

理論:
    倒立振子モデルによる COP 算出式 (MediaPipe座標系, Y下向き正)

    鉛直床反力:
        F_floor = M × (g - acc_y)           [N]

    水平方向 COP:
        COP_x = CoM_x + h_com × acc_x / (g - acc_y)
        COP_z = CoM_z + h_com × acc_z / (g - acc_y)

    ここで:
        h_com = ankle_y_mp - com_y_mp   [m]  (CoMの地面からの高さ)
        MediaPipe Y下向き → ankle_y > com_y → h_com > 0

参考:
    Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement.
    Hof, A.L. (2005). Comparison of three methods to estimate the center
        of mass during balance assessment. J Biomech.
"""
from __future__ import annotations
from typing import Union
from poseestimate_mediapipe.config.constants import GRAVITY

# 変更後 (直接定義する)
def calc_accels(comlist, fps: float) -> tuple:
    datalen = len(comlist)
    timespan = 1.0 / fps
    def _accel(data, i, h):
        return (data[i+1] - 2*data[i] + data[i-1]) / (h * h)
    return tuple([
        _accel(comlist, i, timespan) if i not in (0, datalen - 1) else 0.0
        for i in range(datalen)
    ])

# 型エイリアス
vec3d = tuple[float, float, float]
vec2d = tuple[float, float]


class CopCalculator:
    """
    身体重心 (CoM) 座標と足首参照高さから COP を算出するクラス。

    使い方::

        cop_calc = CopCalculator(
            comlist=comcalculator.comlist,
            ankle_y_list=get_ankle_y_list(movement),
            fps=fps,
            composite_mass=mass + load,
        )
        cop_calc.run()

        # CSV出力
        with open('cop.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(cop_calc.cop_header)
            writer.writerows(cop_calc.get_flat())
    """

    def __init__(
        self,
        comlist: list[vec3d],
        ankle_y_list: list[float],
        fps: float,
        composite_mass: float,
        gravity: float = GRAVITY,
    ) -> None:
        """
        Args:
            comlist        : 身体重心座標リスト [(x,y,z), ...]  単位[m] MediaPipe系
            ankle_y_list   : 各フレームの両足首Y座標平均 [m]  (MediaPipe Y下向き正)
            fps            : 映像フレームレート [Hz]
            composite_mass : 総合質量 = body_mass + load [kg]
            gravity        : 重力加速度 [m/s²]  (config から取得推奨)
        """
        if len(comlist) != len(ankle_y_list):
            raise ValueError(
                f"comlist length ({len(comlist)}) != ankle_y_list length ({len(ankle_y_list)})"
            )

        self._comlist = comlist
        self._ankle_y = ankle_y_list
        self._fps = fps
        self._M = composite_mass
        self._g = gravity

        self._cop: list[vec2d] = []
        self._floor_force: list[float] = []
        self._h_com: list[float] = []

    # ----------------------------------------------------------------
    # 算出
    # ----------------------------------------------------------------

    def run(self) -> None:
        """COP・床反力・CoM高さを全フレーム算出"""
        self._cop.clear()
        self._floor_force.clear()
        self._h_com.clear()

        n = len(self._comlist)

        # 時系列を軸ごとに分解
        com_x = tuple(c[0] for c in self._comlist)
        com_y = tuple(c[1] for c in self._comlist)
        com_z = tuple(c[2] for c in self._comlist)

        # 加速度 (中心二階差分, 端点=0)
        acc_x = calc_accels(com_x, self._fps)
        acc_y = calc_accels(com_y, self._fps)
        acc_z = calc_accels(com_z, self._fps)

        for i in range(n):
            # ── CoM高さ ────────────────────────────────────────
            # MediaPipe Y下向き正 → 足首は重心より Y値が大きい
            h = self._ankle_y[i] - com_y[i]
            self._h_com.append(h)

            # ── 有効重力加速度 (分母ゼロガード) ──────────────
            g_eff = self._g - acc_y[i]

            if abs(g_eff) < 0.01 or h <= 0.0:
                # 近特異点: CoM位置をそのまま返す
                self._cop.append((com_x[i], com_z[i]))
                self._floor_force.append(self._M * self._g)
                continue

            # ── COP (倒立振子モデル) ────────────────────────
            cop_x = com_x[i] + h * acc_x[i] / g_eff
            cop_z = com_z[i] + h * acc_z[i] / g_eff

            # ── 鉛直床反力 ──────────────────────────────────
            f_floor = self._M * g_eff

            self._cop.append((cop_x, cop_z))
            self._floor_force.append(f_floor)

    # ----------------------------------------------------------------
    # アクセサ
    # ----------------------------------------------------------------

    @property
    def coplist(self) -> list[vec2d]:
        """COP座標リスト [(COP_x, COP_z), ...]  単位[m] MediaPipe系"""
        return list(self._cop)

    @property
    def floor_force(self) -> list[float]:
        """鉛直床反力リスト [N]"""
        return list(self._floor_force)

    @property
    def com_height(self) -> list[float]:
        """CoM地面高さリスト h_com [m]"""
        return list(self._h_com)

    @property
    def cop_header(self) -> list[str]:
        """CSV ヘッダ"""
        return ['COP_x', 'COP_z', 'FloorForce_y', 'CoM_height']

    def get_flat(self) -> list[tuple[float, float, float, float]]:
        """CSV 出力用フラットデータ: [(cop_x, cop_z, floor_force, h_com), ...]"""
        return [
            (cx, cz, ff, h)
            for (cx, cz), ff, h in zip(self._cop, self._floor_force, self._h_com)
        ]


# ────────────────────────────────────────────────────────────────────
# ヘルパ関数 (calccomprocess.py に追加する)
# ────────────────────────────────────────────────────────────────────

def get_ankle_y_list(movement) -> list[float]:
    """
    両足首 Y 座標の平均を全フレーム取得する (地面参照用)。

    MediaPipe 座標系では Y 下向き正。スクワット動作では足首は
    ほぼ一定の Y 座標を持ち、CoM の地面高さ基準点となる。

    Args:
        movement: poseestimate_mediapipe.module.movement.Movement オブジェクト

    Returns:
        list[float]: 各フレームの (r_ankle_y + l_ankle_y) / 2  [m]
    """
    from poseestimate_mediapipe.module.com.comcalclator import MP_Pose

    r_idx = int(MP_Pose.r_ankle)
    l_idx = int(MP_Pose.l_ankle)

    return [
        (pose3d.joint(r_idx)[1] + pose3d.joint(l_idx)[1]) / 2.0
        for pose3d in movement.datastore
    ]

"""joint_torque_process.py
---------------------------
下肢関節トルク(足首・膝・腰)推定プロセス [experimental]。

squat_core/joint_torque.py の力学計算を、本プロジェクトの既存パイプライン
(COM_WORKSET / modelbasedpickle / com_features) に接続する。

Workflow上の位置づけ:
    Step 6: model based correct pose       -> out/modelbasedpickle/*.pickle
    Step 7: Calculate COM location          -> out/com_features/*.csv (床反力)
    Step 8: Estimate joint torque (このモジュール)

入力:
    config/COM_WORKSET.csv      (filename, load, mass, subject_id, picklepath)
    config/SUBJECTS_DATA.csv    (sex, r_thigh, r_crus, l_thigh, l_crus [mm] 等)
    out/modelbasedpickle/*.pickle (Movement オブジェクト, Step6の出力)
    out/com_features/*.csv      (床反力 Right/Left, Step7の出力)

出力:
    out/joint_torque/{filename}_joint_moment.csv
        列: frame, ankle_moment_r, ankle_moment_l,
            knee_moment_r, knee_moment_l, hip_moment_r, hip_moment_l

注意:
    - squat_core/joint_torque.py のdocstring記載の単純化(水平床反力=0,
      COPは足首-足先関節の中点と仮定)を前提とした一次推定値。研究目的で
      使う前に検証すること。
    - Comcalclator がセグメント生成に失敗したフレームは (-1,-1,-1) の
      ダミー値になる(module/com/comcalclator.py の既知の挙動)。
      ローパスフィルタは非因果的(filtfilt)なため、そのフレーム周辺の
      モーメントが一時的に大きく乱れることがある。出力をそのまま信用
      せず、入力データ(モデルベース補正後pickle)の欠損フレームが
      少ないことを確認すること。
"""

from __future__ import annotations

import csv
import os
import pathlib
import pickle
from configparser import ConfigParser

from scipy import signal as scipy_signal
from squat_core import joint_torque as jt
from squat_core.kinematics import calc_accels
from squat_core.subjects import load_subjects

from poseestimate_mediapipe.config.COM_CONFIG import ComConfig
from poseestimate_mediapipe.config.poseinfo import Joint
from poseestimate_mediapipe.module.com.aeestimate import AEEstimate
from poseestimate_mediapipe.module.com.comcalclator import Comcalclator
from poseestimate_mediapipe.module.movement import Movement

LEG_SIDES = ('r', 'l')

# 2回の中心差分(角度->角速度->角加速度, 位置->速度->加速度)はノイズを
# (フレームレート)^2 倍に増幅するため、微分前後にローパスフィルタを通す。
# 係数は corrector.py (Step 4: correct pose data) と同じ [correct] セクションを再利用し、
# プロジェクト全体でフィルタ設計を一貫させる。
_MIN_FILTFILT_LENGTH = 15


def _build_lowpass(config: ConfigParser, fps: float) -> tuple:
    fn = fps / 2.0
    fp = config.getfloat('correct', 'fp')
    fs = config.getfloat('correct', 'fs')
    gpass = config.getfloat('correct', 'gpass')
    gstop = config.getfloat('correct', 'gstop')
    order, wn = scipy_signal.buttord(fp / fn, fs / fn, gpass, gstop)
    return scipy_signal.butter(order, wn, btype='low')


def _smooth(series: list[float], ba: tuple) -> list[float]:
    if len(series) < _MIN_FILTFILT_LENGTH:
        return series
    b, a = ba
    return list(scipy_signal.filtfilt(b, a, series))


def process(config: ConfigParser) -> None:
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    comworkset_path = os.path.join(
        packagepath, 'config', config.get('com', 'comworkset'))
    subjects_path = os.path.join(
        packagepath, 'config', config.get('com', 'subjectinfopath'))
    comfeaturedir = os.path.join(packagepath, 'out', 'com_features')
    outdir = os.path.join(packagepath, 'out', 'joint_torque')
    os.makedirs(outdir, exist_ok=True)

    if not os.path.exists(comworkset_path):
        print(f"[ERROR] COM_WORKSET.csv が見つかりません: {comworkset_path}")
        return

    subjects = load_subjects(subjects_path)
    fps = config.getfloat('movie', 'fps')

    with open(comworkset_path, newline='', encoding='utf-8-sig') as f:
        rd = csv.reader(f)
        workset_rows = [row for row in rd]

    for workset in workset_rows[1:]:
        if not workset or not workset[ComConfig.picklepath_id]:
            continue

        picklepath = workset[ComConfig.picklepath_id]
        picklefilename = pathlib.Path(picklepath).stem
        print(f'\n[joint_torque] processing : {picklefilename}')

        comfeature_path = os.path.join(comfeaturedir, f'{picklefilename}.csv')
        if not os.path.exists(picklepath) or not os.path.exists(comfeature_path):
            print(f"  [SKIP] 前提ファイル不足 (pickle または com_features csv が無い): {picklefilename}")
            continue

        try:
            moments = _process_one_file(
                workset, picklepath, comfeature_path, subjects, fps, config)
        except Exception as e:
            print(f"  [ERROR] {picklefilename} の処理中にエラー: {e}")
            continue

        out_path = os.path.join(outdir, f'{picklefilename}_joint_moment.csv')
        with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame',
                'ankle_moment_r', 'ankle_moment_l',
                'knee_moment_r', 'knee_moment_l',
                'hip_moment_r', 'hip_moment_l',
            ])
            writer.writerows(moments)
        print(f'  -> 出力: {out_path}')


def _process_one_file(
    workset: list[str],
    picklepath: str,
    comfeature_path: str,
    subjects: list[dict[str, str]],
    fps: float,
    config: ConfigParser,
) -> list[tuple]:
    with open(picklepath, 'rb') as f:
        movement: Movement = pickle.load(f)

    lowpass = _build_lowpass(config, fps)

    # AEEstimate / Comcalclator は calccomprocess と同じ手順で再構築する
    # (CoMセグメント位置・部分質量を再利用するため)。
    load = float(workset[ComConfig.load_id])
    mass = float(workset[ComConfig.mass_id])
    subject_index = int(workset[ComConfig.col_subjectid]) - 1
    subject_info = subjects[subject_index]

    ae = AEEstimate(load, mass, subject_index + 1)
    ae.set_subjectinfo(subject_info)
    ae.calc()
    composite_mass = ae.mass + ae.load
    sex = ae.sex

    comcalculator = Comcalclator(ae, movement)
    comcalculator.run()

    floor_force_r, floor_force_l = _read_floor_forces(comfeature_path)
    n_frames = min(len(movement.datastore), len(floor_force_r))

    inertia = {
        side: {
            'thigh': jt.segment_moment_of_inertia(
                'thigh', sex, composite_mass, float(subject_info[f'{side}_thigh'])),
            'crus': jt.segment_moment_of_inertia(
                'crus', sex, composite_mass, float(subject_info[f'{side}_crus'])),
        }
        for side in LEG_SIDES
    }
    mass_seg = {
        side: {
            'thigh': ae.partmass[f'{side}_thigh'],
            'crus': ae.partmass[f'{side}_crus'],
        }
        for side in LEG_SIDES
    }

    joint_value = {
        'r': {
            'hip': Joint.RIGHT_HIP.value, 'knee': Joint.RIGHT_KNEE.value,
            'ankle': Joint.RIGHT_ANKLE.value, 'toe': Joint.RIGHT_FOOT_INDEX.value,
        },
        'l': {
            'hip': Joint.LEFT_HIP.value, 'knee': Joint.LEFT_KNEE.value,
            'ankle': Joint.LEFT_ANKLE.value, 'toe': Joint.LEFT_FOOT_INDEX.value,
        },
    }
    segment_id = {
        'r': {'thigh': Comcalclator.segment_map['r_thigh'], 'crus': Comcalclator.segment_map['r_crus']},
        'l': {'thigh': Comcalclator.segment_map['l_thigh'], 'crus': Comcalclator.segment_map['l_crus']},
    }

    # 関節座標(YZ)の時系列, 部分重心(YZ)の時系列を事前に抽出
    joint_yz = {
        side: {
            part: [movement.datastore[i].jointyz(joint_value[side][part]) for i in range(n_frames)]
            for part in ('hip', 'knee', 'ankle', 'toe')
        }
        for side in LEG_SIDES
    }
    # 部分重心位置をローパスフィルタで平滑化してから加速度を求める。
    # 中心差分による微分はノイズをfps^2倍に増幅するため、位置・加速度の
    # 両方を平滑化する(corrector.pyのButterworth設計を再利用、二段平滑化は
    # 移植元のButterworth+Kalman二段フィルタ構成に対応)。
    seg_com_y = {
        side: {
            part: _smooth(list(comcalculator.partcom(segment_id[side][part], 1)[:n_frames]), lowpass)
            for part in ('thigh', 'crus')
        }
        for side in LEG_SIDES
    }
    seg_com_z = {
        side: {
            part: _smooth(list(comcalculator.partcom(segment_id[side][part], 2)[:n_frames]), lowpass)
            for part in ('thigh', 'crus')
        }
        for side in LEG_SIDES
    }
    seg_com_accel_y = {
        side: {part: _smooth(calc_accels(seg_com_y[side][part], fps), lowpass) for part in ('thigh', 'crus')}
        for side in LEG_SIDES
    }
    seg_com_accel_z = {
        side: {part: _smooth(calc_accels(seg_com_z[side][part], fps), lowpass) for part in ('thigh', 'crus')}
        for side in LEG_SIDES
    }

    # 大腿・下腿の矢状面角度 -> 角速度 -> 角加速度 (中心差分, squat_core.kinematics と同じ式)
    # atan2の分枝切断対策として、微分前に必ずunwrap_anglesを通す。角度・角加速度の
    # 両方を平滑化する(理由は上記の部分重心と同様)。
    thigh_angle = {
        side: _smooth(jt.unwrap_angles([
            jt.segment_sagittal_angle(joint_yz[side]['hip'][i], joint_yz[side]['knee'][i])
            for i in range(n_frames)
        ]), lowpass)
        for side in LEG_SIDES
    }
    crus_angle = {
        side: _smooth(jt.unwrap_angles([
            jt.segment_sagittal_angle(joint_yz[side]['knee'][i], joint_yz[side]['ankle'][i])
            for i in range(n_frames)
        ]), lowpass)
        for side in LEG_SIDES
    }
    thigh_angular_accel = {side: _smooth(calc_accels(thigh_angle[side], fps), lowpass) for side in LEG_SIDES}
    crus_angular_accel = {side: _smooth(calc_accels(crus_angle[side], fps), lowpass) for side in LEG_SIDES}

    floor_force = {'r': floor_force_r, 'l': floor_force_l}

    rows = []
    for i in range(n_frames):
        row_values: dict[str, float] = {}
        for side in LEG_SIDES:
            ankle_yz = joint_yz[side]['ankle'][i]
            knee_yz = joint_yz[side]['knee'][i]
            hip_yz = joint_yz[side]['hip'][i]
            toe_yz = joint_yz[side]['toe'][i]

            # 単純化: 水平床反力=0, COPは足首-足先(toe)関節の中点と仮定 (joint_torque.py docstring参照)
            grf_yz = (floor_force[side][i], 0.0)
            cop_yz = ((ankle_yz[0] + toe_yz[0]) / 2.0, (ankle_yz[1] + toe_yz[1]) / 2.0)

            ankle_m = jt.ankle_moment_from_grf(ankle_yz, cop_yz, grf_yz)

            knee_m, knee_reaction = jt.knee_moment(
                ankle_moment=ankle_m,
                ankle_yz=ankle_yz,
                knee_yz=knee_yz,
                crus_com_yz=(seg_com_y[side]['crus'][i], seg_com_z[side]['crus'][i]),
                crus_com_accel_yz=(seg_com_accel_y[side]['crus'][i], seg_com_accel_z[side]['crus'][i]),
                crus_mass=mass_seg[side]['crus'],
                crus_inertia=inertia[side]['crus'],
                crus_angular_accel=crus_angular_accel[side][i],
                grf_yz=grf_yz,
            )

            hip_m = jt.hip_moment(
                knee_moment_value=knee_m,
                knee_yz=knee_yz,
                hip_yz=hip_yz,
                thigh_com_yz=(seg_com_y[side]['thigh'][i], seg_com_z[side]['thigh'][i]),
                thigh_com_accel_yz=(seg_com_accel_y[side]['thigh'][i], seg_com_accel_z[side]['thigh'][i]),
                thigh_mass=mass_seg[side]['thigh'],
                thigh_inertia=inertia[side]['thigh'],
                thigh_angular_accel=thigh_angular_accel[side][i],
                knee_reaction_force_yz=knee_reaction,
            )

            row_values[f'ankle_{side}'] = ankle_m
            row_values[f'knee_{side}'] = knee_m
            row_values[f'hip_{side}'] = hip_m

        rows.append((
            i,
            row_values['ankle_r'], row_values['ankle_l'],
            row_values['knee_r'], row_values['knee_l'],
            row_values['hip_r'], row_values['hip_l'],
        ))

    return rows


def _read_floor_forces(comfeature_path: str) -> tuple[list[float], list[float]]:
    """com_features csv から左右床反力[N]の時系列を読み込む(calccomprocess出力)。"""
    floor_r: list[float] = []
    floor_l: list[float] = []
    with open(comfeature_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            floor_r.append(float(row['floor force Right[N]']))
            floor_l.append(float(row['floor force Left[N]']))
    return floor_r, floor_l

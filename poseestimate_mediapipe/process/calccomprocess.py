from configparser import ConfigParser
import glob
import itertools
import os
import pathlib
import pickle
from typing import Union

from poseestimate_mediapipe.module.com.aeestimate import AEEstimate
from poseestimate_mediapipe.module.com.comcalclator import Comcalclator
from poseestimate_mediapipe.module.movement import Movement
import csv
from poseestimate_mediapipe.module.com.copcalculator import CopCalculator, get_ankle_y_list
from poseestimate_mediapipe.config.COM_CONFIG import ComConfig

vec3d = tuple[float, float, float]


def process(config: ConfigParser):

    #############
    ## 初期処理 ##
    #############

    # base dirの取得
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    # correct pickle, comの出力先パスの取得
    modelbasedpickle = os.path.join(packagepath, 'out', 'modelbasedpickle')
    bodycomdir = os.path.join(packagepath, 'out', 'bodycom')
    partscomdir = os.path.join(packagepath, 'out', 'partscom')

    # outpickles dir
    bodycompickledir = os.path.join(packagepath, 'out', 'bodycompickle')
    partscompickledir = os.path.join(packagepath, 'out', 'partscompickle')
    comfeaturedir = os.path.join(packagepath, 'out', 'com_features')
    copdir        = os.path.join(packagepath, 'out', 'cop')          # ← 追加
    os.makedirs(copdir, exist_ok=True)  
    # 使用するpickleファイルの集合を取得
    searchstring = os.path.join(modelbasedpickle, '*.pickle')
    picklefiles = glob.glob(searchstring)

    # 実行先pickleパスを元に、セッティングcsvを作成(更新)
    comworkset_path = os.path.join(
        packagepath, 'config', config.get('com', 'comworkset'))
    subjects_path = os.path.join(
        packagepath, 'config', config.get('com', 'subjectinfopath'))

    # TODO subjectデータのバリデーション

    # subject読み込み (修正箇所1: encoding='utf-8-sig'を追加)
    with open(subjects_path, 'r', newline='', encoding='utf-8-sig') as f:
        csvreader = csv.reader(f)
        subjects_rows = [row for row in csvreader]

    # 被験者データ構造(dict)の作成
    # TODO 関数にする
    subjects_header = subjects_rows[0]
    subjects: list[dict[str, str]] = []

    for index, subject_list in enumerate(subjects_rows):
        # header rowは使用しないのでスルー
        if index == 0:
            continue

        # 被験者辞書データを作成する
        subject = {name: data for (
            name, data) in zip(subjects_header, subject_list)}
        subjects.append(subject)

    ##############################
    ## トランザクションデータ処理 ##
    ##############################

    # TODO COM_WORKSETの設定確認処理, 関数化
    # ユーザーにworksetを記載したか聞く.
    print('Can you update com workset file with using correct pickle ?? (in config/COM_WORKSET.csv)\n')
    mode = input(
        '(Y/N) | if you miss selections, N is selected automatically.')

    if mode in ('Y', 'y'):
        # COM_WORKSETのUpdate処理
        update_comworkset(comworkset_path, picklefiles)
    else:
        # COM_WORKSETのチェック処理
        check_comworkset(comworkset_path)

    # COM_WORKSET読み込み (修正箇所2: encoding='utf-8-sig'を追加)
    with open(comworkset_path, 'r', newline='', encoding='utf-8-sig') as f:
        rd = csv.reader(f)
        workset_list_transaction = [row for row in rd]

    # トランザクションデータ(WORKSET)の処理
    for workset in workset_list_transaction[1:]:
        print('')

        picklepath = workset[ComConfig.picklepath_id]

        picklefilename = pathlib.Path(picklepath).stem
        print(f'Pocesssing file : {picklefilename}')
        print(f'read pickle file : {picklepath}')

        with open(picklepath, 'rb') as f:
            movement_from_pickle: Movement = pickle.load(f)

        # 阿江の推定用オブジェクトのインスタンス化
        ae_estimate_obj = AEEstimate(float(workset[ComConfig.load_id]),
                                     float(workset[ComConfig.mass_id]),
                                     int(workset[ComConfig.col_subjectid]))

        # 被験者idから被験者参照添え字の算出
        subject_index = int(workset[ComConfig.col_subjectid]) - 1

        # 被験者情報セット、阿江の係数計算
        ae_estimate_obj.set_subjectinfo(
            subject_dict=subjects[subject_index])
        ae_estimate_obj.calc()

        ae_estimate_obj.show_info()

        # CoM計算用オブジェクトの作成
        comcalculator = Comcalclator(ae_estimate_obj, movement_from_pickle)
        comcalculator.run()

        # 重心データ取得
        comlist = comcalculator.comlist
        comheader = ('CoM_x', 'CoM_y', 'CoM_z')
        partscom = comcalculator.partscom
        partscomheader = comcalculator.partscomheader

        # TODO 一連のI/Oはワークフローにまとめる
        # 身体重心 csvファイル化
        with open(os.path.join(os.path.abspath(bodycomdir), f'{picklefilename}.csv'), 'w', newline='') as f:
            comwriter = csv.writer(f)
            comwriter.writerow(comheader)
            comwriter.writerows(comlist)

        # 部分重心 csvファイル化
        with open(os.path.join(os.path.abspath(partscomdir), f'{picklefilename}.csv'), 'w', newline='') as f:
            comwriter = csv.writer(f)
            comwriter.writerow(partscomheader)
            for partscomrow in partscom:
                writtendata = itertools.chain.from_iterable(partscomrow)
                comwriter.writerow(writtendata)

        # com pickle化
        with open(os.path.join(bodycompickledir, f'{picklefilename}.pickle'), 'wb') as f:
            pickle.dump(comlist, f)

        # partscom pickle化
        with open(os.path.join(partscompickledir, f'{picklefilename}.pickle'), 'wb') as f:
            pickle.dump(partscom, f)

        # 実装後に関数化する
        # fps, load, massは引数
        # TODO 関数化ないしクラス化する。
        fps = config.getfloat('movie', 'fps')
        load = ae_estimate_obj.load
        mass = ae_estimate_obj.mass
        compositemass = load + mass

        # comのtupleリストをパースし、各座標ごとに取得する
        separate_axis_comlist = parse_axis_from_comvector(comlist)

        velocity_xyz: list[tuple[float, ...]] = []
        accel_xyz: list[tuple[float, ...]] = []
        inertial_f_xyz: list[tuple[float, ...]] = []
        floor_f_xyz: list[tuple[float, ...]] = []

        features_list = [velocity_xyz, accel_xyz, inertial_f_xyz, floor_f_xyz]

        for com in separate_axis_comlist:
            velocity = calc_velocities(com, fps)
            accel = calc_accels(com, fps)
            inertial_force = calc_inertialforces(accel, compositemass)
            floorforce = calc_floorforces(
                inertial_force, compositemass, config.getfloat('floorforce', 'gravity_coeff'))

            velocity_xyz.append(velocity)
            accel_xyz.append(accel)
            inertial_f_xyz.append(inertial_force)
            floor_f_xyz.append(floorforce)

        # TODO リファクタリング

        # distribution floor force
        com_r_foot = Comcalclator.segment_map['r_foot']
        com_l_foot = Comcalclator.segment_map['l_foot']
        x = 0
        distributed_force = distribute_floorforce(
            comcalculator.partcom(com_r_foot, x), comcalculator.partcom(com_l_foot, x), separate_axis_comlist[0], floor_f_xyz[1])
        print(len(partscom[com_r_foot]), len(partscom[com_l_foot]), len(
            separate_axis_comlist[0]), len(floor_f_xyz[1]))
        # バグの原因, parts comの構造にある。 N * [ 16 ]になっているので。 転地して、16個の列データとして扱う。-> comcalclatorに

        excel_out = [[0.0 for _ in range(len(features_list)*len('xyz') + 2)]
                     for _ in range(len(floor_f_xyz[0]))]

        for kind_ind, feature in enumerate(features_list):
            for col_ind, velocity in enumerate(feature):
                for row_ind, data in enumerate(velocity):
                    excel_out[row_ind][3 * kind_ind + col_ind] = data

        for index, force in enumerate(distributed_force):
            excel_out[index][len(features_list)*len('xyz')] = force[0]
            excel_out[index][len(features_list)*len('xyz') + 1] = force[1]

        feature_names = ('Velocity', 'Accel',
                         'Inertia Froce', 'Floor Force(only y)')
        header = [f'{name}_{axis}' for name in feature_names for axis in 'xyz']
        header.extend(('floor force Right[N]', 'floor force Left[N]'))

        with open(os.path.join(comfeaturedir, f'{picklefilename}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(excel_out)
            
        from poseestimate_mediapipe.module.com.copcalculator import CopCalculator, get_ankle_y_list
        ankle_y_list = get_ankle_y_list(movement_from_pickle)
        cop_calc = CopCalculator(
            comlist=comlist,
            ankle_y_list=ankle_y_list,
            fps=fps,
            composite_mass=compositemass,
            gravity=config.getfloat('floorforce', 'gravity_coeff'),
        )
        cop_calc.run()
        with open(os.path.join(copdir, f'{picklefilename}.csv'), 'w', newline='') as f:
            cop_writer = csv.writer(f)
            cop_writer.writerow(cop_calc.cop_header)
            cop_writer.writerows(cop_calc.get_flat())

def distribute_floorforce(r_foot_com: tuple[float], l_foot_com: tuple[float], com: tuple[float, ...], floorforce) -> list[tuple[float, float]]:
    return [calc_dist_force(l_com, r_com, bodycom, force) for r_com, l_com, bodycom, force in zip(r_foot_com, l_foot_com, com, floorforce)]


def calc_dist_force(l_foot_x: float, r_foot_x: float, com_x: float, force: float) -> tuple[float, float]:
    l_partial = abs(l_foot_x - com_x)
    r_partial = abs(r_foot_x - com_x)
    width = l_partial + r_partial
    r_force = force / width * r_partial
    l_force = force / width * l_partial
    return (r_force, l_force)


def parse_axis_from_comvector(comlist: list[vec3d]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    x, y, z = 0, 1, 2
    com_xaxis = tuple([com[x] for com in comlist])
    com_yaxis = tuple([com[y] for com in comlist])
    com_zaxis = tuple([com[z] for com in comlist])
    return (com_xaxis, com_yaxis, com_zaxis)


# FPSはconfigに
# 中心差分 {f(x+h) - f(x-h)} / 2h


def calc_velocities(comlist: Union[list[float], tuple[float]], fps: float) -> tuple[float]:
    datalen = len(comlist)
    timespan = 1.0 / fps
    return tuple([calc_velocity(comlist, i, timespan) if i not in (0, datalen - 1) else 0.0 for i in range(datalen)])


def calc_velocity(comlist: Union[list[float], tuple[float]], index: int, timespan: float) -> float:
    return (comlist[index+1] - comlist[index-1]) / (2.0 * timespan)


# FPSはconfigに
# 中心二階差分 {f(x+h) - 2h + f(x-h)} / h*h


def calc_accel(left, center, right, timespan) -> float:
    return (left - 2 * center + right) / (timespan * timespan)


def calc_accels(comlist: Union[list[float], tuple[float]], fps: float) -> tuple[float]:
    """ 加速度を算出する

    Args:
        comlist (Union[list[float], tuple[float]]): comのY方向値
        fps (float): 動画FPS値

    Returns:
        tuple[float]: 二階中心差分加速度値
    """
    datalen = len(comlist)
    timespan = 1.0 / fps
    return tuple([calc_accel(comlist[i+1], comlist[i], comlist[i-1], timespan) if i not in (0, datalen - 1) else 0.0 for i in range(datalen)])


def calc_floorforces(inertialforce: Union[list[float], tuple[float]], compositemass: float, gravity: float = 9.79) -> tuple[float]:
    """ 時系列床反力リストを取得する. 内部で_calc_floorforceを呼び出し.

    Args:
        inertialforce (Union[list[float], tuple[float]]): 慣性力のlist-like
        compositemass (float): 体重と荷重の合成質量
        gravity (float, optional): 重力加速度. Defaults to 9.79.

    Returns:
        tuple[float]: 床反力のlist-like data
    """
    return tuple([_calc_floorforce(inertial=inertial, compositemass=compositemass, gravity=gravity) for inertial in inertialforce])


def _calc_floorforce(inertial: float, compositemass: float, gravity: float) -> float:
    """各フレームでの床反力計算用関数

    Args:
        inertial (float): 慣性力, 座標軸に対して符号が逆になる
        compositemass (float): 体重と荷重の合成質量
        gravity (float): 重力加速度. calc_floorforcesから引き継がれる.

    Returns:
        float: 床反力値
    """
    return compositemass * gravity - inertial


def calc_inertialforces(accelarray: Union[list[float], tuple[float]], composite_mass: float) -> tuple[float]:
    return tuple([calc_inertialforce(value, composite_mass) for value in accelarray])


def calc_inertialforce(yaccel: float, mass: float) -> float:
    return yaccel * mass


def check_worksetcol(colnumber: int, worksetrow: tuple[str]):
    if not worksetrow[colnumber]:
        print(f'{colnumber + 1} 列目が記載されていません')
        exit(0)


def check_comworkset(worksetpath: str):
    # 修正箇所3: encoding='utf-8-sig'を追加
    with open(worksetpath, 'r', newline='', encoding='utf-8-sig') as f:
        rd = csv.reader(f)
        datarows = [tuple(row) for row in rd]

    workset_without_header = datarows[1:]

    for worksetrow in workset_without_header:

        print(f'check filename: {worksetrow[ComConfig.filename_id]}')

        # TODO subject存在チェック

        check_worksetcol(ComConfig.load_id, worksetrow)
        check_worksetcol(ComConfig.mass_id, worksetrow)
        check_worksetcol(ComConfig.col_subjectid, worksetrow)

    print('入力ファイルのチェック終了')


def update_comworkset(worksetpath: str, picklefiles: list[str]):

    # 修正箇所4: encoding='utf-8-sig'を追加
    with open(worksetpath, 'w', newline='', encoding='utf-8-sig') as f:
        #
        csvwr = csv.writer(f)
        csvwr.writerow(ComConfig.comworkset_header)

        # id, filename, picklepathの更新
        for index, path in enumerate(picklefiles):
            filename = pathlib.Path(path).stem
            pickle_id = index + 1
            csvwr.writerow((pickle_id, filename, '', '', '', path))

    print('COM WORKSETを更新しました')
    print('subject id, load, massを入力してください')
    print('人を対象にしていないpickleは削除してください')
    exit(0)
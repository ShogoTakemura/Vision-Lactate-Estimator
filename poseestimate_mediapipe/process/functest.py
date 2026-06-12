from configparser import ConfigParser
import glob
import os
import pathlib
import pickle
from typing import Union

from poseestimate_mediapipe.module.modelbased.MODELBASED_CONFIG import ModelBasedConfig
from poseestimate_mediapipe.module.movement import Movement
from poseestimate_mediapipe.config.constants import GRAVITY
from squat_core.kinematics import (
    calc_velocities, calc_velocity,
    calc_accels, calc_accel,
    calc_floorforces, _calc_floorforce,
    distribute_floorforce, calc_dist_force,
)
import csv


vec3d = tuple[float, float, float]


def process(config: ConfigParser):

    #############
    ## 初期処理 ##
    #############

    # base dirの取得
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    # correct pickleのパスの取得
    correctpickledir = os.path.join(packagepath, 'out', 'correctpickle')

    # 使用するpickleファイルの集合を取得
    searchstring = os.path.join(correctpickledir, '*.pickle')
    picklefiles = glob.glob(searchstring)

    # 実行先pickleパスを元に、セッティングcsvを作成(更新)
    correctworkset_path = os.path.join(
        packagepath, 'config', config.get('modelbasecorrect', 'correctworkset'))
    subject_csv_path = os.path.join(
        packagepath, 'config', config.get('com', 'subjectinfopath'))

    # 被験者ファイルから被験者データ構造の生成はprocessに関係ないため、関数化している.
    subjects_infodict_list = generate_subjectinfo(subject_csv_path)

    picklenum = 12

    picklepath = picklefiles[picklenum]

    picklefilename = pathlib.Path(picklepath).stem
    print(f'Pocesssing file : {picklefilename}')
    print(f'read pickle file : {picklepath}')

    with open(picklepath, 'rb') as f:
        movement_from_pickle: Movement = pickle.load(f)
        
    basedpose = movement_from_pickle.fetchframepose(698)
    print('neck', basedpose.neck)
    print('waist front', basedpose.waist_front)
    print('body length', basedpose.bodylength)
    print('thigh length', basedpose.thighlength)
    
    


def generate_subjectinfo(subjectinfocsvpath: str) -> list[dict[str, str]]:
    """被験者情報ファイルから被験者辞書型リストを作成する関数

    Args:
        subjectinfocsvpath (str): 被験者情報ファイルのパス

    Returns:
        list[dict[str, str]]: 被験者情報の辞書型データをリストにしたもの
    """
    # subject info 読み込み
    with open(subjectinfocsvpath, 'r', newline='') as f:
        csvreader = csv.reader(f)
        subjects_rows = [row for row in csvreader]

    # 被験者データ構造(dict)の作成
    subjects_header = subjects_rows[0]
    subjects_infos: list[dict[str, str]] = []

    for index, subject_list in enumerate(subjects_rows):
        # header rowは使用しないのでスルー
        if index == 0:
            continue

        # 被験者辞書データを作成する
        subject = {name: data for (
            name, data) in zip(subjects_header, subject_list)}
        subjects_infos.append(subject)

    return subjects_infos


def parse_axis_from_comvector(comlist: list[vec3d]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    x, y, z = 0, 1, 2
    com_xaxis = tuple([com[x] for com in comlist])
    com_yaxis = tuple([com[y] for com in comlist])
    com_zaxis = tuple([com[z] for com in comlist])
    return (com_xaxis, com_yaxis, com_zaxis)


def calc_inertialforces(accelarray: Union[list[float], tuple[float]], composite_mass: float) -> tuple[float]:
    return tuple([calc_inertialforce(value, composite_mass) for value in accelarray])


def calc_inertialforce(yaccel: float, mass: float) -> float:
    return yaccel * mass


def check_worksetcol(colnumber: int, worksetrow: tuple[str]):
    if not worksetrow[colnumber]:
        print(f'{colnumber + 1} 列目が記載されていません')
        exit(0)


def check_correctworkset(worksetpath: str):
    with open(worksetpath, 'r', newline='') as f:
        rd = csv.reader(f)
        datarows = [tuple(row) for row in rd]

    workset_without_header = datarows[1:]

    for worksetrow in workset_without_header:

        print(f'check filename: {worksetrow[ModelBasedConfig.filename]}')

        # TODO subject存在チェック

        check_worksetcol(ModelBasedConfig.subject, worksetrow)
        check_worksetcol(ModelBasedConfig.baseframe, worksetrow)
        check_worksetcol(ModelBasedConfig.startframe, worksetrow)
        check_worksetcol(ModelBasedConfig.endframe, worksetrow)

    print('入力ファイルのチェック終了')


def update_correctworkset(worksetpath: str, picklefiles: list[str]):

    with open(worksetpath, 'w', newline='') as f:

        # csv writer objの生成
        csvwr = csv.writer(f)
        csvwr.writerow(ModelBasedConfig.modelbasedworkset_header)

        # id, filename, picklepathの更新
        for index, path in enumerate(picklefiles):
            filename = pathlib.Path(path).stem
            pickle_id = index + 1
            csvwr.writerow((pickle_id, filename, '', '', '', '', path))

    print('MODELBASED WORKSETを更新しました')
    print('subject id, baseframe, startframe, endframeを入力してください')
    print('人を対象にしていないpickleは削除してください')
    exit(0)

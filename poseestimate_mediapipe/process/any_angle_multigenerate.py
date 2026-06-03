from configparser import ConfigParser
import os
import pickle
from pathlib import Path
from poseestimate_mediapipe.module import posewriter


def process(config: ConfigParser):

    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    outdir = os.path.join(packagepath, 'out', 'movie3D')

    print('Do you want to generate correct pickle pose video?')
    modelbaseflag = input('Y/N (if you dont select Y, start modelbased mode automatucally) : ')

    if modelbaseflag in ('Y','y'):
        pickledir = os.path.join(packagepath, 'out', 'correctpickle')
    else:
        pickledir = os.path.join(packagepath, 'out', 'modelbasedpickle')
    # correct pickle select処理
    correct_pickle_pathobj = Path(pickledir)
    correct_pickle_paths = list(correct_pickle_pathobj.glob('*.pickle'))


    for pickle_path in correct_pickle_paths:

        with open(pickle_path.resolve(), 'rb') as f:
            loadmove3d = pickle.load(f)

        # TODO #42 多視点動画の作成処理の追加
        writer3d = posewriter.Video3DWriter(
            loadmove3d, f"{pickle_path.stem}_side.mp4", outdir, elev_angle=0, yaw_angle=0)
        writer3d.videosetting(config.getint('movie', 'width'), config.getint(
            'movie', 'height'), config.getfloat('movie', 'fps'))
        writer3d.write3dpose()
        writer3d.release()

        writer3d = posewriter.Video3DWriter(
            loadmove3d, f"{pickle_path.stem}_front.mp4", outdir, elev_angle=0, yaw_angle=90)
        writer3d.videosetting(config.getint('movie', 'width'), config.getint(
            'movie', 'height'), config.getfloat('movie', 'fps'))
        writer3d.write3dpose()
        writer3d.release()

        writer3d = posewriter.Video3DWriter(
            loadmove3d, f"{pickle_path.stem}_top.mp4", outdir, elev_angle=90, yaw_angle=0)
        writer3d.videosetting(config.getint('movie', 'width'), config.getint(
            'movie', 'height'), config.getfloat('movie', 'fps'))
        writer3d.write3dpose()
        writer3d.release()


def selection(pathes: list[Path]) -> int:

    print('=================')
    print('correct pickle list')
    print('=================\n')

    for ind, path in enumerate(pathes):
        print(f'number : {ind}, filename : {path.stem}')

    print('=================')

    pathlist_len = len(pathes)
    number_set = set([str(num) for num in range(pathlist_len)])

    number_str = input('Please select file number.\n')

    if number_str not in number_set:
        raise IOError(
            'You have entered an invalid number. Please enter the correct number.')

    select_number = int(number_str)

    return select_number

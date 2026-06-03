from configparser import ConfigParser
import os
import pickle
from poseestimate_mediapipe.module import posewriter
from poseestimate_mediapipe.module.processbag import ProcessBag
from glob import glob


def process(config: ConfigParser):
    # Parentクラス指定
    # 追加処理を分離
    # pickle処理をメソッド化
    # config読み込み
    # movement, path等はメンバに

    bagfolder = os.path.abspath(
        config.get('multiestimate', 'bagfolder'))

    searchstring = os.path.join(bagfolder, '*.bag')

    bagfiles = glob(searchstring)

    for bagfilepath in bagfiles:

        process = ProcessBag(bagfilepath, config=config)
        # TODO config設定他にも存在している場合のためのメソッド
        # process.readconfig()

        process.estimate()
        move3d = process.generate3Dmovement()

        packagepath = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))

        outdir = os.path.join(packagepath, 'out', 'posecsv')
        movdir = os.path.join(packagepath, 'out', 'movie3D')
        pickledir = os.path.join(packagepath, 'out', 'pickle')

        # pickeが存在するときにpickleを使用するかどうかを聞きたい。
        with open(os.path.join(pickledir, f'{process._filename}.pickle'), 'wb') as f:
            pickle.dump(move3d, f)

        with open(os.path.join(pickledir, f'{process._filename}.pickle'), 'rb') as f:
            loadmove3d = pickle.load(f)

        writer = posewriter.MovementWriter(
            loadmove3d, process.csvfilename, outdir)
        writer.set_info(FPS=config.get('DEFAULT', 'FPS'),
                        subject=config.get('DEFAULT', 'subject'))
        writer.output()

        writer3d = posewriter.Video3DWriter(
            loadmove3d, process.mp4filename, movdir)
        writer3d.videosetting(config.getint('movie', 'width'), config.getint(
            'movie', 'height'), config.getfloat('movie', 'fps'))
        writer3d.write3dpose()

        writer3d.release()

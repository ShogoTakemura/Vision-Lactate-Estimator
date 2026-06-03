from configparser import ConfigParser
import os
import pickle
from poseestimate_mediapipe.module import posewriter
from poseestimate_mediapipe.module.processbag import ProcessBag
from poseestimate_mediapipe.module.utils import get_filepath


def process(config: ConfigParser):
    # main対応箇所
    # file i/o
    # config

    bagfile = get_filepath(['bag'], "D://research", "select bagfile")

    if not bagfile:
        print("file not selected. scripts end...")
        exit()

    process = ProcessBag(bagfile, config=config)
    # process.readconfig()

    process.estimate()
    move3d = process.generate3Dmovement()

    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    outdir = os.path.join(packagepath, 'out', 'posecsv')
    pickledir = os.path.join(packagepath, 'out', 'pickle')

    # pickeが存在するときにpickleを使用するかどうかを聞きたい。
    with open(os.path.join(pickledir, f'{process._filename}.pickle'), 'wb') as f:
        pickle.dump(move3d, f)

    with open(os.path.join(pickledir, f'{process._filename}.pickle'), 'rb') as f:
        loadmove3d = pickle.load(f)

    # TODO outpathはconfigを使用して操作する. 読み込んだconfigをopenに突っ込む
    # TODO #16 Posewriterクラス作成
    # TODO #17 csvフォーマット作成
    # TODO #18 metadataクラスの作成と使用

    writer = posewriter.MovementWriter(loadmove3d, process.csvfilename, outdir)
    writer.set_info(FPS=config.get('DEFAULT', 'FPS'), subject=config.get('DEFAULT', 'subject'))
    writer.output()

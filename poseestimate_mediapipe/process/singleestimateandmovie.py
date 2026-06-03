from configparser import ConfigParser
import os
import pickle
from poseestimate_mediapipe.module import posewriter
from poseestimate_mediapipe.module.processbag import ProcessBag
from poseestimate_mediapipe.module.utils import get_filepath


def process(config: ConfigParser):

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

    outdir = os.path.join(packagepath, 'out', 'movie3D')
    os.makedirs(outdir, exist_ok=True)
    pickledir = os.path.join(packagepath, 'out', 'pickle')
    os.makedirs(pickledir, exist_ok=True)

    with open(os.path.join(pickledir, f'{process._filename}.pickle'), 'wb') as f:
        pickle.dump(move3d, f)

    with open(os.path.join(pickledir, f'{process._filename}.pickle'), 'rb') as f:
        loadmove3d = pickle.load(f)

    # TODO #42 多視点動画の作成処理の追加
    writer3d = posewriter.Video3DWriter(
        loadmove3d, process.mp4filename, outdir)
    writer3d.videosetting(config.getint('movie', 'width'), config.getint(
        'movie', 'height'), config.getfloat('movie', 'fps'))
    writer3d.write3dpose()

    writer3d.release()

from configparser import ConfigParser
import glob
import os
import pathlib
import pickle
from poseestimate_mediapipe.module import posewriter
from poseestimate_mediapipe.module.corrector import Corrector
from poseestimate_mediapipe.module.movement import Movement


def process(config: ConfigParser):

    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    movie3Ddir = os.path.join(packagepath, 'out', 'movie3D')
    modelbasedpickledir = os.path.join(packagepath, 'out', 'modelbasedpickle')
    bodycompickledir = os.path.join(packagepath, 'out', 'bodycompickle')

    searchstring = os.path.join(modelbasedpickledir, '*.pickle')
    comsearchstring = os.path.join(bodycompickledir, '*.pickle')

    correctpicklefiles = glob.glob(searchstring)
    compicklefiles = glob.glob(comsearchstring)
    compickledict = { pathlib.Path(path).stem: path for path in compicklefiles }
    
    for picklepath in correctpicklefiles:

        filename = pathlib.Path(picklepath).stem
        print(f'Pocesssing file : {filename}')
        print(f'read pickle file : {picklepath}')
        
        if not filename in compickledict:
            continue
        else:
            with open(compickledict[filename], 'rb') as f:
                comlist = pickle.load(f)

        with open(picklepath, 'rb') as f:
            movement_from_pickle = pickle.load(f)
                 
        writer3d = posewriter.Video3DWriter(
            movement_from_pickle,f'{filename}_com.mp4', movie3Ddir)
        writer3d.videosetting(config.getint('movie', 'width'), config.getint('movie', 'height'), config.getfloat('movie', 'fps'))
        writer3d.write3dposewithcom(comlist)

        writer3d.release()

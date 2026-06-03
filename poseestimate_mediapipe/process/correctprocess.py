from configparser import ConfigParser
import glob
import os
import pathlib
import pickle
import shutil
from poseestimate_mediapipe.module import posewriter
from poseestimate_mediapipe.module.corrector import Corrector
from poseestimate_mediapipe.module.movement import Movement


def process(config: ConfigParser):

    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    # TODO フォルダ名指定もconfig.ini
    outdir = os.path.join(packagepath, 'out', 'correctpose')
    movie3Ddir = os.path.join(packagepath, 'out', 'movie3D')
    pickledir = os.path.join(packagepath, 'out', 'pickle')
    correctpickledir = os.path.join(packagepath, 'out', 'correctpickle')

    # 処理後の振り分け用フォルダの設定と作成
    processed_dir = os.path.join(pickledir, 'processed')
    missing_dir = os.path.join(pickledir, 'missing_data')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(missing_dir, exist_ok=True)

    searchstring = os.path.join(pickledir, '*.pickle')

    picklefiles = glob.glob(searchstring)

    for picklepath in picklefiles:

        filename = pathlib.Path(picklepath).stem
        print(f'Pocesssing file : {filename}')
        print(f'read pickle file : {picklepath}')

        with open(picklepath, 'rb') as f:
            movement_from_pickle = pickle.load(f)

        print(f'start processing {filename} correct')
        corrector_obj = Corrector(movement_from_pickle)
        corrector_obj.setconfig(config=config)
        
        try:
            corrector_obj.run()
            
            # --- 処理成功後の残りの作業 ---
            # TODO movementオブジェクトをなんとか吐かせる。
            # subject対応
            corrector_obj.csvwrite(outdir,
                                   filename,
                                   config.getint('DEFAULT', 'FPS'),
                                   config.get('DEFAULT', 'subject'))
            print(f'end processing {filename} correct')

            corectmovement = corrector_obj.getmovement()

            with open(os.path.join(correctpickledir, f'{filename}_correct.pickle'), 'wb') as f:
                pickle.dump(corectmovement, f)

            if config.getint('correct', 'mode') == 0:
                writer3d = posewriter.Video3DWriter(
                    corectmovement,f'{filename}_correct.mp4', movie3Ddir)
                writer3d.videosetting(config.getint('movie', 'width'), config.getint('movie', 'height'), config.getfloat('movie', 'fps'))
                writer3d.write3dpose()

                writer3d.release()
                
            # 一連の処理が最後まで成功したら、元のpickleファイルを processed へ移動
            shutil.move(picklepath, os.path.join(processed_dir, os.path.basename(picklepath)))
            print(f"✅ 処理完了: {filename}.pickle を processedフォルダへ移動しました。\n")

        except Exception as e:
            # エラー内容を表示して、missing_data へ移動
            print(f"\n⚠️ エラーが発生したため、このファイルをスキップします: {e}")
            shutil.move(picklepath, os.path.join(missing_dir, os.path.basename(picklepath)))
            print(f"❌ 処理失敗: {filename}.pickle を missing_dataフォルダへ移動しました。\n")
            continue
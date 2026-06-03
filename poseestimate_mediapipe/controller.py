import os
import questionary
from questionary import Choice
import configparser

from poseestimate_mediapipe.process.singleestimateandmovie import process as movieprocess
from poseestimate_mediapipe.process.singleprocess import process as singleestimate
from poseestimate_mediapipe.process.multiestimate import process as multiestimate
from poseestimate_mediapipe.process.correctprocess import process as correctdata
from poseestimate_mediapipe.process.calccomprocess import process as calccom
from poseestimate_mediapipe.process.drawcommovie import process as drawcom
from poseestimate_mediapipe.process.generateanyanglemovie import process as anyangle
from poseestimate_mediapipe.process.any_angle_multigenerate import process as multianyangle
from poseestimate_mediapipe.process.modelbasecorrect import process as modelbasecorrect
from poseestimate_mediapipe.process.functest import process as test
from poseestimate_mediapipe.process import pose_analyze_process
from poseestimate_mediapipe.process import pose_plot_process
from poseestimate_mediapipe.process import calculate_work_process

def processcontrol(selectprocess, config: configparser.ConfigParser) -> None:
    processlist = {
        "single bagfile poseestimate": movieprocess,
        "1 & generate 3D graph movie": singleestimate,
        "Generate any angle view movie": anyangle,
        "process bagfiles and generate for each data (take much time)": multiestimate,
        "generate 3D any angle video generator (take much time)": multianyangle,
        "correct pose data (from pickle files)": correctdata,
        "model based correct pose": modelbasecorrect,
        "calculate com location": calccom,
        "generate com movie": drawcom,
        "function test": test,
        "Analyze body angles and posture": pose_analyze_process.process,
        "Visualize posture analysis graphs": pose_plot_process.process,
        "Calculate Squat Work & Database Export": calculate_work_process.process
    }

    # 選択されたプロセスを実行
    if selectprocess in processlist:
        processlist[selectprocess](config)
    else:
        print(f"Process '{selectprocess}' not found.")

def main():
    config_ini = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.ini')
    config_ini.read(config_path, encoding='utf-8')

    # メニューの表示
    selectprocess = questionary.select(
        "Please select process contains.",
        choices=[
            "single bagfile poseestimate",
            "1 & generate 3D graph movie",
            "Generate any angle view movie",
            "process bagfiles and generate for each data (take much time)",
            "generate 3D any angle video generator (take much time)",
            "correct pose data (from pickle files)",
            "model based correct pose",
            "calculate com location",
            "generate com movie",
            "function test",
            "Analyze body angles and posture",
            "Visualize posture analysis graphs",
            "Calculate Squat Work & Database Export",
            "exit"
        ],
        use_shortcuts=True
    ).ask()

    if selectprocess == "exit":
        exit()

    processcontrol(selectprocess, config_ini)
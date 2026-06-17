import os
import questionary
import configparser

# 既存プロセスのインポート
from poseestimate_mediapipe.process.bagfile_to_movie_process import process as bagfiletomovie
from poseestimate_mediapipe.process.singleestimateandmovie import process as movieprocess
from poseestimate_mediapipe.process.singleprocess import process as singleestimate
from poseestimate_mediapipe.process.multiestimate import process as multiestimate
from poseestimate_mediapipe.process.correctprocess import process as correctdata
from poseestimate_mediapipe.process.update_basedframe_id_process import process as updatebasedframeid
from poseestimate_mediapipe.process.calccomprocess import process as calccom
from poseestimate_mediapipe.process.drawcommovie import process as drawcom
from poseestimate_mediapipe.process.generateanyanglemovie import process as anyangle
from poseestimate_mediapipe.process.any_angle_multigenerate import process as multianyangle
from poseestimate_mediapipe.process.modelbasecorrect import process as modelbasecorrect
from poseestimate_mediapipe.process.joint_torque_process import process as jointtorque
from poseestimate_mediapipe.process.functest import process as test
from poseestimate_mediapipe.process import pose_analyze_process
from poseestimate_mediapipe.process import pose_plot_process
from poseestimate_mediapipe.process import calculate_work_process
from poseestimate_mediapipe.process import rep_posture_analyze_process

# 全体のワークフロー上の並び順（Step番号）でメニューを構成する。
# Step 1: bagfile -> 動画書き出し
# Step 2: (手動) frame_viewer_tool でレップ区間を切り出し → 対話メニュー対象外
# Step 3: 3D姿勢推定
# Step 4: pickleファイルの補正
# Step 5: basedframe_id の自動更新 (Step3で生成したBASEFRAMES.csvを反映)
# Step 6: model based correct pose
# Step 7: 重心(COM)計算・動画生成・任意視点動画生成
# Step 8: 解析・可視化・データベース出力
PROCESS_LIST = {
    "Step 1: Export movie from bagfile":                              bagfiletomovie,
    "Step 3: Single bagfile pose estimate":                           movieprocess,
    "Step 3: Single bagfile pose estimate and generate 3D graph movie": singleestimate,
    "Step 3: Batch pose estimate for all bagfiles (takes time)":       multiestimate,
    "Step 4: Correct pose data (from pickle files)":                  correctdata,
    "Step 5: Update basedframe_id from BASEFRAMES.csv":               updatebasedframeid,
    "Step 6: Model based correct pose":                               modelbasecorrect,
    "Step 7: Calculate COM location":                                 calccom,
    "Step 7: Generate COM movie":                                     drawcom,
    "Step 7: Generate any angle view movie":                          anyangle,
    "Step 7: Batch generate any angle view movie (takes time)":       multianyangle,
    "Step 8: Analyze body angles and posture":                        pose_analyze_process.process,
    "Step 8: Visualize posture analysis graphs":                      pose_plot_process.process,
    "Step 8: Calculate Squat Work and Database Export":               calculate_work_process.process,
    "Step 8: Analyze posture per rep":                                rep_posture_analyze_process.process,
    "Step 8: Estimate joint torque (knee / hip, experimental)":       jointtorque,
    "Function test":                                                  test,
}


def processcontrol(selectprocess, config: configparser.ConfigParser) -> None:
    if selectprocess in PROCESS_LIST:
        PROCESS_LIST[selectprocess](config)
    else:
        print(f"Process '{selectprocess}' not found.")


def main():
    config_ini = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.ini')
    config_ini.read(config_path, encoding='utf-8')

    selectprocess = questionary.select(
        "Please select process. (Step number = recommended workflow order)",
        choices=[*PROCESS_LIST.keys(), "Exit"],
        use_shortcuts=True
    ).ask()

    if selectprocess in (None, "Exit"):
        exit()

    processcontrol(selectprocess, config_ini)

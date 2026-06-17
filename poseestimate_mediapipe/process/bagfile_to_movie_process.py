"""bagfile_to_movie_process.py
-------------------------------
RealSenseのbagfileからcolorフレームのみを取り出してmp4へ書き出す処理。

外部リポジトリ frame_viewer_tool/program/gen_mp4_from_bagfile.py の処理を
本プロジェクトの規約 (process(config) インタフェース, config.ini の
[MOVIETOOL] セクション) に合わせて移植したもの。

Workflowにおける位置づけ:
    Step 1: bagfile -> 動画書き出し (このモジュール)
    Step 2: 手動でレップ区間を切り出し (frame_viewer_tool, 対話メニュー対象外)
    Step 3: 3D姿勢推定 (singleestimateandmovie / multiestimate など)
    Step 4: basedframe_id の自動更新 (update_basedframe_id_process)
    Step 5: model based correct pose
"""

import os
import pathlib
import traceback
from configparser import ConfigParser

import cv2
import numpy as np
import pyrealsense2 as rs2
import questionary

from poseestimate_mediapipe.module.utils import get_filepath


def process(config: ConfigParser) -> None:
    mode = questionary.select(
        "Step 1: Export movie from bagfile - select mode",
        choices=[
            "Convert single bagfile",
            "Convert all bagfiles in folder",
            "back",
        ],
    ).ask()

    if mode in (None, "back"):
        return

    out_dir = config.get('MOVIETOOL', 'movie_out_path')
    os.makedirs(out_dir, exist_ok=True)

    if mode == "Convert single bagfile":
        bagfiles_path = config.get('MOVIETOOL', 'bagfiles_path')
        bagfile = get_filepath(['bag'], bagfiles_path, "select bagfile")
        if not bagfile:
            print("file not selected. process end...")
            return
        _convert_bagfiles_to_movie([pathlib.Path(bagfile)], out_dir)
    else:
        bagfolder = pathlib.Path(config.get('MOVIETOOL', 'bagfiles_path'))
        bagfilelist = sorted(bagfolder.glob("*.bag"))
        if not bagfilelist:
            print(f"bagfile not found in : {bagfolder}")
            return
        _convert_bagfiles_to_movie(bagfilelist, out_dir)


def _convert_bagfiles_to_movie(bagfilelist: list[pathlib.Path], out_dir: str) -> None:
    for bagfile_path in bagfilelist:
        mov_path = os.path.join(out_dir, f'{bagfile_path.stem}.mp4')

        if os.path.exists(mov_path):
            print(f"[SKIP] already exists : {mov_path}")
            continue

        print(f"[START] {bagfile_path.name} -> {mov_path}")
        try:
            _export_single_bagfile(str(bagfile_path), mov_path)
            print(f"[DONE] {mov_path}")
        except Exception as ex:
            print(f"[ERROR] {bagfile_path.name} : {ex}")
            traceback.print_exc()


def _export_single_bagfile(bagfile_path: str, mov_path: str) -> None:
    pipeline = rs2.pipeline()
    rs_cfg = rs2.config()
    rs2.config.enable_device_from_file(rs_cfg, bagfile_path, repeat_playback=False)

    profile = pipeline.start(rs_cfg)

    color_intrinsics = rs2.video_stream_profile(
        profile.get_stream(rs2.stream.color)).get_intrinsics()
    frame_width = color_intrinsics.width
    frame_height = color_intrinsics.height

    # 再生用のリアルタイム再生を無効化(高速に全フレームを読み切る)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    device = profile.get_device()
    found_rgb = any(
        s.get_info(rs2.camera_info.name) == 'RGB Camera' for s in device.sensors)
    if not found_rgb:
        pipeline.stop()
        raise RuntimeError("This bagfile has no RGB camera stream.")

    align = rs2.align(rs2.stream.color)

    cv2.namedWindow("bagfile -> movie", cv2.WINDOW_AUTOSIZE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mov_path, fourcc, 30.0, (frame_width, frame_height))

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                # bagfile再生終了
                break

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            writer.write(color_image)
            cv2.imshow("bagfile -> movie", color_image)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                print("interrupted by user.")
                break
    finally:
        writer.release()
        cv2.destroyAllWindows()
        pipeline.stop()

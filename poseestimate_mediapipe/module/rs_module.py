import configparser
from copy import deepcopy
from typing import Union
from poseestimate_mediapipe.module.posepixelutils import pixel_clip, pixel_to_int, check_pixel_invalid
from poseestimate_mediapipe.module import pose3d as p3d
from poseestimate_mediapipe.module import estimater, movement
import pyrealsense2 as rs2
import cv2
import numpy as np


class StereoProcessing():
    # 汎用なクラスを目指す. 他の処理でも組み込める様にする.
    # realsenseカメラとステレオ画像処理を担当する
    # 副作用の排除
    # process.pyを見て、分離を進める.

    # 関係のないこと
    # * metadataの操作やまとめ
    # * posedataの操作やまとめ
    # * リスト処理

    # TODO リファクタリング

    # 1 selfにする必要があるかどうかを分ける.
    # 2 rs_moduleを一度スーパークラスにして、Depth camera, movieで別の実装ができるように抽象化する。
    # とりあえずステレオカメラ用で一つ作成して、その後にVideo用でもう一個作ってみて、比較して抽象化する。
    # 3 runの処理が長い
    # 4 責務が多すぎ、もっと抽象化して細かくする.

    def __init__(self) -> None:

        # create realsense pipeline obj
        self._rspipeline = rs2.pipeline()

        # create realsense config obj
        self._rsconfig = rs2.config()

        # create movement data store obj
        self.move3d = movement.Movement()

    # def corrector_slot(self, corrector: corrector.PoseCorrector):
    #     self.corrector = corrector

    def estimater_slot(self, estimater: estimater.MpEstimater):
        self.estimater = estimater

    def setconfig(self, bagfilepath, config: configparser.ConfigParser) -> None:
        # 各種settingメソッドの呼び出し用になるパブリックなインタフェースを提供する.
        self._setbagfile(bagfilepath)
        self._systemsetting()
        self._filtersetting(config)
        # self._originfiltersetting()

    def _setbagfile(self, bagfilepath) -> None:
        # rsconfigにファイルパスとconfigをセットする
        rs2.config.enable_device_from_file(
            self._rsconfig, bagfilepath, repeat_playback=False)

    def _systemsetting(self) -> None:

        # configを元にprofile objをインスタンス化
        self._rsprofile = self._rspipeline.start(self._rsconfig)

        # 再生用のオブジェクトを生成する
        self._playback = self._rsprofile.get_device().as_playback()
        self._playback.set_real_time(False)

        # カメラパラメータ類の取得
        self._depth_intrinsics = rs2.video_stream_profile(
            self._rsprofile.get_stream(rs2.stream.depth)).get_intrinsics()
        self._color_intrinsics = rs2.video_stream_profile(
            self._rsprofile.get_stream(rs2.stream.color)).get_intrinsics()

        # デバイスオブジェクトの生成
        self.device = self._rsprofile.get_device()

    def _filtersetting(self, config: configparser.ConfigParser) -> None:
        self.dec_filter = rs2.decimation_filter()
        self.thres_fil = rs2.threshold_filter(config.getfloat('rsfilter', 'thres_min') , config.getfloat('rsfilter', 'thres_max'))
        self.depth_to_disparity = rs2.disparity_transform(True)
        self.disparity_to_depth = rs2.disparity_transform(False)
        self.spat_filter = rs2.spatial_filter()
        self.temp_filter = rs2.temporal_filter()
        
        self.dec_filter.set_option(rs2.option.filter_magnitude, config.getint('rsfilter', 'dec_magni'))
        self.spat_filter.set_option(rs2.option.filter_magnitude, config.getint('rsfilter', 'spat_magni'))
        self.spat_filter.set_option(rs2.option.filter_smooth_alpha, config.getfloat('rsfilter', 'spat_alpha'))
        self.spat_filter.set_option(rs2.option.filter_smooth_delta, config.getint('rsfilter', 'spat_delta'))
        self.spat_filter.set_option(rs2.option.holes_fill, config.getint('rsfilter', 'spat_holl'))
        self.temp_filter.set_option(rs2.option.filter_smooth_alpha, config.getfloat('rsfilter', 'temp_alpha'))
        self.temp_filter.set_option(rs2.option.filter_smooth_delta, config.getint('rsfilter', 'temp_delta'))
        self.temp_filter.set_option(rs2.option.holes_fill, config.getint('rsfilter', 'temp_holl'))

        # カメラアライメント用のオブジェクト生成
        align_to = rs2.stream.color
        self._align = rs2.align(align_to)

        # 背景除去用セッティング
        depth_sensor = self._rsprofile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def _originfiltersetting(self):
        # TODO 自分で実装したフィルタなどの設定
        pass

    @property
    def depth_intr(self):
        return self._depth_intrinsics

    @property
    def color_intr(self):
        return self._color_intrinsics

    # Movement クラスの値を値渡しするメソッド
    def generate3Dmovement(self) -> movement.Movement:
        from copy import deepcopy
        # 渡した先での変更を考慮し、deepcopyとする.
        return deepcopy(self.move3d)

    def run(self):
        # 三次元姿勢推定
        # ここだけクラスに切り出せばいいのでは.
        try:
            # counter
            count = 0

            height = self._color_intrinsics.height
            width = self._color_intrinsics.width

            # TODO リファクタリング対象 良い方法を考える.
            markernum = len(self.estimater.poselandmarks)

            while True:
                # パイプラインから撮影したフレームを取得する
                flag,frames = self._rspipeline.try_wait_for_frames()
                if flag == False: break
                # metadataインスタンス化
                meta = p3d.Metadata()

                meta.update(
                    frames.get_frame_number(),
                    frames.get_timestamp(),
                    frames.get_frame_metadata(
                        rs2.frame_metadata_value.backend_timestamp
                    ),
                    count
                )

                # metadataの表示処理
                meta.show()

                # カラーフレームにデプスフレームを合わせる
                aligned_frames = self._align.process(frames)

                # Get depth frame and color frame
                # TODO 画像・デプス, バリデーション
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Colorize depth frame to jet colormap
                color_image_RGB = np.asanyarray(color_frame.get_data())

                # -----------------------------------------------------------
                # フィルタリング処理 (下に続く)
                # -----------------------------------------------------------
                
                depth_frame = self.dec_filter.process(depth_frame)
                depth_frame = self.thres_fil.process(depth_frame)
                depth_frame = self.depth_to_disparity.process(depth_frame)
                depth_frame = self.spat_filter.process(depth_frame)
                depth_frame = self.temp_filter.process(depth_frame)
                depth_frame = self.disparity_to_depth.process(depth_frame)
                depth_frame = depth_frame.as_depth_frame()
                
                # -----------------------------------------------------------
                # get depth image after filter processing
                depth_image = np.asanyarray(depth_frame.get_data())

                # EstimaterインスタンスにRGB画像を入力
                self.estimater.pose_estimate(color_image_RGB)

                # 姿勢推定のプロットを行った画像を取得する
                color_img_pose_mapped_RGB = self.estimater.poseimg

                # 姿勢推定のプロットを行った画像を取得する
                color_img_pose_mapped_BGR = cv2.cvtColor(
                    color_img_pose_mapped_RGB, cv2.COLOR_RGB2BGR)

                # -----------------------------------------------------------
                # 三次元データ化 (下に続く)
                # TODO 推定した画素位置のデータから、三次元座標を求めるオブジェクトを作成する.
                # Estimate Pose Classを作成して、Estimate point オブジェクトのリストを持たせる。
                # 欲しいものは、pixelを入れたら3Dpointで変換するobject.
                # listにも変換できるし. 名前も入れることができる.
                # それがあれば、他の画素に関して欲しくなっても計算できる。
                # -----------------------------------------------------------

                # 3次元の姿勢推定&データ取得が完了したらオブジェクトにまとめる
                # 各所処理はpose3dにまとめ、引数を入れて三次元姿勢オブジェクトが返ってくる様にする.
                # Bodypart3Dを一個ずつ生成して、pose3Dに入れて, それをmovementに渡す。
                # Estimaterは画像を受け取って二次元座標を返すことだけをrs_moduleからは知覚できるようにする.
                if self.estimater.results.pose_landmarks:
                    print("------")
                    print("Pose detected.")
                    print("------")

                    # 推定した部位の二次元画素位置を抽出
                    pose2Dpixels = self.estimater.get_pose_pixels2D()
                    # 画像内に存在するかクリッピング処理
                    pose2Dpixelscliped = [pixel_clip(pixel, width, height)
                                          for pixel in pose2Dpixels]
                    # 整数化
                    pose2Dpixelstoint = [pixel_to_int(pixel)
                                         for pixel in pose2Dpixelscliped]
                    # デプスデータ取得
                    # TODO depthデータが0になる部分があるので対応する.
                    pixelsdepth = [self.get_pixel_depth(
                        pixel, depth_frame) for pixel in pose2Dpixelstoint]

                    point_3D = self.get3Dpoints(
                        pixelsdepth=pixelsdepth, pixels=pose2Dpixelstoint)

                    # 三次元姿勢オブジェクトのインスタンス
                    pose3d = p3d.Pose3D(point_3D)

                    # 三次元姿勢オブジェクトにmetadata objectを入れる
                    pose3d.set_metadata(deepcopy(meta))

                    # Movement オブジェクトにPose3Dオブジェクトを追加する.
                    self.move3d.append_pose(pose3d)

                    for i, point in enumerate(point_3D):
                        print(
                            f"part {i}, {self.estimater.poselandmarks(i).name} : {point}")

                else:
                    print("------")
                    print("Pose not detected.")
                    # 姿勢がない時のオブジェクトインスタンスの作成
                    emptypose3d = p3d.EmptyPose3D(markernum)

                    # 三次元姿勢オブジェクトにmetadata objectを入れる
                    emptypose3d.set_metadata(deepcopy(meta))

                    # movementオブジェクトへの追加
                    self.move3d.append_pose(emptypose3d)
                    print("------")

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                images = np.hstack((color_img_pose_mapped_BGR, depth_colormap))

                cv2.imshow("Camera Stream", images)
                key = cv2.waitKey(1)

                # if pressed escape exit program
                if key == 27 or key == ord('q'):
                    cv2.destroyAllWindows()
                    break

                # プログラム内カウンタ
                count += 1

        except Exception:
            import traceback
            print(traceback.format_exc())
            cv2.destroyAllWindows()
        finally:
            cv2.destroyAllWindows()

    def _get_3dpoint_location(self, depth_value: float, horizontal_pix: int, verticle_pix: int) -> tuple[float, float, float]:
        if depth_value == -1:
            # depth valueが不明(-1)の時の処理
            return tuple([-1, -1, -1])
        else:
            # depth valueが存在する時の処理
            point = rs2.rs2_deproject_pixel_to_point(
                self.color_intr, [horizontal_pix, verticle_pix], depth_value)
            return tuple(point)

    # 以下二つのメソッドは別のオブジェクトに入れるべきか

    def get3Dpoints(self,
                    pixelsdepth: list[float],
                    pixels: list[list[int]]) -> list[tuple[float, float, float]]:
        ret = [
            self._get_3dpoint_location(depth,
                                       pixels[i][0],
                                       pixels[i][1]
                                       ) for i, depth in enumerate(pixelsdepth)
        ]
        return ret

    def get_pixel_depth(self,
                        pixel: list[int],
                        depthframe) -> Union[int, float]:
        if not check_pixel_invalid(pixel):
            # 無効な場合, 距離-1を渡す
            return -1
        else:
            # 有効な場合, APIを用いて画素の距離を計算する
            return depthframe.get_distance(pixel[0], pixel[1])

from configparser import ConfigParser
from poseestimate_mediapipe.module import rs_module as rsm
from poseestimate_mediapipe.module.estimater import MpEstimater
from poseestimate_mediapipe.module.movement import Movement


class ProcessBag():
    """bagfileを受け取り三次元姿勢推定を制御するクラス.
    """

    def __init__(self, bagfilepath: str, config) -> None:
        # TODO コンストラクタ引数は今後変更する可能性あり。
        from pathlib import Path
        if Path(bagfilepath).exists() and bagfilepath:
            self._filepath = bagfilepath
        else:
            raise FileNotFoundError(
                "bagfileのパスが存在しません。有効なbagfileのパスを指定してください.")

        self._filename = Path(self._filepath).stem
        self.rs_process = rsm.StereoProcessing()
        self.rs_process.setconfig(self._filepath, config=config)

        # TODO correctposeはここでインスタンス化して、StereoProcessingクラスに注入する

        # instanciate MediaPipe obj
        # TODO confidenceの値を変えたいが、Processs bag内でそれを行うのは正解か?
        mp_pose = MpEstimater()
        mp_pose.set_config(min_detection_confidence=0.3,
                           min_tracking_confidence=0.3)

        # rs processにEstimaterをセットする
        self.rs_process.estimater_slot(mp_pose)

    def estimate(self) -> None:
        self.rs_process.run()

    def generate3Dmovement(self) -> Movement:
        # stereoProcessingクラスからMovementオブジェクトを返す
        # TODO 要検討, 階層が問題になりそう
        return self.rs_process.generate3Dmovement()

    @property
    def csvfilename(self) -> str:
        return f"{self._filename}.csv"

    @property
    def mp4filename(self) -> str:
        return f"{self._filename}.mp4"

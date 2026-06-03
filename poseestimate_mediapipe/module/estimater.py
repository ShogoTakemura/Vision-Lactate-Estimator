# 今後別の形でEstimaterが必要になった時はダックタイピングで対応

class MpEstimater:

    def __init__(self) -> None:
        import mediapipe as mp
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

    def set_config(self,
                   static_img_mode=False,
                   model_complex=2,
                   smooth_segmentation=True,
                   smooth_landmarks=True,
                   enable_segmentation=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5) -> None:

        # pose instance
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_img_mode,
            model_complexity=model_complex,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def pose_estimate(self, img) -> None:
        import numpy as np
        if not isinstance(img, np.ndarray):
            raise TypeError(f"img type is np.ndarray, now {type(img)}")

        self.results = self.pose.process(img)
        self._img = img

        self._draw_results()

    def draw_results(self, img) -> None:
        self.mp_drawing.draw_landmarks(
            img,
            self.results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

    def _draw_results(self) -> None:
        self.mp_drawing.draw_landmarks(
            self._img,
            self.results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

    def get_pose_pixels2D(self) -> list[list[float]]:
        # 推定した姿勢中の全ての画素値を抽出する関数
        height, width, _ = self._img.shape

        return [self.landmark_pixel_2D(eachlandmark, width, height)
                for eachlandmark in self.mp_pose.PoseLandmark]

    def landmark_pixel_2D(self,
                          landmarkname,
                          width: int,
                          height: int) -> list[float]:
        # ある一つのPose Landmarkの画素位置を抽出する関数
        pixel_x = self.results.pose_landmarks.landmark[landmarkname].x * width
        pixel_y = self.results.pose_landmarks.landmark[landmarkname].y * height
        return [pixel_x, pixel_y]

    @property
    def poseimg(self):
        from copy import deepcopy
        return deepcopy(self._img)

    @property
    def poselandmarks(self):
        return self.mp_pose.PoseLandmark

# 今後別の形でEstimaterが必要になった時はダックタイピングで対応

class MpEstimater:

    def __init__(self) -> None:
        import mediapipe as mp
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self._roi_offset: tuple[int, int] = (0, 0)
        self._full_img = None

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

    def pose_estimate(
        self,
        img,
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        import numpy as np
        if not isinstance(img, np.ndarray):
            raise TypeError(f"img type is np.ndarray, now {type(img)}")

        self._full_img = img

        if roi is not None:
            x0, y0, x1, y1 = roi
            self._roi_offset = (x0, y0)
            self._img = img[y0:y1, x0:x1].copy()
        else:
            self._roi_offset = (0, 0)
            self._img = img

        self.results = self.pose.process(self._img)
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
        # 推定した姿勢中の全ての画素値を抽出する関数（ROI オフセットを加算してフル画像座標に戻す）
        height, width, _ = self._img.shape
        ox, oy = self._roi_offset
        pixels = [
            self.landmark_pixel_2D(lm, width, height)
            for lm in self.mp_pose.PoseLandmark
        ]
        return [[px + ox, py + oy] for px, py in pixels]

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
        import cv2
        from copy import deepcopy
        ox, oy = self._roi_offset
        if ox == 0 and oy == 0:
            return deepcopy(self._img)
        # アノテーション済みクロップをフル画像に貼り直してROI矩形を描画
        full = self._full_img.copy()
        crop_h, crop_w = self._img.shape[:2]
        full[oy:oy + crop_h, ox:ox + crop_w] = self._img
        cv2.rectangle(full, (ox, oy), (ox + crop_w, oy + crop_h), (0, 255, 0), 2)
        return full

    @property
    def poselandmarks(self):
        return self.mp_pose.PoseLandmark

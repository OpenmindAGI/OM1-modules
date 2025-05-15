import logging
import platform
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# from om1_utils import singleton

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)

default_path = Path(__file__).parent
yolov8n_mlpackage_path = default_path / "yolov8n-face.mlpackage"
yolov8n_lite_path = default_path / "yolov8n-face.pt"


# @singleton
class YOLOFaceDetection:
    """
    YOLOFaceDetection is a singleton class that initializes a YOLO model for face detection.
    """

    def __init__(self, conf: float = 0.3, margin: int = 30):
        """
        Initialize the YOLOFaceDetection class.

        Parameters
        ----------
        conf : float, optional
            Confidence threshold for face detection, by default 0.3
        margin : int, optional
            Margin around detected faces, by default 30
        """
        self._model = None
        self._conf = conf
        self._margin = margin

        try:
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                logger.info("Loading YOLO model for macOS ARM64 with CoreML")
                self._model = YOLO(yolov8n_mlpackage_path)
            else:
                logger.info("Loading YOLO model for other platforms")
                self._model = YOLO(yolov8n_lite_path)
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect faces in the given frame using the YOLO model.

        Parameters
        ----------
        frame : np.ndarray
            The input frame in which to detect faces.

        Returns
        -------
        np.ndarray
            The frame with detected faces blurred.
        """
        if self._model is None:
            logger.error("YOLO model is not initialized.")
            return frame

        if frame is None:
            logger.error("Input frame is None.")
            return frame

        results = self._model.predict(frame, conf=self._conf, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = max(0, x1 - self._margin)
                y1 = max(0, y1 - self._margin)
                x2 = min(frame.shape[1], x2 + self._margin)
                y2 = min(frame.shape[0], y2 + self._margin)
                face_roi = frame[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
                frame[y1:y2, x1:x2] = blurred
        return frame

from om1_vlm.blur import YOLOFaceDetection


def test_yolo_face_detection_init():
    """
    Test initialization of YOLOFaceDetection
    """
    yolo_face_detection = YOLOFaceDetection(conf=0.5, margin=20)
    assert yolo_face_detection._conf == 0.5
    assert yolo_face_detection._margin == 20
    assert yolo_face_detection._model is not None

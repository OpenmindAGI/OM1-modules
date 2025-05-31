import base64
import logging
import time

import cv2
import numpy as np
from blur import YOLOFaceDetection
from flask import Flask, jsonify, request
from flask_cors import CORS

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

app = Flask(__name__)
CORS(app)


@app.errorhandler(404)
def resource_not_found(e):
    """Return a custom 404 error."""
    return jsonify(error="Root not found!"), 404


yolo_face_detection = YOLOFaceDetection(conf=0.5, margin=20)

# @app.route("/blur", methods=["POST"])
# def blur():
#     data = request.get_json()
#     if not data or "image" not in data:
#         return jsonify(error="No image provided"), 400

#     image = data["image"]
#     image = np.array(image, dtype=np.uint8)

#     try:
#         frame = yolo_face_detection.detect(image)
#         return jsonify({"status": "success", "blurred_image": frame.tolist()})
#     except Exception as e:
#         logger.error(f"Error processing image: {e}")
#         return jsonify(error="Error processing image"), 500


@app.route("/blur", methods=["POST"])
def blur():
    """Process image with efficient base64 encoding instead of JSON arrays"""
    start_time = time.perf_counter()
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify(error="No image provided"), 400

    try:
        # Check if the image is base64 encoded or a list
        if isinstance(data["image"], str):
            # Decode base64 image (efficient)
            img_bytes = base64.b64decode(data["image"])
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Handle legacy list format (inefficient)
            logger.warning("Received image as list - this is inefficient!")
            image = np.array(data["image"], dtype=np.uint8)

        # Process the image
        frame = yolo_face_detection.detect(image)

        # Return base64 encoded result (efficient)
        quality = data.get("quality", 90)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(".jpg", frame, encode_param)
        img_str = base64.b64encode(buffer).decode("utf-8")

        end_time = time.perf_counter()
        logger.info(f"Server processing took {end_time - start_time:.3f} seconds")

        return jsonify({"status": "success", "blurred_image": img_str})
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify(error=f"Error processing image: {str(e)}"), 500


def main():
    app.run(host="0.0.0.0", port=1234, debug=False)


if __name__ == "__main__":
    main()

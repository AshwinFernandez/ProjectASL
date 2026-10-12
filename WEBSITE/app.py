"""
Flask app — Real-Time YOLOv11 ASL detection server.
Supports both MJPEG streaming (/video) and JSON REST inference (/predict).
"""

import io
import os
import base64
import argparse
import cv2
from PIL import Image
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "Models/bestBB.pt" if os.path.exists("Models/bestBB.pt") else "bestBB.pt"
print(f"Loading YOLOv11 model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)


# ── Video streaming (Local camera) ─────────────────────────────────────────────
def gen():
    """Yield MJPEG frames annotated with YOLO detections."""
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)
            for result in results:
                frame = result.plot()

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()


# ── Routes ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "classes": model.names
    })


@app.route("/video")
def video():
    """Stream the annotated webcam feed as MJPEG."""
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict ASL bounding boxes and classes from incoming image."""
    try:
        image = None
        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream).convert("RGB")
        elif request.is_json and "image" in request.json:
            raw_b64 = request.json["image"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if image is None:
            return jsonify({"error": "No image provided"}), 400

        img_w, img_h = image.size
        conf_thresh = float(request.args.get("conf", 0.35))
        results = model(image, conf=conf_thresh, verbose=False)

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                cls_name = result.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "box_normalized": [
                        round(x1 / img_w, 4),
                        round(y1 / img_h, 4),
                        round(x2 / img_w, 4),
                        round(y2 / img_h, 4)
                    ]
                })

        return jsonify({
            "status": "success",
            "detections": detections,
            "count": len(detections)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignAI — YOLOv11 ASL Detection Server")
    parser.add_argument("--port", default=5000, type=int, help="Port to listen on (default: 5000)")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port, debug=False)

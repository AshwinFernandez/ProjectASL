"""
Flask app — YOLOv11 ASL real-time detection streamed over HTTP.

Usage:
    python app.py [--port PORT]
"""

import argparse
import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# ── Model ──────────────────────────────────────────────────────────────────────
model = YOLO("Models/bestBB.pt")


# ── Video streaming ─────────────────────────────────────────────────────────────
def gen():
    """Yield MJPEG frames annotated with YOLO detections."""
    cap = cv2.VideoCapture(0)  # 0 = default webcam; 1 = external webcam

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run inference and annotate frame with bounding boxes
            results = model(frame)
            for result in results:
                frame = result.plot()

            # Encode to JPEG and yield as multipart boundary
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


@app.route("/video")
def video():
    """Stream the annotated webcam feed as MJPEG."""
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignAI — YOLOv11 ASL Detection Server")
    parser.add_argument("--port", default=5000, type=int, help="Port to listen on (default: 5000)")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port, debug=False)

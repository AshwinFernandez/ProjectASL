import io
import os
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
# Enable CORS for all domains so Cloudflare Pages can make requests without restrictions
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the real YOLOv11 ASL model
MODEL_PATH = os.environ.get("MODEL_PATH", "bestBB.pt")
print(f"Loading YOLOv11 ASL model from: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print(f"Model loaded successfully. Classes: {model.names}")


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify backend status and model metadata."""
    return jsonify({
        "status": "online",
        "service": "SignAI YOLOv11 ASL Detection API",
        "classes": model.names,
        "total_classes": len(model.names) if hasattr(model, 'names') else 0
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Run inference on an incoming frame from the frontend.
    Accepts:
      - Multipart file upload: key 'image' or 'file'
      - JSON payload with base64 data URL: { "image": "data:image/jpeg;base64,..." }
    Returns:
      JSON with bounding boxes, predicted sign classes, and confidence scores.
    """
    try:
        image = None

        # 1. Check for file upload
        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream).convert("RGB")
        elif "file" in request.files:
            file = request.files["file"]
            image = Image.open(file.stream).convert("RGB")

        # 2. Check for JSON / base64 payload
        elif request.is_json and "image" in request.json:
            raw_b64 = request.json["image"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if image is None:
            return jsonify({"error": "No valid image provided"}), 400

        img_w, img_h = image.size

        # Parse confidence threshold from query or form
        conf_thresh = float(request.args.get("conf", request.form.get("conf", 0.35)))

        # Run YOLO inference
        results = model(image, conf=conf_thresh, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                cls_name = result.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())

                # Normalized coordinates (0.0 to 1.0) for easy client-side scaling
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "box_pixels": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "box_normalized": [
                        round(x1 / img_w, 4),
                        round(y1 / img_h, 4),
                        round(x2 / img_w, 4),
                        round(y2 / img_h, 4)
                    ]
                })

        return jsonify({
            "status": "success",
            "image_width": img_w,
            "image_height": img_h,
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting SignAI API server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

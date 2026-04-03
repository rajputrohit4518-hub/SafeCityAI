from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/safecity_model4/weights/best.pt',
                       force_reload=True)

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img_bytes = file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)

    detections = []
    for *box, conf, cls in results.xyxy[0]:
        detections.append({
            "class": model.names[int(cls)],
            "confidence": float(conf),
            "box": [int(x) for x in box]
        })

    return jsonify(detections)

if __name__ == '__main__':
    app.run(debug=True)
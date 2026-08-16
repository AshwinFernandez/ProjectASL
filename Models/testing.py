from ultralytics import YOLO
model = YOLO('/Users/ashwinfernandez/Projects/ASL/best.pt')
results = model (1, show= True)

for result in results:
    boxes = result.boxes
    classes = result.names